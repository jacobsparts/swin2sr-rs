//! The device backend: the same graph as `src/cpu.rs`, launched as kernels.
//!
//! NEITHER BACKEND IS THE STANDARD FOR THE OTHER: the reference is the published
//! PyTorch network, and this file is judged against it directly (`--verify`, and
//! `tests/parity.rs`, with the same tolerance the CPU backend is held to).
//! `--cuda-selftest` compares each kernel against its CPU twin as well, but that
//! check can only catch a kernel that does not do what this file says it does; it
//! cannot catch a graph this file transcribed wrongly, because both backends would
//! be wrong together.
//!
//! WHY THE GRAPH IS WRITTEN TWICE. The alternative is one device/mode-abstracted
//! graph, and it was rejected deliberately: the two have different natural
//! shapes (the CPU version indexes slices, this one launches grids) and a shared
//! abstraction would hide exactly the facts a reader needs - which buffer is
//! [c][hp][wp] and which is [tokens][c], and where the shift is applied. What
//! keeps them in step is `tests/parity.rs` running the SAME fixture through both
//! and the fact that the ops come in identical order, stated identically.
//!
//! BUFFERS ARE FLAT AND NAMED AFTER THE GRAPH, not pooled: the failure this
//! avoids is two roles sharing one allocation, which on the device is silent (the
//! buffers are already zeroed) and shows up far from the launch that caused it -
//! the same bug the CPU side had with `res`.
use std::collections::HashMap;

use lightgpu::vm::{Args, DevBuf, Launch};

use lightgpu::ops::cpu as lg;

use crate::backend::Backend;
use crate::cpu;
use crate::cuda::{grid_for, Cuda};
use crate::plan::{pad_reflect, Plan};
use crate::weights::{Upsampler, Weights};

const BLOCK: usize = 256;
/// Columns of the fused head's kernel each thread accumulates (`SS_SHS_COLS`).
const SS_SHS_COLS: usize = 4;
const LN_EPS: f32 = 1e-5;

/// Every device buffer one forward needs, sized for a `Plan`. Same fields, same
/// roles as `cpu::Scratch`.
struct Acts {
    /// The image-layout activation: [c][hp][wp].
    cur: DevBuf,
    /// The block's running residual.
    res: DevBuf,
    /// The stage's input, held for RSTB's own `+ x`. It cannot be `res`: that one
    /// is overwritten by every block, so the value the stage's convs have to be
    /// added to would be gone.
    stage: DevBuf,
    /// Image-layout scratch (a conv's destination, the attention scatter's).
    acc: DevBuf,
    /// conv_first's output, held for the body's skip connection.
    body: DevBuf,
    tok: DevBuf,
    tok2: DevBuf,
    attn: DevBuf,
    /// [3][tokens][c]: the fused projection the attention kernel reads.
    qkv: DevBuf,
    mlp: DevBuf,
}

impl Acts {
    fn new(cuda: &Cuda, plan: &Plan, mlp_ratio: usize) -> Result<Acts, String> {
        let plane = plan.plane() * plan.c;
        let tok = plan.tokens() * plan.c;
        Ok(Acts {
            cur: cuda.buf(plane)?,
            res: cuda.buf(plane)?,
            stage: cuda.buf(plane)?,
            acc: cuda.buf(plane)?,
            body: cuda.buf(plane)?,
            tok: cuda.buf(tok)?,
            tok2: cuda.buf(tok)?,
            attn: cuda.buf(tok)?,
            qkv: cuda.buf(3 * tok)?,
            mlp: cuda.buf(tok * mlp_ratio)?,
        })
    }

}

pub struct Gpu<'a> {
    wt: &'a Weights,
    cuda: Cuda,
    /// The checkpoint's tensors that the graph launches against, uploaded once.
    w: HashMap<String, DevBuf>,
    /// `c` zeros, for the kernels that take a bias they must ignore - the key
    /// projection, which the reference gives none, and the gather's optional
    /// normalisation. A device pointer has to be valid even when unused.
    zeros: DevBuf,
    acts: Option<Acts>,
    plan: Option<Plan>,
    /// The last forward's secondary image, for a head that produces one. Held on
    /// the handle rather than added to `Backend::forward`'s return, whose single
    /// plane every existing caller depends on - the same choice the CPU backend made.
    aux: Option<Vec<f32>>,
}

impl<'a> Gpu<'a> {
    pub fn new(wt: &'a Weights) -> Result<Gpu<'a>, String> {
        let cuda = Cuda::new()?;
        let mut w = HashMap::new();
        for name in needed(wt) {
            let host = wt.t(&name);
            // `attn.cpb_pre` IS UPLOADED TRANSPOSED, to [head][query][key]. The
            // kernel's inner loop reads one head's bias plane `n*n` times per query
            // row, and the checkpoint's layout is [query][key][head] - so an
            // untransposed read walks with a stride of `heads` floats and a warp
            // touches 32 cache lines per key to use 6 of the 32 loaded values. The
            // re-indexing is done ONCE here, at upload, where it costs nothing, and
            // the kernel's arithmetic is untouched: same numbers, same order, same
            // accumulation, so the device result is bit-identical to before.
            // `tests/parity.rs`'s golden fixtures pin the whole-file result, so a
            // wrong transpose cannot hide.
            let host = if name.ends_with("attn.cpb_pre") {
                transpose_cpb(host, wt.heads, wt.window * wt.window)
            } else {
                host.to_vec()
            };
            w.insert(name, cuda.upload(&host)?);
        }
        let zeros = cuda.buf(wt.embed.max(wt.window * wt.window))?;
        Ok(Gpu { wt, cuda, w, zeros, acts: None, plan: None, aux: None })
    }

    fn w(&self, name: &str) -> &DevBuf {
        self.w
            .get(name)
            .unwrap_or_else(|| panic!("weight `{name}` was never uploaded - see gpu::needed"))
    }

    // -- the ops ------------------------------------------------------------
    //
    // Each of these is one launch of one kernel, with the grid the element count
    // implies. They are the device half of the functions at the top of
    // `src/cpu.rs`, in the same order, so a reader can go through the two side by
    // side.

    /// The toolkit's `lg_conv3x3_winograd`: F(4,3), 4x4 outputs from a 6x6 patch.
    /// It computes 36 products per 16 outputs where the direct kernel computes 144,
    /// and the two transforms it adds are amortised over the whole channel
    /// reduction, so it is worth ~3x here. It is the same op as
    /// `lg_conv3x3s1p1` - same NCHW layout, same [c_out][c_in][3][3] weights, same
    /// zero-pad-1 convention - so the two are interchangeable and `--verify` is
    /// what says so.
    ///
    /// THREE THINGS THE CALLER MUST GET RIGHT, because the kernel cannot check
    /// them and disagreement is silent (its own comment says so):
    ///
    /// * `ocb` IS ALWAYS 16. It is both the output channels per CTA and the
    ///   stride between weight slots in shared memory, and the threads' own
    ///   channel index is `tid / 16` regardless - so a smaller `ocb` would have
    ///   the upper threads read slots nobody staged.
    /// * `grid.z` is `ceil(c_out / 16)`, and `c_out` need not be a multiple of 16
    ///   (the kernel deactivates the surplus threads).
    /// * The dynamic shared size is `c_chunk * 32 * 36` floats - input tiles plus
    ///   weight slots, both per chunk channel. `C_CHUNK` is what fits twice per SM.
    ///
    /// `act` is 0: none. Every activation in this graph is a separate launch.
    fn conv3x3(&self, inp: &DevBuf, ci: usize, h: usize, wd: usize, name: &str,
               co: usize, out: &DevBuf) -> Result<(), String> {
        const C_CHUNK: usize = 5;
        const OCB: usize = 16;
        let mut a = Args::new();
        a.ptr(inp.ptr).ptr(self.w(&format!("{name}.weight")).ptr)
            .ptr(self.w(&format!("{name}.bias")).ptr).ptr(out.ptr)
            .i32(ci as i32).i32(co as i32).i32(h as i32).i32(wd as i32)
            .i32(C_CHUNK as i32).i32(OCB as i32).i32(0).f32(0.0);
        let grid = (
            (wd.div_ceil(16)) as u32,
            (h.div_ceil(16)) as u32,
            co.div_ceil(OCB) as u32,
        );
        let shared = (C_CHUNK * 32 * 36 * 4) as u32;
        self.cuda.run(
            "lg_conv3x3_winograd",
            Launch::new(grid, (256, 1, 1)).shared(shared),
            &mut a,
        )
    }

    /// The toolkit's `lg_conv1x1`, which is the same kernel this file used to
    /// carry as `ss_conv1x1`: one thread per output element, `c_in` FMAs each. It
    /// is the op the graph's `patch_embed.proj` needs at both levels.
    fn conv1x1(&self, inp: &DevBuf, ci: usize, h: usize, wd: usize, name: &str,
               co: usize, out: &DevBuf) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(inp.ptr).ptr(self.w(&format!("{name}.weight")).ptr)
            .ptr(self.w(&format!("{name}.bias")).ptr).ptr(out.ptr)
            .i32(ci as i32).i32(co as i32).i32(h as i32).i32(wd as i32);
        // `lg_conv1x1_rb`: the same op, register-tiled 4x4, so one block covers a
        // 64x64 patch of (pixel, channel) instead of 256 elements. Needed for the
        // same reason as the linear above - the 1x1 path is 7% of a device run and
        // its grid was `co * h * wd / 256`, i.e. one output element per thread.
        let grid = (grid_for(h * wd, 64).0, grid_for(co, 64).0, 1);
        self.cuda.run("lg_conv1x1_rb", Launch::new(grid, (16, 16, 1)), &mut a)
    }

    /// A [tokens][c] x [c][c] matmul with a named bias. `bias` is looked up under
    /// `<name without .weight>.bias` by the caller's `b` string, because three of
    /// the projections are slices of one fused matrix (`qkv.wq`) whose bias lives
    /// at a different name (`qkv.q_bias`).
    /// The toolkit's `lg_linear`: a 16x16 shared-memory tiled matmul, which is
    /// what the per-thread `ss_linear` this used to call should have been. A
    /// thread there did one output element's `c_in`-long dot product against a
    /// weight row read from global memory with no reuse at all - the same shape
    /// that measured 6 GFLOP/s on the CPU side of this work, and the reason
    /// `ss_linear` was 60% of a forward.
    ///
    /// `out_off` is where in `out` this projection's `rows * co` elements go, in
    /// elements. The toolkit kernel has no offset parameter (its contract is raw
    /// pointers and scalars), so the offset becomes an interior pointer: the fused
    /// qkv's three projections write the three consecutive [tokens][c] regions of
    /// one buffer, which is the layout the attention kernel reads.
    fn linear(&self, x: &DevBuf, rows: usize, ci: usize, wname: &str, bname: Option<&str>,
              co: usize, out: &DevBuf, out_off: usize) -> Result<(), String> {
        let bias = match bname {
            Some(n) => self.w(n).ptr,
            None => self.zeros.ptr,
        };
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(self.w(wname).ptr).ptr(bias).ptr(out.ptr + (out_off * 4) as u64)
            .i32(rows as i32).i32(ci as i32).i32(co as i32);
        // THE REGISTER-BLOCKED FORM. `lg_linear` gives one thread one output
        // element, so its 16x16 block computes a 16x16 tile; `lg_linear_rb` gives
        // one thread a 4x4 sub-tile, so the same 256 threads cover 64x64 outputs
        // and each element of `x` staged into shared memory is reused 4 times
        // more. The two are BIT-IDENTICAL (both accumulate `c` ascending and add
        // the bias last), which is why the swap is safe to make on the strength of
        // the shape alone, and `--cuda-selftest` checks `lg_linear` against
        // `cpu::linear` on the same input as a second opinion.
        //
        // The qkv projection is the shape this is for: `rows` is 1156 tokens and
        // `c_out` is 12, so the 16x16 grid it used to launch - (1, 73) blocks -
        // was spending 16 threads of a 16x16 block per output row and running 73
        // blocks where 19 do.
        let grid = (grid_for(co, 64).0, grid_for(rows, 64).0, 1);
        self.cuda.run("lg_linear_rb", Launch::new(grid, (16, 16, 1)), &mut a)
    }

    fn layer_norm(&self, x: &DevBuf, w: &str, b: &str, y: &DevBuf, c: usize, hw: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(self.w(w).ptr).ptr(self.w(b).ptr).ptr(y.ptr)
            .i32(c as i32).i32(hw as i32).f32(LN_EPS);
        self.cuda.run(
            "lg_channel_layer_norm",
            Launch::new(grid_for(hw, BLOCK), (BLOCK as u32, 1, 1)),
            &mut a,
        )
    }

    fn gelu(&self, x: &DevBuf, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(x.ptr).i32(n as i32);
        self.cuda.run("lg_gelu_erf", Launch::new(grid_for(n, BLOCK), (BLOCK as u32, 1, 1)), &mut a)
    }

    fn lrelu(&self, x: &DevBuf, slope: f32, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(x.ptr).f32(slope).i64(n as i64);
        self.cuda.run("lg_lrelu", Launch::new(grid_for(n, BLOCK), (BLOCK as u32, 1, 1)), &mut a)
    }

    fn copy(&self, src: &DevBuf, dst: &DevBuf, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(src.ptr).ptr(dst.ptr).i64(n as i64);
        self.cuda.run("lg_copy", Launch::new(grid_for(n, BLOCK), (BLOCK as u32, 1, 1)), &mut a)
    }

    /// `dst += src`, elementwise.
    fn add_into(&self, src: &DevBuf, dst: &DevBuf, n: usize) -> Result<(), String> {
        if n > i32::MAX as usize {
            return Err(format!("a {n}-element residual add exceeds the kernels' i32 counts"));
        }
        let mut a = Args::new();
        a.ptr(dst.ptr).ptr(src.ptr).ptr(dst.ptr).i32(n as i32);
        self.cuda.run("lg_add", Launch::new(grid_for(n, BLOCK), (BLOCK as u32, 1, 1)), &mut a)
    }

    fn gather(&self, plan: &Plan, x: &DevBuf, shift: usize,
              norm: Option<(&str, &str)>, tok: &DevBuf) -> Result<(), String> {
        // NULL means "do not normalise". It must be a null pointer, not a zeroed
        // buffer: the kernel's contract is `norm_w == 0`, and a valid pointer to
        // zeros silently takes the normalising branch and writes zero tokens -
        // which is how the first version of this file produced a black image with
        // every self-consistency check passing.
        let (nw, nb) = match norm {
            Some((w, b)) => (self.w(w).ptr, self.w(b).ptr),
            None => (0u64, 0u64),
        };
        let mut a = Args::new();
        a.ptr(x.ptr).ptr(nw).ptr(nb).ptr(tok.ptr)
            .i32(plan.nw as i32).i32(plan.n as i32).i32(plan.nww as i32).i32(plan.win as i32)
            .i32(plan.hp as i32).i32(plan.wp as i32).i32(plan.c as i32).i32(shift as i32)
            .f32(LN_EPS);
        let total = plan.nw * plan.n * plan.c;
        self.cuda
            .run("ss_window_gather", Launch::new(grid_for(total, BLOCK), (BLOCK as u32, 1, 1)), &mut a)
    }

    fn scatter(&self, plan: &Plan, tok: &DevBuf, shift: usize, x: &DevBuf) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(tok.ptr).ptr(x.ptr)
            .i32(plan.nw as i32).i32(plan.n as i32).i32(plan.nww as i32).i32(plan.win as i32)
            .i32(plan.hp as i32).i32(plan.wp as i32).i32(plan.c as i32).i32(shift as i32);
        let total = plan.nw * plan.n * plan.c;
        self.cuda
            .run("ss_window_scatter", Launch::new(grid_for(total, BLOCK), (BLOCK as u32, 1, 1)), &mut a)
    }

    fn attention(&self, plan: &Plan, name: &str, shift: usize, qkv: &DevBuf, out: &DevBuf) -> Result<(), String> {
        if plan.n > 64 {
            return Err(format!(
                "a {}x{} window is {} tokens, more than the attention kernel's logits row \
                 (SS_MAX_N = 64). A window that large is not in any released checkpoint.",
                plan.win, plan.win, plan.n
            ));
        }
        if self.wt.heads > 12 {
            return Err(format!(
                "{} heads is more than the attention kernel's key-norm table \
                 (SS_MAX_HEADS = 12). Not a shape any released checkpoint has.",
                self.wt.heads
            ));
        }
        if self.wt.head_dim > 64 {
            return Err(format!(
                "head_dim {} is more than the attention kernel's register row \
                 (SS_MAX_HD = 64). Not a shape any released checkpoint has.",
                self.wt.head_dim
            ));
        }
        let mut a = Args::new();
        // `name` is the BLOCK (`layers.s.residual_group.blocks.b`); the two tables
        // live one level deeper, under `attn.`, where the converter writes them.
        a.ptr(qkv.ptr).ptr(self.w(&format!("{name}.attn.logit_scale")).ptr)
            .ptr(self.w(&format!("{name}.attn.cpb_pre")).ptr).ptr(out.ptr)
            .i32(plan.nw as i32).i32(plan.n as i32).i32(plan.nww as i32).i32(plan.win as i32)
            .i32(plan.hp as i32).i32(plan.wp as i32)
            .i32(self.wt.heads as i32).i32(self.wt.head_dim as i32).i32(shift as i32);
        // One block per WINDOW, one thread per (query, head): the block is
        // (n, heads) threads, which for this model is 384 - twelve warps, where
        // one-block-per-(window, head) gave two. The work is identical; the
        // difference is whether the SM has anything to run while a dependent chain
        // of FMAs and `expf` calls is in flight, and that difference was 11.6 ms
        // against 0.16 ms of arithmetic per launch.
        let launch = Launch::new((plan.nw as u32, 1, 1), (plan.n as u32, self.wt.heads as u32, 1));
        self.cuda.run("ss_attention", launch, &mut a)
    }

    /// The head's last octave: a 3x3 conv to `4 * feat` channels and the 2x
    /// shuffle that always follows it, in one launch (`ss_conv3x3_shuffle2`).
    ///
    /// WHAT IT REPLACES. `conv3x3` to `4 * feat` channels into a buffer and then
    /// `pixel_shuffle2` out of it - two launches and, at the last octave, the
    /// largest single allocation in the engine (`4 * feat` channels over the whole
    /// padded plane). The CPU backend has done this fusion since its own memory
    /// pass (`cpu::conv3x3_shuffle2`, 152 MB of a 380 MB peak); this is the device
    /// half of the same pair, and the two are held together by the selftest, which
    /// compares this launch against that function.
    ///
    /// THE KERNEL IS ROW-BLOCKED, NOT WINDOW-BLOCKED: one block is one output row,
    /// and each thread writes both the even and the odd column of its output row
    /// (the shuffle interleaves two conv channels). A block also stages the three
    /// rows the conv reads - `y - 1`, `y` and `y + 1`, one shared slot each, with
    /// the missing one at either edge written as ZEROS so its taps contribute
    /// nothing. Clamping the missing row into a neighbour's slot instead (which is
    /// what reusing one slot would do) is NOT the same computation: it counts the
    /// centre row twice at the top edge, and that was the first version of this
    /// kernel.
    ///
    /// `blockDim.x` IS `ceil(wd / 4)` AND NOT `wd`: each thread carries four
    /// strided columns of accumulators (`SS_SHS_COLS` in the kernel), so the block
    /// stays at 270 threads for the widest plane this engine plans (1080) and there
    /// is no column cap to refuse a wide image. One thread per column, which is
    /// what the first version did, needed `2 * wd <= 1024` and refused every
    /// pixel-shuffle checkpoint above 480 pixels of input - a limit the fixtures
    /// could not see, because they are 33 pixels wide.
    ///
    /// The shared stage is `3 * chunk * (wd + 2) + 2` floats, where `chunk` is the
    /// largest channel block whose three rows fit the 48 KB a launch may take
    /// without the driver's opt-in (which this engine's launch layer does not
    /// expose - `lightgpu::vm::Launch` passes a bare `shared` to `cuLaunchKernel`).
    /// The reduction is chunked rather than the rows being kept whole because the
    /// head's last octave is 64 channels wide at 64 columns, 50696 bytes as one
    /// block.
    fn conv3x3_shuffle2(&self, inp: &DevBuf, ci: usize, h: usize, wd: usize, name: &str,
                        feat: usize, out: &DevBuf) -> Result<(), String> {
        // ONE THREAD PER FOUR COLUMNS (SS_SHS_COLS in the kernel), so the block is
        // `ceil(wd / 4)` threads: 270 at the widest plane this engine plans, well
        // inside the 1024 a block has. That is what makes the head fit the widest
        // image at all - with one thread per column the kernel needed
        // `2 * wd <= 1024` and refused every pixel-shuffle checkpoint above 480.
        let block = wd.div_ceil(SS_SHS_COLS).min(1024).max(1) as u32;
        // THE CHANNEL REDUCTION IS CHUNKED, so the three stage rows fit the 48 KB a
        // launch may take without the driver's opt-in (which this engine's launch
        // layer does not expose). The chunk is the LARGEST that fits, because each
        // one costs a `__syncthreads` pair and a re-read of the three rows.
        // THE STAGED ROW IS `SS_SHS_COLS * blockDim.x + 2` COLUMNS, not `wd + 2`:
        // the kernel's j-th accumulator reaches past `wd` whenever the block does
        // not divide it, and those columns are staged as zeros. The host has to
        // size the same row the kernel indexes - sizing it by `wd + 2` makes the
        // kernel read its own stage out of bounds.
        let row = SS_SHS_COLS * block as usize + 2;
        let chunk = ((48 * 1024 / 4 - 2) / (3 * row)).min(ci).max(1);
        let shared = (3 * chunk * row + 2) * 4;
        let mut a = Args::new();
        a.ptr(inp.ptr).ptr(self.w(&format!("{name}.weight")).ptr)
            .ptr(self.w(&format!("{name}.bias")).ptr).ptr(out.ptr)
            .i32(ci as i32).i32(feat as i32).i32(h as i32).i32(wd as i32)
            .i32(chunk as i32);
        let grid = ((2 * h * feat) as u32, 1, 1);
        self.cuda.run(
            "ss_conv3x3_shuffle2",
            Launch::new(grid, (block, 1, 1)).shared(shared as u32),
            &mut a,
        )
    }

    /// Depth-to-space, `F.pixel_shuffle(x, 2)`: the toolkit's `lg_pixel_shuffle`.
    ///
    /// This was the project kernel `ss_pixel_shuffle2` - a FLAT grid with a runtime
    /// divide by 4 - until the forward direction was promoted into the toolkit
    /// (which had only its inverse, `lg_pixel_unshuffle2`, and which is why this
    /// head refused any scale but 2). The promotion ties the fixed-2 kernels of
    /// nafnet-rs and this engine exactly and is 1.8-2.1x the flat-grid form that
    /// hat-rs used to carry, so THE GRID IS PART OF THE CALL: 32x8 with the output
    /// channel in `blockIdx.z`, which is a launch shape a flat `grid_for(total)`
    /// would get wrong.
    fn pixel_shuffle2(&self, inp: &DevBuf, c4: usize, h: usize, wd: usize, out: &DevBuf) -> Result<(), String> {
        let c = c4 / 4;
        let (oh, ow) = (h * 2, wd * 2);
        let mut a = Args::new();
        a.ptr(inp.ptr).ptr(out.ptr).i32(c as i32).i32(h as i32).i32(wd as i32).i32(2);
        let grid = (ow.div_ceil(32) as u32, oh.div_ceil(8) as u32, c as u32);
        self.cuda
            .run("lg_pixel_shuffle", Launch::new(grid, (32, 8, 1)), &mut a)
    }

    /// The toolkit's `lg_upsample2x_nearest`, which is the same op this file used
    /// to carry as `ss_upsample2x_nearest`. Its contract is the integer halving
    /// (`out[y][x] = in[y >> 1][x >> 1]`), which is what PyTorch's
    /// `F.interpolate(scale_factor=2, mode='nearest')` does and what the head needs.
    fn upsample2x(&self, inp: &DevBuf, c: usize, h: usize, wd: usize, out: &DevBuf) -> Result<(), String> {
        let mut a = Args::new();
        a.ptr(inp.ptr).ptr(out.ptr).i32(c as i32).i32(h as i32).i32(wd as i32);
        let oh = h * 2;
        let ow = wd * 2;
        let grid = (grid_for(ow, 32).0, grid_for(oh, 8).0, 1);
        self.cuda.run("lg_upsample2x_nearest", Launch::new(grid, (32, 8, 1)), &mut a)
    }

    // -- the graph ----------------------------------------------------------

    /// One forward pass over an already mean-adjusted plane, [3][h][w] at the
    /// image's own size. The padding is the first step here exactly as it is on
    /// the CPU side - both call `plan::pad_reflect`, so the two backends cannot
    /// disagree about the geometry the reference's padding produces.
    /// Returns the restored image and, for a head that produces one, the secondary
    /// `aux` image - already through `cpu::finish_aux`, exactly as on the CPU side.
    fn forward_plan(&self, plan: &Plan, input: &[f32])
        -> Result<(Vec<f32>, Option<Vec<f32>>), String> {
        let acts = self.acts.as_ref().expect("acts are built with the plan");
        let c = plan.c;
        let hw = plan.plane();
        let tokens = plan.tokens();
        let tok = tokens * c;
        let hidden = self.wt.mlp_ratio * c;

        // 1. conv_first on the padded plane, then the top-level patch_embed.
        let padded = pad_reflect(input, 3, plan.h, plan.w, plan.hp, plan.wp);
        let inp = self.cuda.upload(&padded)?;
        self.conv3x3(&inp, 3, plan.hp, plan.wp, "conv_first", c, &acts.cur)?;
        // The body's skip: conv_first's output, held across all six stages.
        self.copy(&acts.cur, &acts.body, hw * c)?;
        self.conv1x1(&acts.cur, c, plan.hp, plan.wp, "patch_embed.proj", c, &acts.acc)?;
        self.layer_norm(&acts.acc, "patch_embed.norm.weight", "patch_embed.norm.bias", &acts.cur, c, hw)?;

        // 2. The RSTB stages.
        for (s, &depth) in self.wt.depths.iter().enumerate() {
            let (conv, p1x1) = (format!("layers.{s}.conv"), format!("layers.{s}.patch_embed.proj"));
            self.copy(&acts.cur, &acts.stage, hw * c)?;
            self.copy(&acts.cur, &acts.res, hw * c)?;
            for b in 0..depth {
                let shift = plan.shift_for(b);
                let p = format!("layers.{s}.residual_group.blocks.{b}");
                let qkv = format!("{p}.attn.qkv");
                // POST-NORM: the attention reads the raw activation; `norm1` is
                // applied to its output, before the residual add.
                self.gather(plan, &acts.cur, shift, None, &acts.tok)?;
                // The fused qkv as three matmuls whose weight rows are slices of
                // the checkpoint's [3c][c] matrix. q and v carry their biases; k's
                // is the reference's zero column. They land contiguously as
                // [3][tokens][c], which is the order the attention kernel reads.
                self.linear(&acts.tok, tokens, c, &format!("{qkv}.wq"), Some(&format!("{qkv}.q_bias")),
                            c, &acts.qkv, 0)?;
                self.linear(&acts.tok, tokens, c, &format!("{qkv}.wk"), None, c, &acts.qkv, tok)?;
                self.linear(&acts.tok, tokens, c, &format!("{qkv}.wv"), Some(&format!("{qkv}.v_bias")),
                            c, &acts.qkv, 2 * tok)?;
                self.attention(plan, &p, shift, &acts.qkv, &acts.attn)?;
                // The output projection, on the window-major tokens - the last
                // line of the reference's WindowAttention, before window_reverse.
                self.linear(&acts.attn, tokens, c, &format!("{p}.attn.proj.weight"),
                            Some(&format!("{p}.attn.proj.bias")), c, &acts.tok2, 0)?;
                self.scatter(plan, &acts.tok2, shift, &acts.acc)?;
                self.layer_norm(&acts.acc, &format!("{p}.norm1.weight"), &format!("{p}.norm1.bias"),
                                &acts.tok2, c, hw)?;
                self.add_into(&acts.tok2, &acts.res, hw * c)?;
                // x = x + norm2(mlp(x)): the MLP also reads the raw activation.
                self.gather(plan, &acts.res, shift, None, &acts.tok)?;
                self.linear(&acts.tok, tokens, c, &format!("{p}.mlp.fc1.weight"),
                            Some(&format!("{p}.mlp.fc1.bias")), hidden, &acts.mlp, 0)?;
                self.gelu(&acts.mlp, tokens * hidden)?;
                self.linear(&acts.mlp, tokens, hidden, &format!("{p}.mlp.fc2.weight"),
                            Some(&format!("{p}.mlp.fc2.bias")), c, &acts.attn, 0)?;
                self.scatter(plan, &acts.attn, shift, &acts.acc)?;
                self.layer_norm(&acts.acc, &format!("{p}.norm2.weight"), &format!("{p}.norm2.bias"),
                                &acts.tok2, c, hw)?;
                self.add_into(&acts.tok2, &acts.res, hw * c)?;
                self.copy(&acts.res, &acts.cur, hw * c)?;
            }
            // patch_unembed is a reshape, so the blocks' output is already the
            // image layout: 3x3 conv, then the 1x1 patch_embed conv, then the
            // stage's own residual.
            self.conv3x3(&acts.cur, c, plan.hp, plan.wp, &conv, c, &acts.acc)?;
            self.conv1x1(&acts.acc, c, plan.hp, plan.wp, &p1x1, c, &acts.cur)?;
            self.add_into(&acts.stage, &acts.cur, hw * c)?;
        }

        // 3. The final LayerNorm and the body's residual - which is against
        //    CONV_FIRST's output, not the body's.
        let normed = self.cuda.buf(hw * c)?;
        self.layer_norm(&acts.cur, "norm.weight", "norm.bias", &normed, c, hw)?;
        self.conv3x3(&normed, c, plan.hp, plan.wp, "conv_after_body", c, &acts.acc)?;
        self.add_into(&acts.body, &acts.acc, hw * c)?;
        // `acc` is a scratch buffer by contract - a conv's destination, consumed by
        // the next op. The head reads `cur`, exactly as the CPU backend's head
        // reads `scr.cur` after the same copy; leaving this out fed the head the
        // LAST STAGE's output, which is a plausible-looking image and wrong
        // everywhere.
        self.copy(&acts.acc, &acts.cur, hw * c)?;

        // 4. The reconstruction head, then the same host epilogue the CPU backend
        //    runs (`cpu::finish`): the denormalisation and the crop are host
        //    arithmetic either way, and one copy of them cannot disagree with
        //    itself.
        // The head leaves its result on the device (the two shuffle heads) or in
        // host memory (the direct head's is a device buffer too - see `head`), so
        // the one download happens here and the epilogue is the CPU backend's.
        // The compressed head's OTHER input, computed here because `padded` is here:
        // a host bicubic resample to the PADDED output grid, uploaded, then one
        // conv3x3 on the device. See the long note in the arm for why the resample
        // is host arithmetic and why the geometry is the padded one.
        let bicubic = if self.wt.upsampler == Upsampler::PixelShuffleAux {
            let feat = self.wt.t("conv_bicubic.weight").len() / (3 * 9);
            let (oh, ow) = (plan.hp * self.wt.scale, plan.wp * self.wt.scale);
            let mut resized = vec![0.0f32; 3 * oh * ow];
            cpu::bicubic_resize(&padded, 3, plan.hp, plan.wp, oh, ow, &mut resized);
            let dres = self.cuda.upload(&resized)?;
            let bic = self.cuda.buf(feat * oh * ow)?;
            self.conv3x3(&dres, 3, oh, ow, "conv_bicubic", feat, &bic)?;
            Some(bic)
        } else {
            None
        };
        let (dev, dev_aux) = self.head(plan, &acts, bicubic.as_ref())?;
        let mut planes = vec![0.0f32; dev.bytes / 4];
        dev.download(&mut planes)?;
        // The aux gets its OWN epilogue - `x / img_range + mean`, and no crop, since
        // it is emitted on the padded plane. That is `cpu::finish_aux`, the same
        // function the CPU backend uses: the device produces the activations, the
        // host applies the denormalisation, and there is one copy of it.
        let aux = match &dev_aux {
            Some(d) => {
                let mut host = vec![0.0f32; d.bytes / 4];
                d.download(&mut host)?;
                Some(cpu::finish_aux(self.wt, plan, &host))
            }
            None => None,
        };
        Ok((cpu::finish(self.wt, plan, &planes, plan.wp * self.wt.scale), aux))
    }

    /// The reconstruction head. Returns the main output and, for the compressed
    /// head, the SECOND image it produces - both device buffers, downloaded by the
    /// caller, which then runs the same host epilogue the CPU backend does.
    /// `bicubic` is the compressed head's pre-upsample branch, already on the device
    /// at `feat` channels over the padded output grid - computed by `forward_plan`,
    /// which is where the padded input plane lives. `None` for every other head.
    fn head(&self, plan: &Plan, acts: &Acts, bicubic: Option<&DevBuf>)
        -> Result<(DevBuf, Option<DevBuf>), String> {
        let wt = self.wt;
        let c = plan.c;
        let hw = plan.plane();
        let scale = wt.scale;
        // `feat` belongs to the pixel-shuffle and nearest+conv heads' FIRST layer
        // and does not exist in the direct one - read it per branch, as the CPU
        // backend does.
        match wt.upsampler {
            Upsampler::PixelShuffle => {
                let feat = wt.t("conv_before_upsample.0.weight").len() / (c * 9);
                let a = self.cuda.buf(feat * hw)?;
                self.conv3x3(&acts.cur, c, plan.hp, plan.wp, "conv_before_upsample.0", feat, &a)?;
                self.lrelu(&a, 0.01, feat * hw)?;
                let (mut h2, mut w2) = (plan.hp, plan.wp);
                let mut cur = a;
                let cur_c = feat;
                for o in 0..wt.upsampler.octaves(scale) {
                    // The conv and the shuffle as one launch, so the `4 * feat`
                    // intermediate - the largest allocation the head had - is never
                    // materialised. See `conv3x3_shuffle2`.
                    let shuf = self.cuda.buf(cur_c * 4 * h2 * w2)?;
                    self.conv3x3_shuffle2(&cur, cur_c, h2, w2, &format!("upsample.{}", 2 * o),
                                          cur_c, &shuf)?;
                    cur = shuf;
                    h2 *= 2;
                    w2 *= 2;
                }
                let planes = self.cuda.buf(3 * h2 * w2)?;
                self.conv3x3(&cur, cur_c, h2, w2, "conv_last", 3, &planes)?;
                Ok((planes, None))
            }
            Upsampler::PixelShuffleDirect => {
                let out_ch = 3 * scale * scale;
                let up = self.cuda.buf(out_ch * hw)?;
                self.conv3x3(&acts.cur, c, plan.hp, plan.wp, "upsample.0", out_ch, &up)?;
                // The one-step head's shuffle is scale x scale, and the kernel
                // behind it - the toolkit's `lg_pixel_shuffle` - takes the factor
                // as a RUNTIME argument, so the device could run any of them. The
                // guard stays because the FACTOR IS NOT WHAT IS UNVERIFIED: the
                // head's own plan, its `3*scale*scale` conv and its output geometry
                // have only ever been exercised at 2, and forwarding a differently
                // shaped plane into the png writer is what a silent wrong answer
                // would look like. The CPU backend's `pixel_shuffle` is general, so
                // this is a limit of SCOPE rather than of kernels - lifting it is
                // `c = out_ch / (scale * scale)` below once a checkpoint needs it.
                if scale != 2 {
                    return Err(format!(
                        "the pixelshuffledirect head at scale {scale} is not verified on the \
                         device: the op itself is general (lg_pixel_shuffle takes r), but this \
                         head has only ever been run at 2, and the released lightweight \
                         checkpoint is x2"
                    ));
                }
                let planes = self.cuda.buf(3 * plan.hp * scale * plan.wp * scale)?;
                self.pixel_shuffle2(&up, out_ch, plan.hp, plan.wp, &planes)?;
                Ok((planes, None))
            }
            Upsampler::NearestConv => {
                let feat = wt.t("conv_before_upsample.0.weight").len() / (c * 9);
                let a = self.cuda.buf(feat * hw)?;
                self.conv3x3(&acts.cur, c, plan.hp, plan.wp, "conv_before_upsample.0", feat, &a)?;
                // 0.01 for the body's activation, 0.2 for the three convs on the
                // way out: both slopes are in the reference and neither is the
                // other.
                self.lrelu(&a, 0.01, feat * hw)?;
                let (mut h2, mut w2) = (plan.hp, plan.wp);
                let mut cur = a;
                for step in 0..2 {
                    let up = self.cuda.buf(feat * 4 * h2 * w2)?;
                    self.upsample2x(&cur, feat, h2, w2, &up)?;
                    h2 *= 2;
                    w2 *= 2;
                    let next = self.cuda.buf(feat * h2 * w2)?;
                    let name = if step == 0 { "conv_up1" } else { "conv_up2" };
                    self.conv3x3(&up, feat, h2, w2, name, feat, &next)?;
                    self.lrelu(&next, 0.2, feat * h2 * w2)?;
                    cur = next;
                }
                let hr = self.cuda.buf(feat * h2 * w2)?;
                self.conv3x3(&cur, feat, h2, w2, "conv_hr", feat, &hr)?;
                self.lrelu(&hr, 0.2, feat * h2 * w2)?;
                let planes = self.cuda.buf(3 * h2 * w2)?;
                self.conv3x3(&hr, feat, h2, w2, "conv_last", 3, &planes)?;
                Ok((planes, None))
            }
            // The compressed head: the classical one with a bicubic shortcut around it
            // and a second output image. See `cpu.rs` for the same graph written as
            // slices - the ops are in the same order, deliberately.
            //
            // THE BICUBIC RESAMPLE ITSELF RUNS ON THE HOST, and only its 3-channel
            // result is uploaded. There is no device kernel for it and inventing one
            // for a branch that reads three channels would cost more than it saves;
            // `cpu::bicubic_resize` is the same function `tests/bicubic.rs` holds to
            // torch to 7.8e-6, so the device runs the checked implementation rather
            // than a second transcription of it. Everything downstream - both convs,
            // the two octaves and the add - is on the device as usual.
            Upsampler::PixelShuffleAux => {
                let feat = wt.t("conv_bicubic.weight").len() / (3 * 9);
                // The PADDED output grid: see the long note in `cpu.rs`. The
                // reference's `H, W` are the dims of the plane it was HANDED, and the
                // task wrapper hands it the padded one, so its own crop is a no-op
                // and the octaves' `conv_last` runs at this size too.
                let bic = bicubic.expect("the compressed head needs the bicubic branch - see forward_plan");

                let a = self.cuda.buf(feat * hw)?;
                self.conv3x3(&acts.cur, c, plan.hp, plan.wp, "conv_before_upsample.0", feat, &a)?;
                // 0.01, the `nn.LeakyReLU` default - not the 0.2 the real-world head's
                // convs use. The two heads differ here and both slopes are in the
                // reference.
                self.lrelu(&a, 0.01, feat * hw)?;
                // The SECOND output, off the padded plane: three channels, and the
                // only tensor the aux branch reads.
                let aux = self.cuda.buf(3 * hw)?;
                self.conv3x3(&a, feat, plan.hp, plan.wp, "conv_aux", 3, &aux)?;

                let x = self.cuda.buf(feat * hw)?;
                self.conv3x3(&aux, 3, plan.hp, plan.wp, "conv_after_aux.0", feat, &x)?;
                self.lrelu(&x, 0.01, feat * hw)?;

                let (mut h2, mut w2) = (plan.hp, plan.wp);
                let mut cur = x;
                let cur_c = feat;
                for o in 0..wt.upsampler.octaves(scale) {
                    let shuf = self.cuda.buf(cur_c * 4 * h2 * w2)?;
                    self.conv3x3_shuffle2(&cur, cur_c, h2, w2, &format!("upsample.{}", 2 * o),
                                          cur_c, &shuf)?;
                    cur = shuf;
                    h2 *= 2;
                    w2 *= 2;
                }
                // `x = upsample(x) + bicubic`, elementwise over the whole padded
                // grid. `lg_add` is `dst += src` in place, and `cur` is not read
                // again, so the sum lands in the octave buffer - the same thing
                // `cpu.rs` does into `summed`.
                self.add_into(bic, &cur, feat * h2 * w2)?;
                let planes = self.cuda.buf(3 * h2 * w2)?;
                self.conv3x3(&cur, cur_c, h2, w2, "conv_last", 3, &planes)?;
                Ok((planes, Some(aux)))
            }
        }
    }
}

impl Backend for Gpu<'_> {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn forward(&mut self, h: usize, w: usize, input: &[f32]) -> Result<Vec<f32>, String> {
        let plan = match self.plan {
            Some(p) if p.h == h && p.w == w => p,
            _ => {
                let p = Plan::new(h, w, self.wt.window, self.wt.embed);
                self.acts = Some(Acts::new(&self.cuda, &p, self.wt.mlp_ratio)?);
                self.plan = Some(p);
                p
            }
        };
        let (planes, aux) = self.forward_plan(&plan, input)?;
        self.aux = aux;
        Ok(planes)
    }

    fn aux(&self) -> Option<&[f32]> {
        self.aux.as_deref()
    }
}

/// The checkpoint tensors the graph launches against, derived from the
/// architecture in the header rather than listed by hand: a list would drift the
/// first time a stage's depth changed.
/// The bytes `Gpu::new` will hold on the device for this checkpoint: the upload
/// set, four bytes an element. The activation buffers are the run's own and are
/// budgeted separately (`backend::per_pixel_floats`), but the weights are not
/// optional and a host that only knows its free VRAM does not know this yet.
///
/// The cube's own 512 MiB granularity (see `Cuda`) is NOT in here: this is what
/// the tensors are worth, not what the allocator rounds them to. A caller sizing
/// a job wants the tensors; only a caller sizing an allocation wants the cube,
/// and that is a fact about `Cuda`, not about the checkpoint.
/// `[query][key][head]` -> `[head][query][key]`, for the attention bias table.
///
/// `cpu::attention` documents and does this same transpose, per call; on the device
/// it is done ONCE at upload, because the table is a weight and rereading it in the
/// checkpoint's layout costs a stride-`heads` gather in the kernel's innermost loop.
/// A pure re-indexing: the same numbers, in the same order, with the same
/// accumulation, so the device result is bit-identical to the untransposed read.
pub fn transpose_cpb(src: &[f32], heads: usize, n: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; heads * n * n];
    for q in 0..n {
        for k in 0..n {
            for h in 0..heads {
                dst[(h * n + q) * n + k] = src[(q * n + k) * heads + h];
            }
        }
    }
    dst
}

pub fn uploaded_bytes(wt: &Weights) -> u64 {
    let mut total = 0u64;
    for name in needed(wt) {
        let n = wt.shape(&name).iter().product::<usize>() as u64;
        total += n * 4;
    }
    total
}

fn needed(wt: &Weights) -> Vec<String> {
    let mut v = vec![
        "conv_first.weight".into(), "conv_first.bias".into(),
        "conv_after_body.weight".into(), "conv_after_body.bias".into(),
        "norm.weight".into(), "norm.bias".into(),
        "patch_embed.proj.weight".into(), "patch_embed.proj.bias".into(),
        "patch_embed.norm.weight".into(), "patch_embed.norm.bias".into(),
    ];
    // `conv_last` belongs to the two heads that END in it. The direct head's last
    // layer is the conv that produces all 3*scale^2 planes, so listing `conv_last`
    // here makes the engine refuse a checkpoint it can run - see `weights.rs`,
    // where the same distinction had to be made for the shape check.
    for (s, &depth) in wt.depths.iter().enumerate() {
        v.push(format!("layers.{s}.conv.weight"));
        v.push(format!("layers.{s}.conv.bias"));
        v.push(format!("layers.{s}.patch_embed.proj.weight"));
        v.push(format!("layers.{s}.patch_embed.proj.bias"));
        for b in 0..depth {
            let p = format!("layers.{s}.residual_group.blocks.{b}");
            for k in ["attn.qkv.wq", "attn.qkv.wk", "attn.qkv.wv", "attn.qkv.q_bias",
                      "attn.qkv.v_bias", "attn.logit_scale", "attn.cpb_pre", "attn.proj.weight",
                      "attn.proj.bias", "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                      "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"] {
                v.push(format!("{p}.{k}"));
            }
        }
    }
    match wt.upsampler {
        Upsampler::PixelShuffle => {
            v.push("conv_last.weight".into());
            v.push("conv_last.bias".into());
            v.push("conv_before_upsample.0.weight".into());
            v.push("conv_before_upsample.0.bias".into());
            for o in 0..wt.upsampler.octaves(wt.scale) {
                v.push(format!("upsample.{}.weight", 2 * o));
                v.push(format!("upsample.{}.bias", 2 * o));
            }
        }
        Upsampler::PixelShuffleDirect => {
            v.push("upsample.0.weight".into());
            v.push("upsample.0.bias".into());
        }
        Upsampler::NearestConv => {
            for k in ["conv_before_upsample.0.weight", "conv_before_upsample.0.bias",
                      "conv_up1.weight", "conv_up1.bias", "conv_up2.weight", "conv_up2.bias",
                      "conv_hr.weight", "conv_hr.bias", "conv_last.weight", "conv_last.bias"] {
                v.push(k.into());
            }
        }
        // The compressed head: the two pixel-shuffle octaves' weights are the
        // same `upsample.{2o}` names the classical head uses, so it needs the
        // same listing, plus its three extra convs. Listing them here is what
        // makes `Gpu::new` upload them; without it the head would launch against
        // a name `w()` panics on.
        Upsampler::PixelShuffleAux => {
            v.push("conv_before_upsample.0.weight".into());
            v.push("conv_before_upsample.0.bias".into());
            v.push("conv_last.weight".into());
            v.push("conv_last.bias".into());
            for o in 0..wt.upsampler.octaves(wt.scale) {
                v.push(format!("upsample.{}.weight", 2 * o));
                v.push(format!("upsample.{}.bias", 2 * o));
            }
            for k in ["conv_bicubic.weight", "conv_bicubic.bias",
                      "conv_aux.weight", "conv_aux.bias",
                      "conv_after_aux.0.weight", "conv_after_aux.0.bias"] {
                v.push(k.into());
            }
        }
    }
    v
}


// ---------------------------------------------------------------------------
// `--cuda-selftest`: every kernel against its CPU twin.
// ---------------------------------------------------------------------------

/// Each kernel, run once on a small deterministic input, against the function it
/// is a device transcription of.
///
/// WHAT THIS CAN AND CANNOT SHOW. It catches a kernel that does not compute what
/// its CPU twin computes - a wrong tap order, an off-by-one in the window index,
/// a mask that is applied when it should not be. It CANNOT catch a graph this
/// file transcribed wrongly: both backends would be wrong together, and the only
/// thing that sees that is `--verify` / `tests/parity.rs` against the published
/// network's output. That is why the README quotes the fixture's tolerance and
/// this check separately.
pub fn selftest(wt: &Weights) -> Result<(), String> {
    let cuda = Cuda::new()?;
    let mut checks = 0;
    let tol = 2e-5f32;

    // A deterministic input, so a failure that is reported once can be
    // reproduced: x[i] = sin(i * 0.7) - the same generator the CPU selftest uses.
    let seq = |n: usize, phase: f32| -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.7 + phase).sin()).collect()
    };

    {
        // `lg_conv3x3s1p1` against cpu::conv3x3, at the three shapes the graph
        // uses: the wide-to-wide stage conv (180 -> 180), the head's widening one,
        // and the 3-channel conv_first.
        let (h, w) = (7usize, 9usize);
        for (ci, co) in [(4usize, 4usize), (4, 12), (12, 3)] {
            let x = seq(ci * h * w, 0.3);
            let wgt = seq(co * ci * 9, 1.1);
            let bias = seq(co, 2.2);
            let mut want = vec![0.0f32; co * h * w];
            cpu::conv3x3(&x, ci, h, w, &wgt, co, &bias, &mut want);
            let dx = cuda.upload(&x)?;
            let dw = cuda.upload(&wgt)?;
            let db = cuda.upload(&bias)?;
            let dy = cuda.buf(co * h * w)?;
            let total = co * h * w;
            let mut a = Args::new();
            a.ptr(dx.ptr).ptr(dw.ptr).ptr(db.ptr).ptr(dy.ptr)
                .i32(ci as i32).i32(co as i32).i32(h as i32).i32(w as i32);
            cuda.run("lg_conv3x3s1p1", Launch::new(grid_for(total, BLOCK), (BLOCK as u32, 1, 1)), &mut a)?;
            let mut got = vec![0.0f32; total];
            dy.download(&mut got)?;
            close("lg_conv3x3s1p1", &want, &got, tol)?;
            checks += 1;
        }
    }

    {
        // `lg_conv3x3_winograd` against the DIRECT `lg_conv3x3s1p1`, which is
        // itself checked against `cpu::conv3x3` above. The winograd kernel is what
        // the graph launches, and it is the one kernel here whose arithmetic is not
        // a transcription of its CPU twin - the transforms and the 36-products-per-
        // 16-outputs reduction give a different rounding, so the tolerance is the
        // winograd transform's own (~1e-6 relative on these magnitudes) rather than
        // the 2e-5 the rest of this function uses.
        //
        // IT IS CHECKED AT ALL because a wrong transform is not a crash: it is a
        // plausible image, off by a little, and `--verify`'s 2e-3 would not see it.
        let (h, w) = (33usize, 39usize);       // not a multiple of 16: the edge tiles
        for (ci, co) in [(3usize, 16usize), (16, 3), (16, 20)] {
            let x = seq(ci * h * w, 0.3);
            let wgt = seq(co * ci * 9, 1.1);
            let bias = seq(co, 2.2);
            let mut want = vec![0.0f32; co * h * w];
            cpu::conv3x3(&x, ci, h, w, &wgt, co, &bias, &mut want);
            let dx = cuda.upload(&x)?;
            let dw = cuda.upload(&wgt)?;
            let db = cuda.upload(&bias)?;
            let dy = cuda.buf(co * h * w)?;
            const C_CHUNK: usize = 5;
            const OCB: usize = 16;
            let mut a = Args::new();
            a.ptr(dx.ptr).ptr(dw.ptr).ptr(db.ptr).ptr(dy.ptr)
                .i32(ci as i32).i32(co as i32).i32(h as i32).i32(w as i32)
                .i32(C_CHUNK as i32).i32(OCB as i32).i32(0).f32(0.0);
            let grid = (w.div_ceil(16) as u32, h.div_ceil(16) as u32, co.div_ceil(OCB) as u32);
            let shared = (C_CHUNK * 32 * 36 * 4) as u32;
            cuda.run("lg_conv3x3_winograd", Launch::new(grid, (256, 1, 1)).shared(shared), &mut a)?;
            let mut got = vec![0.0f32; co * h * w];
            dy.download(&mut got)?;
            close(&format!("lg_conv3x3_winograd {ci}->{co}"), &want, &got, 2e-4)?;
            checks += 1;
        }
    }

    {
        // `lg_conv1x1` against cpu::conv1x1, at a non-square shape: the kernel
        // indexes its plane as `y * wd + x`, so a square one would not catch a
        // transposed stride.
        let (ci, co) = (6usize, 9usize);
        let (h, w) = (5usize, 8usize);
        let x = seq(ci * h * w, 0.9);
        let wgt = seq(co * ci, 1.7);
        let bias = seq(co, 0.4);
        let mut want = vec![0.0f32; co * h * w];
        let mut ctmp = vec![0.0f32; co * h * w];
        let mut wp = vec![0.0f32; ci * co];
        cpu::conv1x1(&x, ci, h * w, &wgt, co, &bias, &mut want, &mut ctmp, &mut wp);
        let dx = cuda.upload(&x)?;
        let dy = cuda.buf(co * h * w)?;
        let mut a = Args::new();
        a.ptr(dx.ptr).ptr(cuda.upload(&wgt)?.ptr).ptr(cuda.upload(&bias)?.ptr).ptr(dy.ptr)
            .i32(ci as i32).i32(co as i32).i32(h as i32).i32(w as i32);
        cuda.run("lg_conv1x1", Launch::new(grid_for(co * h * w, BLOCK), (BLOCK as u32, 1, 1)), &mut a)?;
        let mut got = vec![0.0f32; co * h * w];
        dy.download(&mut got)?;
        close("lg_conv1x1", &want, &got, tol)?;
        checks += 1;
    }

    {
        // `lg_linear` against cpu::linear, at the shapes the graph uses and at the
        // two ways the graph bends it.
        //
        // `lg_linear` takes no output offset - its contract is raw pointers and
        // scalars - so the fused qkv's three projections reach their three regions
        // of one buffer through an INTERIOR POINTER, and the third case below is
        // that pointer arithmetic. And unlike the `ss_linear` this replaced, it
        // always dereferences `bias`: the key projection's "no bias" is a zero
        // buffer, so the fourth case is that buffer, and it is the check that a
        // caller cannot quietly pass null.
        let (rows, ci, co) = (13usize, 6usize, 5usize);
        let x = seq(rows * ci, 0.2);
        let wgt = seq(co * ci, 2.4);
        let bias = seq(co, 3.1);
        let zeros = vec![0.0f32; co];
        let mut want = vec![0.0f32; rows * co];
        let mut xt = vec![0.0f32; rows * ci];
        let mut wp = vec![0.0f32; ci * co];
        cpu::linear(&x, rows, ci, &wgt, co, &bias, &mut want, &mut xt, &mut wp);
        let mut want_no_bias = want.clone();
        for r in 0..rows {
            for o in 0..co {
                want_no_bias[r * co + o] -= bias[o];
            }
        }
        for (label, bias_ptr, off, expected) in [
            ("lg_linear", cuda.upload(&bias)?.ptr, 0usize, &want),
            ("lg_linear (offset)", cuda.upload(&bias)?.ptr, 11, &want),
            ("lg_linear (zero bias)", cuda.upload(&zeros)?.ptr, 0, &want_no_bias),
        ] {
            let dx = cuda.upload(&x)?;
            let dy = cuda.buf(off + rows * co)?;
            let mut a = Args::new();
            a.ptr(dx.ptr).ptr(cuda.upload(&wgt)?.ptr).ptr(bias_ptr)
                .ptr(dy.ptr + (off * 4) as u64)
                .i32(rows as i32).i32(ci as i32).i32(co as i32);
            let grid = (grid_for(co, 16).0, grid_for(rows, 16).0, 1);
            cuda.run("lg_linear", Launch::new(grid, (16, 16, 1)), &mut a)?;
            let mut all = vec![0.0f32; off + rows * co];
            dy.download(&mut all)?;
            close(label, expected, &all[off..], tol)?;
            checks += 1;
        }
    }

    {
        // `ss_conv3x3_shuffle2` against cpu::conv3x3_shuffle2, which is the SAME
        // fusion written as two passes on the CPU and is itself checked bit-for-bit
        // against `conv3x3` + `pixel_shuffle2` above. Shapes: a non-square one, `h
        // = 1` (the plan's height for a one-pixel image, and the case where two of
        // the conv's three rows do not exist), and a widening one where `feat` is
        // four times `c_in` as the head's last octave is.
        for (ci, feat, h, wd) in [(4usize, 3usize, 5usize, 7usize), (4, 3, 1, 6), (6, 24, 4, 5),
                                  (2, 2, 3, 61), (2, 2, 1, 1025)] {
            let x = seq(ci * h * wd, 0.35);
            let wgt = seq(4 * feat * ci * 9, 1.45);
            let bias = seq(4 * feat, 0.75);
            let mut want = vec![0.0f32; feat * 4 * h * wd];
            cpu::conv3x3_shuffle2(&x, ci, h, wd, &wgt, feat, &bias, &mut want);
            let dx = cuda.upload(&x)?;
            let dy = cuda.buf(want.len())?;
            let mut a = Args::new();
            // The graph's own choice is the largest chunk that fits; the selftest
            // uses a small one as well, so the chunk loop's boundary is exercised
            // rather than only ever running once.
            let chunk = if h == 1 && wd == 6 { 2 } else { ci };
            let block = wd.div_ceil(SS_SHS_COLS).max(1) as u32;
            let row = SS_SHS_COLS * block as usize + 2;
            a.ptr(dx.ptr).ptr(cuda.upload(&wgt)?.ptr).ptr(cuda.upload(&bias)?.ptr).ptr(dy.ptr)
                .i32(ci as i32).i32(feat as i32).i32(h as i32).i32(wd as i32)
                .i32(chunk as i32);
            let shared = (3 * chunk.min(ci) * row + 2) * 4;
            cuda.run(
                "ss_conv3x3_shuffle2",
                Launch::new(((2 * h * feat) as u32, 1, 1), (block, 1, 1)).shared(shared as u32),
                &mut a,
            )?;
            let mut got = vec![0.0f32; want.len()];
            dy.download(&mut got)?;
            close(&format!("ss_conv3x3_shuffle2 {ci}->{feat}x4 at {h}x{wd}"), &want, &got, tol)?;
            checks += 1;
        }
    }

    {
        // The head's pixel shuffle - the toolkit's `lg_pixel_shuffle` since the
        // forward direction was promoted - and lg_upsample2x_nearest, against their
        // cpu twins. Shapes chosen OFF the 32x8 tile (5x6 rows and columns) so the
        // tail guards and the channel-in-blockIdx.z grid are both exercised.
        let (c4, h, w) = (8usize, 5usize, 6usize);
        let x = seq(c4 * h * w, 0.5);
        let mut want = vec![0.0f32; (c4 / 4) * 4 * h * w];
        cpu::pixel_shuffle2(&x, c4, h, w, &mut want);
        let dx = cuda.upload(&x)?;
        let dy = cuda.buf(want.len())?;
        let mut a = Args::new();
        let (c, oh, ow) = (c4 / 4, h * 2, w * 2);
        a.ptr(dx.ptr).ptr(dy.ptr).i32(c as i32).i32(h as i32).i32(w as i32).i32(2);
        let grid = (ow.div_ceil(32) as u32, oh.div_ceil(8) as u32, c as u32);
        cuda.run("lg_pixel_shuffle", Launch::new(grid, (32, 8, 1)), &mut a)?;
        let mut got = vec![0.0f32; want.len()];
        dy.download(&mut got)?;
        close("lg_pixel_shuffle (the head's shuffle)", &want, &got, tol)?;
        checks += 1;

        let c = 4usize;
        let y = seq(c * h * w, 1.3);
        let mut want = vec![0.0f32; c * 4 * h * w];
        cpu::upsample2x_nearest(&y, c, h, w, &mut want);
        let dyin = cuda.upload(&y)?;
        let dout = cuda.buf(want.len())?;
        let mut a = Args::new();
        a.ptr(dyin.ptr).ptr(dout.ptr).i32(c as i32).i32(h as i32).i32(w as i32);
        let grid = (grid_for(w * 2, 32).0, grid_for(h * 2, 8).0, 1);
        cuda.run("lg_upsample2x_nearest", Launch::new(grid, (32, 8, 1)), &mut a)?;
        let mut got = vec![0.0f32; want.len()];
        dout.download(&mut got)?;
        close("lg_upsample2x_nearest", &want, &got, tol)?;
        checks += 1;
    }

    {
        // The window pair, against cpu::gather / cpu::scatter. The three shifts
        // the graph uses are all exercised: 0, win/2 and a non-cyclic one (a
        // shift larger than the window, which the roll wraps).
        //
        // THIS IS THE PAIR THAT CANNOT BE TRUSTED TO AGREE WITH ITSELF - a gather
        // and a scatter wrong in the same way are still mutually inverse. So the
        // check is against the CPU twin's INDEX MAP, not against a round trip.
        for (h, w, win, shift) in [(16usize, 16usize, 8usize, 0usize), (16, 16, 8, 4), (16, 24, 8, 4)] {
            let plan = Plan::new(h, w, win, 4);
            let x = seq(4 * plan.hp * plan.wp, 0.8);
            let mut want = vec![0.0f32; plan.nw * plan.n * 4];
            cpu::gather(&plan, &x, shift, None, &mut want);

            let dx = cuda.upload(&x)?;
            let dtok = cuda.buf(want.len())?;
            let mut a = Args::new();
            a.ptr(dx.ptr).ptr(0).ptr(0).ptr(dtok.ptr)
                .i32(plan.nw as i32).i32(plan.n as i32).i32(plan.nww as i32).i32(plan.win as i32)
                .i32(plan.hp as i32).i32(plan.wp as i32).i32(4).i32(shift as i32).f32(LN_EPS);
            cuda.run("ss_window_gather", Launch::new(grid_for(want.len(), BLOCK), (BLOCK as u32, 1, 1)), &mut a)?;
            let mut got = vec![0.0f32; want.len()];
            dtok.download(&mut got)?;
            close(&format!("ss_window_gather shift={shift}"), &want, &got, tol)?;
            checks += 1;

            // The scatter, from tokens that are NOT the gather's output: with
            // zeros the inverse of anything, and with the gather's output it would
            // round-trip even if both index maps were the same mistake.
            let tok = seq(want.len(), 2.6);
            let mut want_plane = vec![0.0f32; 4 * plan.hp * plan.wp];
            cpu::scatter(&plan, &tok, shift, &mut want_plane);
            let dtok2 = cuda.upload(&tok)?;
            let dplane = cuda.buf(want_plane.len())?;
            let mut a = Args::new();
            a.ptr(dtok2.ptr).ptr(dplane.ptr)
                .i32(plan.nw as i32).i32(plan.n as i32).i32(plan.nww as i32).i32(plan.win as i32)
                .i32(plan.hp as i32).i32(plan.wp as i32).i32(4).i32(shift as i32);
            cuda.run("ss_window_scatter", Launch::new(grid_for(tok.len(), BLOCK), (BLOCK as u32, 1, 1)), &mut a)?;
            let mut got_plane = vec![0.0f32; want_plane.len()];
            dplane.download(&mut got_plane)?;
            close(&format!("ss_window_scatter shift={shift}"), &want_plane, &got_plane, tol)?;
            checks += 1;
        }

        // And the gather's normalising mode against the same fold the CPU side
        // does in its gather.
        let plan = Plan::new(16, 16, 8, 4);
        let x = seq(4 * plan.hp * plan.wp, 0.8);
        let nw = seq(4, 0.1);
        let nb = seq(4, 0.2);
        let mut want = vec![0.0f32; plan.nw * plan.n * 4];
        cpu::gather(&plan, &x, 0, Some((&nw, &nb)), &mut want);
        let dx = cuda.upload(&x)?;
        let dtok = cuda.buf(want.len())?;
        let mut a = Args::new();
        a.ptr(dx.ptr).ptr(cuda.upload(&nw)?.ptr).ptr(cuda.upload(&nb)?.ptr).ptr(dtok.ptr)
            .i32(plan.nw as i32).i32(plan.n as i32).i32(plan.nww as i32).i32(plan.win as i32)
            .i32(plan.hp as i32).i32(plan.wp as i32).i32(4).i32(0).f32(LN_EPS);
        cuda.run("ss_window_gather", Launch::new(grid_for(want.len(), BLOCK), (BLOCK as u32, 1, 1)), &mut a)?;
        let mut got = vec![0.0f32; want.len()];
        dtok.download(&mut got)?;
        close("ss_window_gather (norm)", &want, &got, tol)?;
        checks += 1;
    }

    {
        // ss_attention against cpu::attention, on a synthetic qkv built in the
        // memory order both use: [3][tokens][c].
        // The toy shape catches an index arithmetic mistake, and the checkpoint's
        // own (heads, head_dim) catches one that only shows at the width the graph
        // runs - the lightweight checkpoint's head_dim is 10 where the others are
        // 30, so the register row and the shared key-norm table are exercised at
        // two widths. The plan's channel count has to follow the shape: both twins
        // take it from there, not from `heads * head_dim`.
        for (heads, head_dim) in [(2usize, 2usize), (wt.heads, wt.head_dim)] {
        let c = heads * head_dim;
        let plan = Plan::new(16, 16, 8, c);
        for shift in [0usize, 4] {
            let qkv = seq(3 * plan.nw * plan.n * c, 0.45);
            // One learned temperature per head, as the checkpoint's table is.
            let ls: Vec<f32> = (0..heads).map(|h| 1.3 - 0.2 * h as f32).collect();
            let cpb = seq(plan.n * plan.n * heads, 0.05);
            // The kernel reads the transposed layout, as the graph's upload provides.
            let cpbt = transpose_cpb(&cpb, heads, plan.n);
            let mut want = vec![0.0f32; plan.nw * plan.n * c];
            cpu::attention(&plan, heads, head_dim, &ls, &cpb, &qkv, shift, &mut want);

            let dqkv = cuda.upload(&qkv)?;
            let dout = cuda.buf(want.len())?;
            let mut a = Args::new();
            a.ptr(dqkv.ptr).ptr(cuda.upload(&ls)?.ptr).ptr(cuda.upload(&cpbt)?.ptr).ptr(dout.ptr)
                .i32(plan.nw as i32).i32(plan.n as i32).i32(plan.nww as i32).i32(plan.win as i32)
                .i32(plan.hp as i32).i32(plan.wp as i32)
                .i32(heads as i32).i32(head_dim as i32).i32(shift as i32);
            let launch = Launch::new((plan.nw as u32, 1, 1), (plan.n as u32, heads as u32, 1));
            cuda.run("ss_attention", launch, &mut a)?;
            let mut got = vec![0.0f32; want.len()];
            dout.download(&mut got)?;
            close(&format!("ss_attention {heads}x{head_dim} shift={shift}"), &want, &got, tol)?;
            checks += 1;
        }
        }
    }

    {
        // The toolkit kernels this graph launches, against the same CPU twins the
        // graph uses. These are lightgpu's, but a shape this engine passes wrongly
        // is this engine's bug and would otherwise only appear as a wrong image.
        let (c, hw) = (5usize, 12usize);
        let x = seq(c * hw, 0.6);
        let w = seq(c, 0.7);
        let b = seq(c, 0.8);
        let mut want = vec![0.0f32; c * hw];
        lg::channel_layer_norm(&x, &w, &b, &mut want, c, hw, LN_EPS);
        let dx = cuda.upload(&x)?;
        let dy = cuda.buf(c * hw)?;
        let mut a = Args::new();
        a.ptr(dx.ptr).ptr(cuda.upload(&w)?.ptr).ptr(cuda.upload(&b)?.ptr).ptr(dy.ptr)
            .i32(c as i32).i32(hw as i32).f32(LN_EPS);
        cuda.run("lg_channel_layer_norm", Launch::new(grid_for(hw, BLOCK), (BLOCK as u32, 1, 1)), &mut a)?;
        let mut got = vec![0.0f32; c * hw];
        dy.download(&mut got)?;
        close("lg_channel_layer_norm", &want, &got, tol)?;
        checks += 1;

        let mut want = seq(21, 0.9);
        cpu::gelu_erf_inplace(&mut want);
        let dx = cuda.upload(&seq(21, 0.9))?;
        let dy = cuda.buf(21)?;
        let mut a = Args::new();
        a.ptr(dx.ptr).ptr(dy.ptr).i32(21);
        cuda.run("lg_gelu_erf", Launch::new(grid_for(21, BLOCK), (BLOCK as u32, 1, 1)), &mut a)?;
        let mut got = vec![0.0f32; 21];
        dy.download(&mut got)?;
        close("lg_gelu_erf", &want, &got, tol)?;
        checks += 1;

        let mut want = seq(21, 0.9);
        cpu::leaky_relu(&mut want, 0.2);
        let dx = cuda.upload(&seq(21, 0.9))?;
        let dy = cuda.buf(21)?;
        let mut a = Args::new();
        a.ptr(dx.ptr).ptr(dy.ptr).f32(0.2).i64(21);
        cuda.run("lg_lrelu", Launch::new(grid_for(21, BLOCK), (BLOCK as u32, 1, 1)), &mut a)?;
        let mut got = vec![0.0f32; 21];
        dy.download(&mut got)?;
        close("lg_lrelu", &want, &got, tol)?;
        checks += 1;
    }

    println!("cuda selftest: {checks} kernel/shape pairs checked against their CPU twins");
    Ok(())
}

/// Fail with the worst element, where it is, and both values - a bare "mismatch"
/// costs a debugging session.
fn close(what: &str, want: &[f32], got: &[f32], tol: f32) -> Result<(), String> {
    if want.len() != got.len() {
        return Err(format!("{what}: {} elements expected, {} produced", want.len(), got.len()));
    }
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..want.len() {
        let d = (want[i] - got[i]).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    if worst > tol {
        return Err(format!(
            "{what}: max |diff| {worst:.3e} at {at} > {tol:.1e} (cpu {:+.7}, cuda {:+.7})",
            want[at], got[at]
        ));
    }
    Ok(())
}
