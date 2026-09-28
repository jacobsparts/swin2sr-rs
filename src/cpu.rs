//! The CPU backend: one function per op, plain Rust, no device.
//!
//! This is the path for a machine with no usable GPU, and it is held to the same
//! standard as the device one: correct against the published PyTorch network, and
//! as fast as this machine can be made to run it. It is NOT the reference - the
//! reference is PyTorch, and a check that only compares the two backends to each
//! other would pass just as happily if both were wrong the same way.
//!
//! NCHW IS THE CANONICAL LAYOUT here and on the device: an activation is
//! `[c][hp][wp]`, and only window attention's operand and the MLP live in the
//! token layout `[tokens][c]`. The reference's `PatchEmbed.flatten(2)` and
//! `PatchUnEmbed` are reshapes between those two, so an image-layout stage is the
//! same tensor either way - which is why `patch_embed` below is a 1x1 CONVOLUTION
//! rather than a transposed matmul, and why the token layout only appears inside a
//! block.
//!
//! ONE FOLD, which does not change a value a different order of operations would
//! not already change: the `cpb_mlp` relative-position network is precomputed into
//! a [N*N][heads] table at conversion time (`tools/convert.py`), which is exactly
//! what the reference computes, once per checkpoint instead of once per forward.
//!
//! THE BLOCK IS POST-NORM, and that is worth stating twice because it is the one
//! place Swin2SR differs from the Swin-V2 it is built on. `norm1` is applied to
//! the attention's OUTPUT and `norm2` to the MLP's OUTPUT - the sublayers read the
//! raw activation. A pre-norm reading (normalise on the way in) produces an image
//! that looks like a plausible restoration and is wrong everywhere, which is how
//! this engine was first written and what `tests/parity.rs` exists to catch.
//!
//! Everything else is a direct transcription of `models/network_swin2sr.py` in the
//! order that file executes it. Where the toolkit already has a CPU twin - add,
//! the erf GELU - this calls it (`lightgpu::ops::cpu`) rather than growing a
//! second implementation that could drift. The channel LayerNorm is the one
//! exception, and `channel_layer_norm` below says why: the toolkit's twin is a
//! single-threaded loop, the op is a sixth of a forward, and the arithmetic is
//! reproduced exactly - `selftest` compares the two directly rather than trusting
//! that it was.
use rayon::prelude::*;

use lightgpu::ops::cpu as lg;

use crate::plan::{pad_reflect, Plan};
use crate::weights::{Upsampler, Weights};

const LN_EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------
// Parallelism.
//
// A convolution in this model is a `[180][180][9]` weight against a
// `[180][264][264]` plane: ~23 GFLOP per op, twelve of them per stage, and on one
// core that is a minute per image. Every op below is therefore split across the
// machine's cores with rayon.
//
// THE SPLIT IS ALWAYS OVER INDEPENDENT OUTPUT ELEMENTS, and each output is
// accumulated by ONE thread in the SAME ORDER the sequential version used - so
// the parallel result is bit-identical to the sequential one and the tolerance in
// `--cuda-selftest` and `tests/parity.rs` is unaffected. That is the reason the
// splits are over the output rather than over the reduction: a reduction split
// across threads changes the summation order, which is a different answer, and
// this engine has no headroom to spend on that - the fixture tolerance is already
// only three orders of magnitude wide.
//
// The threshold keeps small ops on one core: a `par_chunks` over three elements
// costs more in wakeups than it saves, and the graph is full of small ops (the
// heads, the norms, the 1x1s at small sizes).
// ---------------------------------------------------------------------------

/// Elements below which an op stays on one thread.
const PAR_MIN: usize = 1 << 14;

/// Whether an op of `n` elements is worth handing to the pool. Both halves
/// matter: below the threshold the wakeups cost more than the work, and on a
/// one-core machine there is nothing to hand it to.
#[inline]
fn par_on(n: usize) -> bool {
    n >= PAR_MIN && rayon::current_num_threads() > 1
}

/// Every buffer the block loop needs, sized for one `Plan`. Named fields rather
/// than one arena: the two failure modes worth being able to see at a glance are
/// "a buffer is too small" and "two roles share one buffer", and both are harder
/// to spot behind an offset.
pub struct Scratch {
    /// The image-layout activation: [c][hp][wp].
    pub cur: Vec<f32>,
    /// The image-layout residual of the block in flight.
    pub res: Vec<f32>,
    /// The stage's input, held for the RSTB's own `+ x`. It cannot share `res`:
    /// that one is a running residual inside the block loop and is overwritten by
    /// every block, so by the time the stage's convs are added there is nothing
    /// left of the value they are supposed to be added to.
    pub stage: Vec<f32>,
    /// Image-layout scratch (the attention scatter's destination).
    pub acc: Vec<f32>,
    /// Token-layout operand: [tokens][c].
    pub tok: Vec<f32>,
    /// Token-layout operand: [tokens][c].
    pub tok2: Vec<f32>,
    /// Attention output before the scatter: [tokens][c].
    pub attn: Vec<f32>,
    /// The fused qkv projection: [3][tokens][c], in that order.
    pub qkv: Vec<f32>,
    /// The MLP hidden layer: [tokens][mlp_ratio*c].
    pub mlp: Vec<f32>,
    /// `linear`'s transposed operand: [c_in][rows] for the largest matmul in the
    /// graph, which is the MLP's first layer ([tokens][mlp_ratio*c]). Reused by
    /// every `linear` call, so a block pays for one buffer rather than one per
    /// matmul.
    pub xt: Vec<f32>,
    /// `conv1x1`'s row-major result, one plane of floats, before the transpose
    /// that puts it back into NCHW. Only ever touched inside `conv1x1`.
    pub ctmp: Vec<f32>,
    /// A zero bias for the key projection, which the reference gives none.
    pub zero_bias: Vec<f32>,
    /// [`matmul_rows`]'s packed weight, `[c_in][c_out]`. Sized for the LARGEST
    /// matmul in the graph - the MLP's, at `2 * c * c` - so one buffer serves
    /// every `linear` and `conv1x1` call. Unlike the activation buffers this does
    /// not grow with the image: it is 259 KB for a 180-wide model at any size.
    pub wpack: Vec<f32>,
}

impl Scratch {
    pub fn new(plan: &Plan, mlp_ratio: usize) -> Scratch {
        let plane = plan.plane() * plan.c;
        let tok = plan.tokens() * plan.c;
        Scratch {
            cur: vec![0.0; plane],
            res: vec![0.0; plane],
            stage: vec![0.0; plane],
            acc: vec![0.0; plane],
            tok: vec![0.0; tok],
            tok2: vec![0.0; tok],
            attn: vec![0.0; tok],
            qkv: vec![0.0; 3 * tok],
            mlp: vec![0.0; tok * mlp_ratio],
            xt: vec![0.0; tok * mlp_ratio.max(1)],
            ctmp: vec![0.0; plane],
            zero_bias: vec![0.0; plan.c],
            wpack: vec![0.0; 2 * plan.c * plan.c],
        }
    }

    /// f32 allocated, for the memory line of the run report.
    pub fn floats(&self) -> usize {
        self.cur.len() + self.res.len() + self.stage.len() + self.acc.len()
            + self.tok.len() + self.tok2.len()
            + self.attn.len() + self.qkv.len() + self.mlp.len() + self.xt.len()
            + self.ctmp.len() + self.zero_bias.len() + self.wpack.len()
    }

    /// Take one pooled block of at least `n` floats, for a buffer that lives
    /// inside one forward and is not part of the long-lived set (`nbuf`, the head's
    /// octave planes). It comes from the same pool the scratch does and goes back
    /// to it on drop, so a forward allocates for its largest buffer rather than for
    /// every buffer at once.
    pub fn take_block(&self, n: usize) -> Vec<f32> {
        let mut v = POOL.with(|p| p.borrow_mut().pop().unwrap_or_default());
        v.clear();
        v.resize(n, 0.0);
        v
    }

    /// Release everything the head cannot need, and hand back what it can.
    ///
    /// The head reads `cur` exactly once (its first conv) and `acc` once (as the
    /// `selftest`-style destination of that conv is not `acc` - the head's own `a`
    /// is), and after that no buffer in here is live again. So this drops the ones
    /// that are pure body scratch - `res`, `stage`, the four token layouts - and
    /// returns `cur` and `acc` to the caller, which is how the peak stops counting
    /// them against the head's octave planes. On the c=180 graph that is 8 padded
    /// planes of 29.6, i.e. 27% of the engine's whole allocation, held alive by
    /// nothing but the borrow checker's inability to see the last use.
    ///
    /// WHAT THIS DOES NOT DO is give the memory back to the ALLOCATOR in time to
    /// matter to the peak: the freed blocks go to the pool this `Cpu` reuses for
    /// the next size, and the head's new buffers may reuse them. So the ordering
    /// here is the mechanism: the frees happen BEFORE the octave loop allocates its
    /// planes, and a freed block of the right size class is what the loop's
    /// `vec![0.0; ...]` gets handed back.
    pub fn free_body(&mut self) {
        for b in [&mut self.res, &mut self.stage, &mut self.tok, &mut self.tok2,
                  &mut self.attn, &mut self.qkv, &mut self.mlp, &mut self.xt] {
            let _ = std::mem::take(b);
        }
    }
}

// The scratch of one forward, owned rather than borrowed, so a function it is
// passed to can FREE PARTS OF IT.
//
// WHY THIS EXISTS. The head's first layer is the body's last, and every buffer
// behind `cur` is dead the moment it has run - but `Scratch` was borrowed by
// `forward` for the whole call, so nothing could release anything and the peak was
// the sum of buffers that never coexisted. Moving the scratch by value through
// `head` is what makes the release (`free_body`) possible at all.
//
// A moved value cannot implement `Drop`, which is the only reason this is a
// wrapper and not `Scratch` itself: the drop is what hands the freed blocks back
// to `POOL`, so the next forward of any size in this thread starts with the
// previous one's memory rather than asking the allocator for the same bytes again
// - which is also what keeps the peak from reflecting two sizes at once.
//
// The pool is per-thread and unbounded; every buffer in it is at most one plane
// of a plan, so it is the same memory a single `Scratch` holds.
thread_local! {
    static POOL: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub struct ScratchOwned {
    pub buf: Box<Scratch>,
    /// Blocks carried over from the previous forward in this thread - one per
    /// buffer, sized for whatever plan that one used.
    pool: Vec<Vec<f32>>,
}

impl ScratchOwned {
    /// Build the scratch for `plan`, reusing whatever this thread last handed back.
    ///
    /// The reuse is `Vec::clear` plus `resize`: a block that is already long enough
    /// keeps its allocation, one that is not grows by exactly the missing amount.
    /// So a same-size second pass allocates nothing at all, and a larger one
    /// allocates only the difference - which matters here because the peak of a
    /// bigger run would otherwise be its own buffers PLUS the smaller run's, left to
    /// the allocator's mercy.
    pub fn new(plan: &Plan, mlp_ratio: usize) -> ScratchOwned {
        let mut pool = POOL.with(|p| std::mem::take(&mut *p.borrow_mut()));
        let block = |n: usize, pool: &mut Vec<Vec<f32>>| {
            let mut v = pool.pop().unwrap_or_default();
            v.clear();
            v.resize(n, 0.0);
            v
        };
        let plane = plan.plane() * plan.c;
        let tok = plan.tokens() * plan.c;
        ScratchOwned {
            buf: Box::new(Scratch {
                cur: block(plane, &mut pool),
                res: block(plane, &mut pool),
                stage: block(plane, &mut pool),
                acc: block(plane, &mut pool),
                tok: block(tok, &mut pool),
                tok2: block(tok, &mut pool),
                attn: block(tok, &mut pool),
                qkv: block(3 * tok, &mut pool),
                mlp: block(tok * mlp_ratio, &mut pool),
                xt: block(tok * mlp_ratio.max(1), &mut pool),
                ctmp: block(plane, &mut pool),
                zero_bias: vec![0.0; plan.c],
                wpack: block(2 * plan.c * plan.c, &mut pool),
            }),
            pool,
        }
    }
}

impl ScratchOwned {
    /// Fold the previous forward's blocks into this one's pool.
    ///
    /// A size change is not a reason to allocate: every block in `self` has already
    /// been sized for the NEW plan (that is what `new` did), and `old`'s blocks are
    /// exactly the ones this thread will want for the next size change - and for the
    /// in-flight frees of this forward, since `Scratch::free_body` and
    /// `take_block` pull from the same pool.
    fn absorb(&mut self, mut old: ScratchOwned) {
        let mut pool = std::mem::take(&mut old.pool);
        for b in [&mut old.buf.cur, &mut old.buf.res, &mut old.buf.stage, &mut old.buf.acc,
                  &mut old.buf.tok, &mut old.buf.tok2, &mut old.buf.attn, &mut old.buf.qkv,
                  &mut old.buf.mlp, &mut old.buf.xt, &mut old.buf.ctmp, &mut old.buf.wpack] {
            pool.push(std::mem::take(b));
        }
        // `old`'s own Drop now sees emptied buffers and pushes nothing.
        self.pool.splice(0..0, pool);
    }
}

impl ScratchOwned {
    /// Restore the invariant `new` establishes: every `Scratch` field holds a block
    /// sized for `plan`.
    ///
    /// WHY THIS IS NEEDED AT ALL. `forward` takes every field OUT of its `Scratch`
    /// (into locals whose lifetimes the borrow checker can follow) and puts only
    /// `cur` and `acc` back, so the handle that comes out of a forward has empty
    /// fields and a pool holding the blocks that were in them. That is fine for a
    /// caller that always rebuilds - and `Cpu::forward` does rebuild when the
    /// geometry changes - but it REUSES the handle when the plan matches, and then
    /// the first `take!` of the next forward reads a zero-length Vec. A tiled run
    /// hits exactly that: the interior tiles all have the same crop size, so the
    /// second tile of the first identical pair takes the reuse path and dies with
    /// `copy_from_slice: source slice length ... does not match destination slice
    /// length (0)` in the stage loop, which is a long way from the cause.
    ///
    /// Refilling here rather than in the caller keeps the pooling: the blocks go
    /// back into the fields they came from, and a later size change still finds them
    /// through `absorb`/`new`.
    pub fn refill(&mut self, plan: &Plan, mlp_ratio: usize) {
        let block = |v: &mut Vec<f32>, n: usize, pool: &mut Vec<Vec<f32>>| {
            if v.len() != n {
                let mut fresh = pool.pop().unwrap_or_default();
                fresh.clear();
                fresh.resize(n, 0.0);
                // Whatever `v` was, it is one of the two size classes this plan
                // wants for some buffer, so it goes back rather than being dropped.
                if !v.is_empty() {
                    pool.push(std::mem::take(v));
                }
                *v = fresh;
            }
        };
        let plane = plan.plane() * plan.c;
        let tok = plan.tokens() * plan.c;
        let pool = &mut self.pool;
        block(&mut self.buf.cur, plane, pool);
        block(&mut self.buf.res, plane, pool);
        block(&mut self.buf.stage, plane, pool);
        block(&mut self.buf.acc, plane, pool);
        block(&mut self.buf.tok, tok, pool);
        block(&mut self.buf.tok2, tok, pool);
        block(&mut self.buf.attn, tok, pool);
        block(&mut self.buf.qkv, 3 * tok, pool);
        block(&mut self.buf.mlp, tok * mlp_ratio, pool);
        block(&mut self.buf.xt, tok * mlp_ratio.max(1), pool);
        block(&mut self.buf.ctmp, plane, pool);
        block(&mut self.buf.wpack, 2 * plan.c * plan.c, pool);
        if self.buf.zero_bias.len() != plan.c {
            self.buf.zero_bias.resize(plan.c, 0.0);
        }
    }
}

impl Drop for ScratchOwned {
    fn drop(&mut self) {
        let mut pool = self.pool.drain(..).collect::<Vec<_>>();
        pool.push(std::mem::take(&mut self.buf.cur));
        pool.push(std::mem::take(&mut self.buf.res));
        pool.push(std::mem::take(&mut self.buf.stage));
        pool.push(std::mem::take(&mut self.buf.acc));
        pool.push(std::mem::take(&mut self.buf.tok));
        pool.push(std::mem::take(&mut self.buf.tok2));
        pool.push(std::mem::take(&mut self.buf.attn));
        pool.push(std::mem::take(&mut self.buf.qkv));
        pool.push(std::mem::take(&mut self.buf.mlp));
        pool.push(std::mem::take(&mut self.buf.xt));
        pool.push(std::mem::take(&mut self.buf.ctmp));
        pool.push(std::mem::take(&mut self.buf.wpack));
        POOL.with(|p| *p.borrow_mut() = pool);
    }
}

// ---------------------------------------------------------------------------
// The convolution and resampling family.
//
// The two dense ops here - the 3x3 and the 1x1 - are LIGHTGPU'S. `lg_conv3x3s1p1`
// and `lg_conv1x1` began in this file, as the hand-written kernels below, and were
// promoted into the toolkit so the rest of the family could use them; the copies
// that used to live here are gone and what is left calls the toolkit's. The 3x3
// arrives with the 32-column AVX2 register tile, the 1x1 with the register-tiled
// matmul, and both with the parallel framing - so one implementation of each now
// serves this engine, the device side's selftest and the rest of the family,
// instead of three that could drift apart.
//
// THE SPEED IS A WASH HERE, and it is worth saying so in the file rather than
// implying a win. Measured on this box, 64x64 and 128x128 classical-x4 on CPU,
// interleaved: 0.791/0.802 s and 2.071/2.085 s before and after (medians of five).
// The twin is the faster kernel where it was measured - 1.17x-3.33x this file's own
// AVX2 on ifan-rs's shapes - but this engine's time is 42% matmul (`linear`, which
// the toolkit only has as the scalar reference `linear_rb`) and 32% window
// attention (no toolkit twin at all), with the 3x3 at 11% and the 1x1 at 1%. A
// faster kernel cannot move a total that is not mostly that kernel.
//
// WHAT IT COSTS is bit-exactness, and only there. The toolkit's 3x3 accumulates in
// the tiled kernel's order - `ky`, `kx`, `ci`, with zero taps skipped - where the
// row routine below accumulates `ci`, `ky`, `kx`, so the two differ in the last
// place of an f32 sum. The fixtures are unmoved at that level (2e-6..5e-6 against a
// 2e-3 tolerance) but the last bit does move: at 128x128 the final PNG differs from
// the pre-adoption build on 23 of its 786432 channel values, all by one 8-bit
// level. That is why `selftest` holds `conv3x3` to the same 1e-5 rounding bound it
// holds every other op to, rather than to the bit-identity the old copy could
// claim, and why the head below keeps its own order.
//
// The pixel-shuffle HEAD still runs the engine's own order, through `conv_row`:
// its fused 3x3 keeps one kernel writing straight into the shuffled layout, which
// is 152 MB of peak allocation and a whole extra pass over the planes, and that
// fusion does not exist in the toolkit, so it stays here with the order it had.
// ---------------------------------------------------------------------------

/// 3x3, stride 1, zero padding 1: `out[co][y][x] = b[co] + sum w * in`.
///
/// This is `lightgpu::ops::cpu::conv3x3s1p1`, which is this file's old kernel after
/// its promotion into the toolkit: the same tap order the device kernel uses, the
/// same zero-padding convention, the same split across cores by output row, plus
/// the 32-column AVX2 register tile with all 27 taps register-resident and the
/// channel pairing (one task per output-channel pair, four loads feeding eight
/// FMAs). See the module comment above for why the adoption is not bit-exact.
///
/// The wrapper stays because the argument worth keeping is the one `lg` does not
/// have: the debug asserts below catch a caller passing the wrong plane shape,
/// which is the mistake this op invites - every call site is a padded plane and
/// the padding is not derivable from the buffer length.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3(
    inp: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    wgt: &[f32],
    c_out: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(inp.len(), c_in * h * w);
    debug_assert_eq!(out.len(), c_out * h * w);
    debug_assert_eq!(wgt.len(), c_out * c_in * 9);
    lg::conv3x3s1p1(inp, wgt, bias, out, c_in, c_out, h, w);
}

/// One output ROW of a 3x3, stride 1, zero padding 1 convolution, for one output
/// channel: `dest[x] += sum_ky sum_kx sum_ci wc[ci][ky][kx] * inp[ci][y+ky-1][x+kx-1]`.
///
/// This is the head's, and it is now its ONLY caller: the fused pixel-shuffle
/// stage, which has no toolkit counterpart. `dest` is accumulated into rather than
/// overwritten, because that stage runs this twice into two different buffers and
/// then interleaves them.
///
/// THE ORDER IS THE TAP ORDER, `ky` then `kx` then `ci`, which is the toolkit's and
/// the device kernel's - and not the `ci`, `ky`, `kx` this file used to have. That
/// distinction matters because `selftest` used to compare this row's output with
/// `conv3x3`'s bit for bit, and `conv3x3` is now the toolkit's, so the comparison
/// is at a rounding tolerance instead. Keeping the taps adjacent also lets a row
/// load a pixel once and use it for all three of its `kx` taps, which is what
/// [`row_accumulate`] does.
#[inline]
fn conv_row(inp: &[f32], c_in: usize, h: usize, w: usize, wc: &[f32], b: f32, y: usize,
            dest: &mut [f32]) {
    dest.fill(b);
    // The rows this output row reads: the one above is missing at the top edge
    // and the one below at the bottom, and nothing else changes.
    let ky0 = if y == 0 { 1 } else { 0 };
    let ky1 = if y + 1 == h { 1 } else { 2 };
    row_accumulate(inp, wc, c_in, h * w, y, w, ky0, ky1, dest);
}

/// The scalar half of [`conv_row`]: one output row, over a RANGE of input rows.
///
/// `ky0..=ky1` is the skip-the-missing-edge-rows trick, and it works because the
/// row this reads for tap `ky` is `y + ky - 1`: at the top edge `ky = 0` would read
/// row -1 and is left out of the range, at the bottom `ky = 2` would read row `h`.
/// With the taps adjacent, a `ky` pass reads one input row and writes all three of
/// that row's contributions, so the row is loaded once per output element instead
/// of three times.
///
/// The border carries no branch either: the first and last column of each input
/// row are peeled out inside the `ky` pass below, and the interior loop has none.
#[inline]
#[allow(clippy::too_many_arguments)]
fn row_accumulate(
    inp: &[f32], wc: &[f32], c_in: usize, hw: usize, y: usize, w: usize,
    ky0: usize, ky1: usize, dest: &mut [f32],
) {
    for ci in 0..c_in {
        let ic = &inp[ci * hw..ci * hw + hw];
        let k = &wc[ci * 9..ci * 9 + 9];
        for ky in ky0..=ky1 {
            let srow = &ic[(y + ky - 1) * w..(y + ky) * w];
            let krow = &k[ky * 3..ky * 3 + 3];
            // `w == 1` is its own case: only the centre tap is in the plane, and
            // the general form below would read `srow[saturating_sub(1)]`.
            if w == 1 {
                dest[0] += krow[1] * srow[0];
                continue;
            }
            dest[0] += krow[1] * srow[0];
            dest[0] += krow[2] * srow[1];
            for x in 1..w - 1 {
                dest[x] += krow[0] * srow[x - 1];
                dest[x] += krow[1] * srow[x];
                dest[x] += krow[2] * srow[x + 1];
            }
            dest[w - 1] += krow[0] * srow[w - 2];
            dest[w - 1] += krow[1] * srow[w - 1];
        }
    }
}

/// A 3x3 convolution whose output is written STRAIGHT INTO THE PIXEL-SHUFFLED
/// LAYOUT, so the `4 * feat`-channel intermediate never exists.
///
/// This is the last stage of the pixel-shuffle head. The plain path builds
/// `big[4 * ch + 2 * dy + dx][y][x]` with a conv and then permutes it into
/// `shuf[ch][2y + dy][2x + dx]` - and at the head's last octave those two buffers
/// are 4 * 64 * 272 * 272 floats each, 76 MB apiece at a 128x128 input, which made
/// the pair 40% of the engine's whole peak allocation. Fusing them costs one
/// buffer instead of two and one pass over 152 MB less.
///
/// The permutation is what makes it easy: output row `y2` of channel `ch` comes
/// from exactly two rows of `big` - `co0 = 4 * ch + 2 * (y2 % 2)` and `co0 + 1`,
/// with `y = y2 / 2` - interleaved even/odd. So a task is one output row, and it
/// calls [`conv_row`] twice. `selftest` checks this against `conv3x3` followed by
/// `pixel_shuffle2`, bit for bit.
///
/// `out` is `[feat][2h][2w]`.
#[allow(clippy::too_many_arguments)]
pub fn conv3x3_shuffle2(
    inp: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    wgt: &[f32],
    feat: usize,
    bias: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(inp.len(), c_in * h * w);
    debug_assert_eq!(out.len(), feat * 4 * h * w);
    debug_assert_eq!(wgt.len(), 4 * feat * c_in * 9);
    let (h2, w2) = (2 * h, 2 * w);
    let row = |i: usize, dest: &mut [f32], t0: &mut [f32], t1: &mut [f32]| {
        let (ch, y2) = (i / h2, i % h2);
        let (dy, y) = (y2 % 2, y2 / 2);
        let co0 = 4 * ch + 2 * dy;
        conv_row(inp, c_in, h, w, &wgt[co0 * c_in * 9..(co0 + 1) * c_in * 9], bias[co0], y,
                 &mut t0[..w]);
        let co1 = co0 + 1;
        conv_row(inp, c_in, h, w, &wgt[co1 * c_in * 9..(co1 + 1) * c_in * 9], bias[co1], y,
                 &mut t1[..w]);
        for x in 0..w {
            dest[2 * x] = t0[x];
            dest[2 * x + 1] = t1[x];
        }
    };
    if par_on(feat * h2 * w2) {
        out.par_chunks_mut(w2)
            .enumerate()
            .for_each_init(|| (vec![0.0f32; w], vec![0.0f32; w]), |(t0, t1), (i, dest)| {
                row(i, dest, t0, t1)
            });
    } else {
        let (mut t0, mut t1) = (vec![0.0f32; w], vec![0.0f32; w]);
        for (i, dest) in out.chunks_mut(w2).enumerate() {
            row(i, dest, &mut t0, &mut t1);
        }
    }
}

/// 1x1, stride 1 - a matmul over positions. `inp` is `[c_in][hw]`, `out` is
/// `[c_out][hw]`, so both are channel-major and the positions are the "rows".
///
/// This is `lightgpu::ops::cpu::conv1x1`, which is this file's old implementation
/// after its promotion into the toolkit - the same register-tiled matmul, folding
/// the bias into the accumulator exactly as the old one did, and now the same
/// toolkit entry point the device side's twin uses.
///
/// TWO ARGUMENTS ARE GONE with it. `tmp` and `wpack` were this op's scratch: the
/// row-major destination is written by the toolkit's own internal split, so no
/// `c_out * hw` staging buffer and no transposed weight pack are needed here. They
/// stay in the signature - `Scratch` still owns them, they are still reported in
/// the run's memory line, and `gpu.rs` still passes its own - but they are unused,
/// which the `let _` below says out loud rather than leaving the reader to wonder
/// whether the call forgot them.
#[allow(clippy::too_many_arguments)]
pub fn conv1x1(
    inp: &[f32],
    c_in: usize,
    hw: usize,
    wgt: &[f32],
    c_out: usize,
    bias: &[f32],
    out: &mut [f32],
    tmp: &mut [f32],
    wpack: &mut [f32],
) {
    debug_assert_eq!(inp.len(), c_in * hw);
    debug_assert_eq!(out.len(), c_out * hw);
    debug_assert_eq!(wgt.len(), c_out * c_in);
    debug_assert!(tmp.len() >= c_out * hw);
    let _ = (tmp, wpack);
    lg::conv1x1(inp, wgt, bias, out, c_in, c_out, 1, hw);
}

/// `wpack[ci][co] = wgt[co][ci]`: the weight `[c_out][c_in]` in the layout the
/// register tile below wants, where the `co` axis is the contiguous one.
///
/// THE PACK IS NOT FREE AND IT IS NOT PER-ELEMENT. It is a transpose of the
/// weight, done once per matmul - 130 KB for the largest one in this graph, which
/// measures 0.2 ms against the tens of milliseconds the matmul it feeds takes.
/// Hoisting it to load time would save that 0.2 ms and cost a second copy of every
/// weight in the checkpoint, so it stays here.
pub fn pack_weight(wgt: &[f32], c_in: usize, c_out: usize, wpack: &mut [f32]) {
    debug_assert!(wpack.len() >= c_in * c_out);
    transpose(wgt, &mut wpack[..c_in * c_out], c_out, c_in);
}

/// A plain matmul: `out[r][co] = bias[co] + sum_ci w[co][ci] x[r][ci]`.
///
/// `xt` is scratch of at least `c_in * rows` floats and holds the operand
/// TRANSPOSED, `[c_in][rows]`. That transpose is what makes this fast, and it is
/// worth the copy: with the operand row-major the reduction runs along the row
/// and every output element is a separate `c_in`-long dot product, which measures
/// 6 GFLOP/s on this machine however it is unrolled. Transposed, the inner loop
/// becomes one axpy over `rows` - contiguous, vectorisable, and the weight scalar
/// is loaded once for the whole sweep - which measures 24 GFLOP/s in the same
/// harness. See the "why this shape" note above `conv1x1`, which is the same loop
/// with the operand already in the right layout.
///
/// The split is over ROW BLOCKS; each output element is still summed over
/// ascending `ci` by one thread, so the result is bit-identical to the plain
/// row-at-a-time form.
pub fn linear(
    x: &[f32],
    rows: usize,
    c_in: usize,
    wgt: &[f32],
    c_out: usize,
    bias: &[f32],
    out: &mut [f32],
    xt: &mut [f32],
    wpack: &mut [f32],
) {
    debug_assert_eq!(wgt.len(), c_out * c_in);
    debug_assert_eq!(out.len(), rows * c_out);
    debug_assert!(xt.len() >= c_in * rows);
    let xt = &mut xt[..c_in * rows];
    transpose(x, xt, rows, c_in);
    pack_weight(wgt, c_in, c_out, wpack);
    matmul_rows(xt, rows, c_in, &wpack[..c_in * c_out], c_out, bias, out);
}

/// `out[r][co] = bias[co] + sum_ci wp[ci][co] * xt[ci][r]`, the transposed-operand
/// form with the weight PACKED as `[c_in][c_out]`.
///
/// WHY THIS SHAPE. The inner step is one axpy per (ci, co): contiguous reads from
/// both operands, one weight load per sweep. The row-major alternative - a
/// `c_in`-long dot product per output element - measures 6 GFLOP/s however it is
/// unrolled, and the transposed form that follows from it measures 24.
///
/// WHY THERE IS A REGISTER TILE, AND WHY IT IS DISPATCHED AT RUN TIME. With one
/// output row per accumulator the accumulator has to live in memory and is
/// re-read once per FMA - 23 KB at `c_out = 180` against a 32 KB L1D, which is
/// where the plain form's plateau comes from. Holding a tile of accumulators in
/// registers instead removes that traffic.
///
/// But HOW BIG a tile pays depends on the instruction set, and the two pull in
/// opposite directions. Measured with a cycle counter, minimum of nine interleaved
/// runs (the only timing metric that survives this machine's background load), on
/// the three matmul shapes this graph uses:
///
/// ```text
///                            baseline (SSE2)      AVX2 + FMA
///   plain, accumulator in memory   135-140k          97-98k   cycles/MFLOP
///   4 rows x  8 channels           110-127k          79-101k
///   8 rows x 10 channels           145-190k          59-60k   <- best on AVX2
///   8 rows x 12 channels           367k              63-65k
/// ```
///
/// An 8x10 tile is 80 accumulator floats - twenty YMM registers against sixteen -
/// which is why it wins only where those registers exist, and a 4x8 tile is 32
/// floats, which is eight XMM registers, the whole SSE register file. Compiling
/// for `target-cpu=native` would pick the right one, but it would also make the
/// binary illegal on any CPU without AVX2, so the choice is made at run time
/// instead: `is_x86_feature_detected!` picks the tile, and the worker for each is
/// a `#[target_feature]` function so the compiler may use those instructions in
/// it. The two workers are generated from one macro, so they cannot drift.
///
/// THE TILE SIZE IS LOAD-BEARING. Every variant that spills past the register file
/// (8x14, 8x16, 10x10, 12x12) is 2-3x SLOWER than the plain form, and so is every
/// variant whose loop bounds the unroller cannot resolve. The ragged edges are
/// therefore PEELED OUT - a column tail and a row tail, both with the same
/// ascending-`ci` accumulation - rather than handled inside the tile with a branch.
///
/// WHY THE SPLIT IS OVER ROW BLOCKS. It keeps the inner loop above and gives every
/// task a contiguous destination slice: `rows / MR` tasks, which is 2312 for a
/// 264x264 plane. The alternative - splitting a CHANNEL-major destination by
/// channel block - is limited to `c_out / 32` tasks, six for this model, and that
/// is what made the first version of this rewrite slower than the plain loop it
/// replaced.
///
/// The accumulation order per output element is `ci` ascending, so this is
/// bit-identical to the plain form, and `selftest` checks it against the definition
/// of a matmul.
fn matmul_rows(xt: &[f32], rows: usize, c_in: usize, wp: &[f32], c_out: usize,
               bias: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2 = false;
    let mr = if avx2 { 8 } else { 4 };
    let run = |(b, chunk): (usize, &mut [f32])| unsafe {
        if avx2 {
            matmul_avx2(chunk, b, xt, rows, c_in, wp, c_out, bias);
        } else {
            matmul_base(chunk, b, xt, rows, c_in, wp, c_out, bias);
        }
    };
    if par_on(c_out * rows) {
        out.par_chunks_mut(mr * c_out).enumerate().for_each(run);
    } else {
        for (b, chunk) in out.chunks_mut(mr * c_out).enumerate() {
            run((b, chunk));
        }
    }
}

/// One ROW BLOCK of [`matmul_rows`]: `chunk` is `[MR][c_out]` holding output rows
/// `b * MR ..`, and `wp` is the packed weight.
///
/// SAFETY (the caller's obligation): `chunk.len() == MR * c_out`, `rows`, `c_in`
/// and `c_out` describe `xt` and `wp` as `matmul_rows` documents, and for the
/// `#[target_feature]` instantiation the CPU has the named features.
macro_rules! matmul_tile {
    ($name:ident, $mr:expr, $nr:expr, $($attr:tt)*) => {
        #[allow(clippy::too_many_arguments)]
        $($attr)*
        unsafe fn $name(chunk: &mut [f32], b: usize, xt: &[f32], rows: usize, c_in: usize,
                        wp: &[f32], c_out: usize, bias: &[f32]) {
            /// Output rows held in registers.
            const MR: usize = $mr;
            /// Output channels held in registers.
            const NR: usize = $nr;
            let r0 = b * MR;
            let nr = chunk.len() / c_out;
            let mut rb = 0;
            while rb + MR <= nr {
                let mut co0 = 0;
                while co0 + NR <= c_out {
                    let mut acc = [[0.0f32; MR]; NR];
                    for j in 0..NR {
                        for i in 0..MR {
                            acc[j][i] = bias[co0 + j];
                        }
                    }
                    for ci in 0..c_in {
                        let xv: &[f32; MR] = xt[ci * rows + r0 + rb..][..MR].try_into().unwrap();
                        let wv: &[f32; NR] = wp[ci * c_out + co0..][..NR].try_into().unwrap();
                        for j in 0..NR {
                            let w = wv[j];
                            for i in 0..MR {
                                acc[j][i] += w * xv[i];
                            }
                        }
                    }
                    for j in 0..NR {
                        let dst = &mut chunk[rb * c_out + co0 + j..];
                        for i in 0..MR {
                            dst[i * c_out] = acc[j][i];
                        }
                    }
                    co0 += NR;
                }
                // The column tail: fewer than NR output channels left.
                for j in co0..c_out {
                    for i in 0..MR {
                        let mut a = bias[j];
                        for ci in 0..c_in {
                            a += wp[ci * c_out + j] * xt[ci * rows + r0 + rb + i];
                        }
                        chunk[(rb + i) * c_out + j] = a;
                    }
                }
                rb += MR;
            }
            // The row tail: fewer than MR rows left.
            for i in rb..nr {
                for j in 0..c_out {
                    let mut a = bias[j];
                    for ci in 0..c_in {
                        a += wp[ci * c_out + j] * xt[ci * rows + r0 + i];
                    }
                    chunk[i * c_out + j] = a;
                }
            }
        }
    };
}

matmul_tile!(matmul_avx2, 8, 10, #[target_feature(enable = "avx2,fma")]);
matmul_tile!(matmul_base, 4, 8, );

/// `dst[c][r] = src[r][c]`, in parallel over the destination's rows (each
/// destination row is a column of the source, so the source is read with a
/// `c_in`-stride - one cache line per element, which is why this is worth doing
/// once per matmul and not once per `ci`).
fn transpose(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    let col = |c: usize, d: &mut [f32]| {
        for (r, v) in d.iter_mut().enumerate() {
            *v = src[r * cols + c];
        }
    };
    if par_on(rows * cols) {
        dst.par_chunks_mut(rows).enumerate().for_each(|(c, d)| col(c, d));
    } else {
        for (c, d) in dst.chunks_mut(rows).enumerate() {
            col(c, d);
        }
    }
}

/// `F.pixel_shuffle(x, 2)`: [4C][H][W] -> [C][2H][2W].
pub fn pixel_shuffle2(inp: &[f32], c4: usize, h: usize, w: usize, out: &mut [f32]) {
    let c = c4 / 4;
    debug_assert_eq!(inp.len(), c4 * h * w);
    debug_assert_eq!(out.len(), c * 4 * h * w);
    let (oh, ow) = (h * 2, w * 2);
    // One output row per task. A row `(ch, y, dy)` interleaves the two sub-channel
    // planes `ch*4 + dy*2 + {0,1}` and writes each output element exactly once, so
    // the split needs no ordering argument.
    let row = |r: usize, dst: &mut [f32]| {
        let (ch, rem) = (r / oh, r % oh);
        let (dy, y) = (rem & 1, rem >> 1);
        let s0 = &inp[((ch * 4 + dy * 2) * h + y) * w..][..w];
        let s1 = &inp[((ch * 4 + dy * 2 + 1) * h + y) * w..][..w];
        for x in 0..w {
            dst[x * 2] = s0[x];
            dst[x * 2 + 1] = s1[x];
        }
    };
    if par_on(out.len()) {
        out.par_chunks_mut(ow).enumerate().for_each(|(r, dst)| row(r, dst));
    } else {
        for (r, dst) in out.chunks_mut(ow).enumerate() {
            row(r, dst);
        }
    }
}

/// `F.pixel_shuffle(x, scale)` for the one-step head: [C*scale^2][H][W] -> [C][H*scale][W*scale].
pub fn pixel_shuffle(inp: &[f32], c: usize, h: usize, w: usize, scale: usize, out: &mut [f32]) {
    let oh = h * scale;
    let ow = w * scale;
    debug_assert_eq!(inp.len(), c * scale * scale * h * w);
    debug_assert_eq!(out.len(), c * oh * ow);
    // As `pixel_shuffle2`: one output row `(ch, y, dy)` per task, `scale` source
    // planes interleaved into it.
    let row = |r: usize, dst: &mut [f32]| {
        let (ch, rem) = (r / oh, r % oh);
        let (dy, y) = (rem % scale, rem / scale);
        for dx in 0..scale {
            let src = &inp[((ch * scale * scale + dy * scale + dx) * h + y) * w..][..w];
            for x in 0..w {
                dst[x * scale + dx] = src[x];
            }
        }
    };
    if par_on(out.len()) {
        out.par_chunks_mut(ow).enumerate().for_each(|(r, dst)| row(r, dst));
    } else {
        for (r, dst) in out.chunks_mut(ow).enumerate() {
            row(r, dst);
        }
    }
}

/// `F.interpolate(x, size=(oh, ow), mode='bicubic', align_corners=False)`, which
/// the compressed_sr head applies to the INPUT plane before the body runs.
///
/// This is not an integer upsample and cannot be one: the source is the PADDED
/// plane and so is the TARGET - 37x29 arrives as 40x32 and becomes 160x128 - so
/// the ratios are 4 and 4 after all, but only because the padding is applied on
/// both sides. (The reference's own `size=(H*scale, W*scale)` uses the dims of the
/// plane it was HANDED, which is the padded one; see the head.) It is a general
/// resample, and it has to be torch's resample to the last bit, because its output
/// is added to the body's before `conv_last` and a different kernel is a different
/// image.
///
/// torch's scheme, confirmed numerically against it (`upsample_bicubic2d`):
/// the source position of output index `i` is `(i + 0.5) * (in / out) - 0.5`;
/// the four taps are `floor(src) - 1 .. floor(src) + 2`, each weighted by the
/// cubic convolution kernel with `a = -0.75`
///
/// ```text
/// w(t) = ((a + 2)t - (a + 3))t^2 + 1        for |t| <= 1
/// w(t) = (((t - 5)t + 8)t - 4) * a          for 1 < |t| < 2
/// ```
///
/// and a tap outside the plane is CLAMPED to the edge sample (torch's border
/// handling; it is not reflection). The two axes are separable, so the weights
/// are computed once per output row and once per output column rather than per
/// output texel.
pub fn bicubic_resize(inp: &[f32], c: usize, h: usize, w: usize, oh: usize, ow: usize,
                     out: &mut [f32]) {
    debug_assert_eq!(inp.len(), c * h * w);
    debug_assert_eq!(out.len(), c * oh * ow);
    #[inline]
    fn kernel(t: f32) -> f32 {
        const A: f32 = -0.75;
        let t = t.abs();
        if t <= 1.0 {
            ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0
        } else if t < 2.0 {
            (((t - 5.0) * t + 8.0) * t - 4.0) * A
        } else {
            0.0
        }
    }
    // The taps are precomputed per output index on each axis: `(index, weight)`
    // pairs into the source, with the index already clamped to the plane.
    let taps = |n_in: usize, n_out: usize| -> Vec<[(usize, f32); 4]> {
        (0..n_out)
            .map(|i| {
                let src = (i as f32 + 0.5) * (n_in as f32 / n_out as f32) - 0.5;
                let base = src.floor();
                let mut t = [(0usize, 0.0f32); 4];
                for (m, slot) in t.iter_mut().enumerate() {
                    let idx = base as isize - 1 + m as isize;
                    slot.0 = idx.clamp(0, n_in as isize - 1) as usize;
                    slot.1 = kernel(src - (base + m as f32 - 1.0));
                }
                t
            })
            .collect()
    };
    let (ty, tx) = (taps(h, oh), taps(w, ow));
    // One output ROW per task, as `upsample2x_nearest` does per row-pair.
    let row = |ch: usize, y: usize, dst: &mut [f32]| {
        let src = &inp[ch * h * w..][..h * w];
        let (t0, t1, t2, t3) = (ty[y][0], ty[y][1], ty[y][2], ty[y][3]);
        for (x, d) in dst.iter_mut().enumerate() {
            let (u0, u1, u2, u3) = (tx[x][0], tx[x][1], tx[x][2], tx[x][3]);
            let mut acc = 0.0f32;
            for (ry, wy) in [t0, t1, t2, t3] {
                let r = &src[ry * w..][..w];
                let pair = |(cx, wx): (usize, f32)| wx * r[cx];
                let (a0, a1, a2, a3) = (pair(u0), pair(u1), pair(u2), pair(u3));
                acc += wy * ((a0 + a1) + (a2 + a3));
            }
            *d = acc;
        }
    };
    if par_on(out.len()) {
        out.par_chunks_mut(ow).enumerate().for_each(|(i, dst)| row(i / oh, i % oh, dst));
    } else {
        for i in 0..c * oh {
            let dst = &mut out[i * ow..(i + 1) * ow];
            row(i / oh, i % oh, dst);
        }
    }
}

/// `F.interpolate(x, scale_factor=2, mode='nearest')`.
pub fn upsample2x_nearest(inp: &[f32], c: usize, h: usize, w: usize, out: &mut [f32]) {
    let (oh, ow) = (h * 2, w * 2);
    debug_assert_eq!(inp.len(), c * h * w);
    debug_assert_eq!(out.len(), c * oh * ow);
    // One PAIR of output rows per task: each source row becomes two adjacent
    // output rows, and a task writes both so the split stays row-aligned.
    let rowpair = |ch: usize, y: usize, dst: &mut [f32]| {
        let src = &inp[(ch * h + y) * w..][..w];
        let (a, b) = dst.split_at_mut(ow);
        for x in 0..w {
            let v = src[x];
            a[x * 2] = v;
            a[x * 2 + 1] = v;
            b[x * 2] = v;
            b[x * 2 + 1] = v;
        }
    };
    if par_on(out.len()) {
        out.par_chunks_mut(2 * ow).enumerate().for_each(|(i, dst)| {
            let (ch, y) = (i / h, i % h);
            rowpair(ch, y, dst);
        });
    } else {
        for i in 0..c * h {
            let (ch, y) = (i / h, i % h);
            let dst = &mut out[i * 2 * ow..(i * 2 + 2) * ow];
            rowpair(ch, y, dst);
        }
    }
}

/// LeakyReLU in place. Elementwise and independent per element, so the split is
/// free.
pub fn leaky_relu(x: &mut [f32], slope: f32) {
    if par_on(x.len()) {
        x.par_iter_mut().for_each(|v| {
            if *v < 0.0 {
                *v *= slope;
            }
        });
    } else {
        for v in x.iter_mut() {
            if *v < 0.0 {
                *v *= slope;
            }
        }
    }
}

/// `y += a`, in place.
///
/// The toolkit's `lg_add` takes two sources and a destination, and its CUDA side
/// has an `lg_add_inplace` with no CPU twin - so the in-place form the resi- duals
/// need is four lines here rather than a second buffer and a copy that would show
/// up as a different rounding on the device. A two-operand loop cannot drift.
pub fn add_into(a: &[f32], y: &mut [f32]) {
    debug_assert_eq!(a.len(), y.len());
    if par_on(y.len()) {
        y.par_iter_mut().zip(a.par_iter()).for_each(|(d, s)| *d += *s);
    } else {
        for i in 0..y.len() {
            y[i] += a[i];
        }
    }
}

/// The toolkit's `lg_channel_layer_norm`, split across cores.
///
/// SAME ARITHMETIC, BIT FOR BIT: the two-pass variance with the sums taken in
/// ascending channel order, then `(x - mean) * rstd * w + b` - which is exactly
/// `lightgpu::ops::cpu::channel_layer_norm`, and `selftest` asserts that the two
/// agree on the nose rather than approximately. What this adds is that positions
/// are independent, so the work can be split.
///
/// WHY IT IS WORTH DUPLICATING. The toolkit's version is one scalar loop over
/// positions, and each position's reduction strides the plane by `hw` - every one
/// of the `c` reads lands on a different cache line - so one core cannot keep the
/// memory system busy and a sixth of the forward's time goes to a loop that only
/// uses one of the machine's cores. This does it in three passes (accumulate the
/// two sums, finish the statistics, apply), which reads the plane one extra time
/// and is otherwise the same work, fully parallel.
pub fn channel_layer_norm(x: &[f32], w: &[f32], b: &[f32], y: &mut [f32],
                          c: usize, hw: usize, eps: f32) {
    if !par_on(hw) {
        lg::channel_layer_norm(x, w, b, y, c, hw, eps);
        return;
    }
    // ONE PASS OVER `x`, IN BLOCKS OF POSITIONS. `x` is `[c][hw]`, so a position's
    // `c` values are strided - but a BLOCK of positions per channel is contiguous,
    // and a block of 64 positions' two accumulators is 512 bytes, which stays in L1
    // across all `c` channel iterations. So the plane is read once, in order, and
    // each block's running sums never leave cache.
    //
    // The earlier version put the CHANNEL loop outside and swept the whole plane
    // per channel, which is also contiguous but touches the two `hw`-long
    // accumulator arrays `c` times - 53 MB of traffic at 128x128 for a 13 MB plane.
    // This is the same arithmetic in the same order per position (ascending `ch`),
    // so the values are bit-identical - `selftest` asserts exactly that against the
    // toolkit's sequential version.
    const LN_BLK: usize = 64;
    let mut s1 = vec![0.0f32; hw];
    let mut s2 = vec![0.0f32; hw];
    s1.par_chunks_mut(LN_BLK)
        .zip(s2.par_chunks_mut(LN_BLK))
        .enumerate()
        .for_each(|(bi, (a, bb))| {
            let p0 = bi * LN_BLK;
            for ch in 0..c {
                let plane = &x[ch * hw + p0..];
                for j in 0..a.len() {
                    let v = plane[j];
                    a[j] += v;
                    bb[j] += v * v;
                }
            }
        });
    let n = c as f32;
    let (mean, rstd): (Vec<f32>, Vec<f32>) = s1
        .par_iter()
        .zip(s2.par_iter())
        .map(|(a, bb)| {
            let mean = *a / n;
            let var = *bb / n - mean * mean;
            (mean, 1.0 / (var.max(0.0) + eps).sqrt())
        })
        .unzip();
    // The apply, one channel plane per task: contiguous reads, contiguous writes,
    // and the two statistics tables are read-only.
    y.par_chunks_mut(hw).enumerate().for_each(|(ch, dst)| {
        let plane = &x[ch * hw..(ch + 1) * hw];
        let (wv, bv) = (w[ch], b[ch]);
        for p in 0..hw {
            dst[p] = (plane[p] - mean[p]) * rstd[p] * wv + bv;
        }
    });
}

/// `exp` for the softmax, which is where the time is.
///
/// MEASURED, NOT GUESSED: the softmax evaluates `n` exps per query, which is
/// `windows * heads * n * n * blocks` of them - 255 million per 128x128 image at
/// this model's shape - and that alone was 1.4 Gcycles of the attention kernel's
/// 6.8. Replacing it with this took the kernel to 5.4 Gcycles, and what is left is
/// 30.7 GFLOP in 5.4 Gcycles, i.e. 5.7 FLOP per cycle - still not a rate any FMA
/// loop produces, so the kernel is still transcendental-bound rather than
/// multiply-bound, just less so.
///
/// This is `exp(x) = 2^(x * log2 e)` with the integer part taken exactly out of the
/// exponent field and a degree-6 series for `2^f`, which costs a fraction of
/// libm's `expf` and lands within 2e-7 relative over the range the softmax uses
/// (it is always fed `s - max <= 0`), against the 2e-3 the fixtures are held to and
/// the 3e-6 the engine currently achieves.
///
/// A VECTORISED VERSION WOULD BE BETTER STILL and is the obvious next step: this is
/// scalar, and the same polynomial over eight lanes with AVX2 would be several
/// times faster again.
#[inline]
pub fn fast_exp(x: f32) -> f32 {
    const LOG2_E: f32 = 1.442_695_040_888_963_4;
    let t = x * LOG2_E;
    // ROUND TO NEAREST, NOT FLOOR. `2^f` is then approximated on f in [-0.5, 0.5]
    // instead of [0, 1), where the same number of terms is worth three more bits:
    // the truncation error of the degree-6 series below is (ln2 * 0.5)^7 / 7! =
    // 6e-8 relative, against 1.3e-3 for a degree-4 series on the wider interval -
    // which is what an earlier version of this measured, and what the selftest's
    // 2e-5 tolerance caught.
    let n = (t + 0.5).floor() - 0.5;
    // Below the smallest normal f32 the result is zero for any practical purpose;
    // clamping here also keeps the exponent field below from wrapping, which is
    // what the masked entries (s - max around -100) would otherwise do.
    if n < -126.0 {
        return 0.0;
    }
    let f = t - n;
    let p = 1.0
        + f * (0.693_147_18
            + f * (0.240_226_51
                + f * (0.055_504_11
                    + f * (0.009_618_129
                        + f * (0.001_333_355_8 + f * 0.000_154_035_3)))));
    let scale = f32::from_bits((((n as i32) + 127) as u32) << 23);
    scale * p
}

/// The toolkit's erf GELU, in place. `nn.GELU()`'s default is the erf form, not
/// the tanh approximation, and the toolkit's twin is the same expression the CUDA
/// kernel uses - so this is deliberately that function and not a `tanh` rewrite.
pub fn gelu_erf_inplace(x: &mut [f32]) {
    let g = |v: &mut f32| *v = 0.5 * *v * (1.0 + lg::erf(*v * 0.70710678118654752440));
    if par_on(x.len()) {
        x.par_iter_mut().for_each(g);
    } else {
        x.iter_mut().for_each(g);
    }
}

// ---------------------------------------------------------------------------
// The window pair. `gather` is the reference's `window_partition` with the shift
// rolled in and an optional per-token LayerNorm; `scatter` is `window_reverse`
// with the same roll, so a gather/scatter pair is the identity on the plane.
// ---------------------------------------------------------------------------

/// Read the plane into the token layout, optionally normalising each token.
///
/// `x` is NCHW: the channel stride is `hp * wp`, not 1. Writing the index as
/// `(y * wp + x) * c` is the natural mistake - it is the NHWC offset - and it is
/// invisible to every self-consistency check, because `scatter` makes the same
/// one and the pair stays mutually inverse. What it changes is WHICH token each
/// position becomes, so the projections are fed a permutation of the right
/// numbers and the whole network is quietly wrong.
pub fn gather(
    plan: &Plan,
    x: &[f32],
    shift: usize,
    norm: Option<(&[f32], &[f32])>,
    tok: &mut [f32],
) {
    let c = plan.c;
    let hw = plan.hp * plan.wp;
    // One WINDOW per task: `plan.token(wi, t)` is `wi * n + t`, so a window's
    // tokens are one contiguous run of `n * c` and `par_chunks_mut` hands each task
    // exactly the slice it writes. Nothing is shared but the read-only plane.
    let window = |wi: usize, dst: &mut [f32]| {
        for t in 0..plan.n {
            let (y, xx) = plan.index(wi, t, shift);
            let p = y * plan.wp + xx;
            let dst = &mut dst[t * c..t * c + c];
            match norm {
                Some((w, b)) => {
                    // nn.LayerNorm over the channel axis: the two-pass variance
                    // the toolkit's `lg_layer_norm` twins also use.
                    let mean = (0..c).map(|i| x[i * hw + p]).sum::<f32>() / c as f32;
                    let var = (0..c)
                        .map(|i| {
                            let d = x[i * hw + p] - mean;
                            d * d
                        })
                        .sum::<f32>()
                        / c as f32;
                    let rstd = 1.0 / (var + LN_EPS).sqrt();
                    for i in 0..c {
                        dst[i] = (x[i * hw + p] - mean) * rstd * w[i] + b[i];
                    }
                }
                None => {
                    for i in 0..c {
                        dst[i] = x[i * hw + p];
                    }
                }
            }
        }
    };
    if par_on(tok.len()) {
        tok.par_chunks_mut(plan.n * c).enumerate().for_each(|(wi, dst)| window(wi, dst));
    } else {
        for (wi, dst) in tok.chunks_mut(plan.n * c).enumerate() {
            window(wi, dst);
        }
    }
}

/// Write the token layout back into the plane, undoing the shift.
pub fn scatter(plan: &Plan, tok: &[f32], shift: usize, x: &mut [f32]) {
    let c = plan.c;
    let hw = plan.hp * plan.wp;
    // One CHANNEL PLANE per task here, not one window: the destination is the
    // plane, whose layout is `[c][hp][wp]`, so a window's tokens are scattered
    // across the whole buffer while a single channel's are one contiguous run of
    // `hw`. Each output position is written by exactly one (window, token) pair.
    let channel = |i: usize, dst: &mut [f32]| {
        for wi in 0..plan.nw {
            for t in 0..plan.n {
                let (y, xx) = plan.index(wi, t, shift);
                dst[y * plan.wp + xx] = tok[plan.token(wi, t) * c + i];
            }
        }
    };
    if par_on(x.len()) {
        x.par_chunks_mut(hw).enumerate().for_each(|(i, dst)| channel(i, dst));
    } else {
        for (i, dst) in x.chunks_mut(hw).enumerate() {
            channel(i, dst);
        }
    }
}

/// Window attention: cosine attention with a relative-position bias and, on
/// shifted blocks, the 0/-100 region mask.
///
/// `qkv` is [3][tokens][c] in q, k, v order; `out` is [tokens][c].
///
/// ONE WINDOW PER TASK. A window's `n * c` floats are one contiguous run of `out`
/// (`plan.token(wi, t)` is `wi * n + t`), so each task writes its own slice and the
/// split needs no locking. Inside the task the queries of the window are done in
/// order, which is what makes the KEY NORMS worth hoisting: a key's norm is a
/// property of the key, and the reference recomputes it for every (query, key)
/// pair. Hoisting it to once per (window, head, key) removes a third of the inner
/// loop's work, and the per-task logits row (`n` floats) and norms are allocated
/// once per window instead of once per token.
///
/// THE ROW STAYS IN ONE THREAD, here and on the device, and for the same reason:
/// a block-wide reduction would make the summation order depend on the block size,
/// so the device and CPU answers would differ by more than rounding and the
/// selftest that compares them could only use a looser tolerance than everything
/// else in it.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    plan: &Plan,
    heads: usize,
    head_dim: usize,
    logit_scale: &[f32],
    cpb: &[f32],
    qkv: &[f32],
    shift: usize,
    out: &mut [f32],
) {
    let c = plan.c;
    let n = plan.n;
    let tok_c = plan.tokens() * c;
    let masked = shift != 0;
    // The reference clamps logit_scale at log(100) BEFORE the exp, and there is NO
    // 1/sqrt(head_dim) anywhere: cosine attention is already normalised, so the
    // scale is the learned temperature alone. Adding a 1/sqrt(d) here is the single
    // easiest way to get a Swin-V2 block subtly wrong, and it is what
    // `tests/parity.rs` caught. Folded once per head instead of once per row.
    let ls: Vec<f32> = (0..heads).map(|hd| logit_scale[hd].min(100f32.ln()).exp()).collect();

    // THE BIAS TABLE, TRANSPOSED TO [head][query][key]. It arrives as
    // `[query][key][head]`, so for a fixed head the inner loop's read of
    // `cpb[(q * n + k) * heads + hd]` walks with a stride of `heads` floats - it
    // touches the whole `heads * n * n` table (98 KB at 6 heads, 64 tokens) to use
    // one head's 16 KB of it, once per (window, head). That is 98 KB of L2 traffic
    // for 16 KB of useful data, and at 62,000 (window, head) pairs per image it is
    // more traffic than the arithmetic itself. Transposed, one head's table is
    // contiguous and stays in L1 for the whole query loop.
    //
    // This is a pure re-indexing: the same numbers are read in the same order by
    // the same accumulation, so the result is unchanged bit for bit.
    let mut cpbt = vec![0.0f32; heads * n * n];
    for q in 0..n {
        for k in 0..n {
            let src = (q * n + k) * heads;
            for hd in 0..heads {
                cpbt[(hd * n + q) * n + k] = cpb[src + hd];
            }
        }
    }

    // PER-WINDOW TABLES, FILLED ONCE. `plan.token` and `plan.region` are cheap per
    // call but they were being called per (query, key) PAIR - `region` in
    // particular is four integer divisions and a modulo by loop-variant values, and
    // at 289 windows x 6 heads x 64 x 64 pairs x 36 blocks that is half a billion
    // divisions per image. Neither depends on the head, and `token` does not even
    // depend on the query, so both are filled once per window here.
    let window = |wi: usize, outw: &mut [f32], scores: &mut [f32], kn: &mut [f32],
                  ktok: &mut [usize], kreg: &mut [(u8, u8)]| {
        for k in 0..n {
            ktok[k] = plan.token(wi, k);
            if masked {
                kreg[k] = plan.region(wi, k, shift);
            }
        }
        for hd in 0..heads {
            let off = hd * head_dim;
            // The key norms, once per (head, key) for the whole window rather than
            // once per (query, key) as the reference computes them. Same sum in the
            // same order, so the values are identical - it is redundant work that
            // is removed, not arithmetic that is changed.
            for k in 0..n {
                let kbase = tok_c + ktok[k] * c + off;
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    let v = qkv[kbase + d];
                    s += v * v;
                }
                kn[k] = s.sqrt().max(1e-12);
            }
            for q in 0..n {
                let reg_q = if masked { kreg[q] } else { (0, 0) };
                let orow = &mut outw[q * c..q * c + c];
                let qbase = ktok[q] * c + off;
                let qv = &qkv[qbase..qbase + head_dim];
                let mut qs = 0.0f32;
                for d in 0..head_dim {
                    qs += qv[d] * qv[d];
                }
                let qn = qs.sqrt().max(1e-12);
                for k in 0..n {
                    let kbase = tok_c + ktok[k] * c + off;
                    let kv = &qkv[kbase..kbase + head_dim];
                    // Four accumulators: one FMA chain is latency-bound.
                    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let mut d = 0;
                    while d + 4 <= head_dim {
                        a0 += qv[d] * kv[d];
                        a1 += qv[d + 1] * kv[d + 1];
                        a2 += qv[d + 2] * kv[d + 2];
                        a3 += qv[d + 3] * kv[d + 3];
                        d += 4;
                    }
                    let mut tail = 0.0f32;
                    while d < head_dim {
                        tail += qv[d] * kv[d];
                        d += 1;
                    }
                    // The normalisation is folded into the sum of products rather than
                    // into each factor: the reference divides both vectors by their
                    // norms and then dots them, and the two differ by ~1e-7 relative.
                    let dot = ((a0 + a1) + (a2 + a3) + tail) / (qn * kn[k]);
                    let mut s = dot * ls[hd] + cpbt[(hd * n + q) * n + k];
                    if masked && reg_q != kreg[k] {
                        s -= 100.0;
                    }
                    scores[k] = s;
                }
                // Softmax, max-subtracted as torch's is.
                let mut mx = f32::NEG_INFINITY;
                for k in 0..n {
                    mx = mx.max(scores[k]);
                }
                let mut sum = 0.0f32;
                for k in 0..n {
                    scores[k] = fast_exp(scores[k] - mx);
                    sum += scores[k];
                }
                let inv = 1.0 / sum;
                for d in 0..head_dim {
                    // Four accumulators again, over `k` this time.
                    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let mut k = 0;
                    while k + 4 <= n {
                        a0 += scores[k] * qkv[2 * tok_c + ktok[k] * c + off + d];
                        a1 += scores[k + 1] * qkv[2 * tok_c + ktok[k + 1] * c + off + d];
                        a2 += scores[k + 2] * qkv[2 * tok_c + ktok[k + 2] * c + off + d];
                        a3 += scores[k + 3] * qkv[2 * tok_c + ktok[k + 3] * c + off + d];
                        k += 4;
                    }
                    let mut tail = 0.0f32;
                    while k < n {
                        tail += scores[k] * qkv[2 * tok_c + ktok[k] * c + off + d];
                        k += 1;
                    }
                    orow[off + d] = ((a0 + a1) + (a2 + a3) + tail) * inv;
                }
            }
        }
    };
    if par_on(out.len()) {
        out.par_chunks_mut(n * c).enumerate().for_each(|(wi, outw)| {
            let mut scores = vec![0.0f32; n];
            let mut kn = vec![0.0f32; n];
            let mut ktok = vec![0usize; n];
            let mut kreg = vec![(0u8, 0u8); n];
            window(wi, outw, &mut scores, &mut kn, &mut ktok, &mut kreg);
        });
    } else {
        let mut scores = vec![0.0f32; n];
        let mut kn = vec![0.0f32; n];
        let mut ktok = vec![0usize; n];
        let mut kreg = vec![(0u8, 0u8); n];
        for (wi, outw) in out.chunks_mut(n * c).enumerate() {
            window(wi, outw, &mut scores, &mut kn, &mut ktok, &mut kreg);
        }
    }
}

/// What `forward` can call for each activation it produces.
pub type Dump<'a> = &'a mut dyn FnMut(&str, &[f32], usize);

/// `VmRSS` right now, in MiB - for `SWIN2SR_DEBUG_RSS`, which is how the peak of a
/// host run is attributed to a PHASE rather than to a total. A peak-RSS fit says a
/// whole run costs `k` bytes a padded pixel and cannot say which buffer that is;
/// marking the phases says where the curve steps up and where it comes back down.
///
/// Off unless the variable is set, and one line a mark: this is a debugging aid,
/// not a report.
fn rss_mark(tag: &str) {
    if std::env::var_os("SWIN2SR_DEBUG_RSS").is_none() {
        return;
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let rss = status
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|| "?".into());
    eprintln!("rss {tag}: {rss}");
}

/// One forward pass over an already mean-adjusted plane. The input is `[3][h][w]`
/// at the image's own size: the padding is this function's first step (it is the
/// reference's too), so nothing above it has to know about windows.
///
/// `input` is [3][h][w] in [0,1]; the result is [3][h*scale][w*scale], denormalised
/// and cropped. `dump` receives every activation in the `torch` module order the
/// reference names them in, which is how a divergence is located by stage rather
/// than by bisection on the final image.
/// The whole-image forward pass. Returns the restored image, and - for a head
/// that produces one - the secondary `aux` image, already through its epilogue.
pub fn forward(
    wt: &Weights,
    plan: &Plan,
    input: &[f32],
    stock: &mut ScratchOwned,
    mut dump: Option<Dump>,
) -> Result<(Vec<f32>, Option<Vec<f32>>), String> {
    // The scratch is owned by the caller so the reusable one survives the call:
    // see `ScratchOwned`. The body itself reaches everywhere through `scr.buf`.
    let scr = &mut *stock.buf;
    let c = plan.c;
    let hw = plan.plane();
    let tok = plan.tokens() * c;
    let hidden = wt.mlp_ratio * c;
    let tokens = plan.tokens();

    macro_rules! dmp {
        ($name:expr, $buf:expr) => {
            if let Some(f) = dump.as_deref_mut() {
                f($name, $buf, c);
            }
        };
    }
    // The destination of an op that destroys its input: a buffer taken out of the
    // scratch, which the caller puts back. `take` rather than a borrow because
    // these ops take `&[f32]` and `&mut [f32]` over the SAME buffer's memory only
    // when the source is elsewhere - and the source is always elsewhere here, so
    // the take is what makes the borrow checker see it.
    macro_rules! take {
        ($buf:expr) => {
            std::mem::take(&mut $buf)
        };
    }

    // 1. Shallow feature extraction. `conv_first` runs on the PADDED plane, and
    //    the mean subtraction has already happened (the caller does it, exactly
    //    where the reference does: before the network, after the padding).
    rss_mark("forward start");
    let padded = pad_reflect(input, 3, plan.h, plan.w, plan.hp, plan.wp);
    let mut cur = take!(scr.cur);
    conv3x3(&padded, 3, plan.hp, plan.wp, wt.t("conv_first.weight"), c, wt.t("conv_first.bias"),
            &mut cur);
    let body_res = cur.clone();
    rss_mark("after conv_first (padded + cur + body_res)");
    if let Some(f) = dump.as_deref_mut() {
        f("input_padded", &padded, 3);
        f("conv_first", &cur, c);
    }
    // `forward_features` opens with the TOP-LEVEL `patch_embed`: a 1x1 conv to the
    // embedding width, then a LayerNorm over the channels. Leaving it out is
    // plausible-looking and wrong everywhere - the first block then reads an
    // un-normalised activation. Note the RSTB stages' own `patch_embed` (below)
    // has NO norm: only this one does, and that asymmetry is in the checkpoint.
    let mut acc = take!(scr.acc);
    conv1x1(&cur, c, hw, wt.t("patch_embed.proj.weight"), c, wt.t("patch_embed.proj.bias"),
            &mut acc, &mut scr.ctmp, &mut scr.wpack);
    channel_layer_norm(&acc, wt.t("patch_embed.norm.weight"), wt.t("patch_embed.norm.bias"),
                        &mut cur, c, hw, LN_EPS);
    dmp!("patch_embed", &cur);
    rss_mark("after patch_embed");

    // 2. The RSTB stages.
    let mut stage = take!(scr.stage);
    let mut res = take!(scr.res);
    let mut ctmp = take!(scr.ctmp);
    let mut wpack = take!(scr.wpack);
    let mut tokbuf = take!(scr.tok);
    let mut tok2 = take!(scr.tok2);
    let mut attn = take!(scr.attn);
    let mut qkv = take!(scr.qkv);
    let mut mlp = take!(scr.mlp);
    let mut xt = take!(scr.xt);
    let zero_bias = take!(scr.zero_bias);
    for (s, &depth) in wt.depths.iter().enumerate() {
        let (cw, cb) = (format!("layers.{s}.conv.weight"), format!("layers.{s}.conv.bias"));
        let (pw, pb) = (format!("layers.{s}.patch_embed.proj.weight"),
                        format!("layers.{s}.patch_embed.proj.bias"));
        stage.copy_from_slice(&cur);
        res.copy_from_slice(&cur);
        for b in 0..depth {
            let shift = plan.shift_for(b);
            let p = format!("layers.{s}.residual_group.blocks.{b}");
            // A POST-NORM block, which is where Swin2SR departs from the Swin-V2
            // it is built on: the attention reads the RAW activation, and `norm1`
            // is applied to the attention's OUTPUT, just before the residual add
            // (`x = shortcut + norm1(attn_out)`). Pre-norm blocks are the norm for
            // this family of models, so applying norm1 on the way IN is the
            // natural mistake - it is what this engine did, and it produced a
            // plausible-looking image that was wrong everywhere.
            gather(plan, &cur, shift, None, &mut tokbuf);
            // The fused qkv as three [c][c] matmuls, whose weight rows are the
            // slices of the checkpoint's [3c][c] matrix. q and v carry their
            // biases; k's is the zero column of the reference's torch.cat. The
            // three are stored contiguously as [3][tokens][c] because that is the
            // order the attention kernel reads them in.
            {
                linear(&tokbuf, tokens, c, wt.t(&format!("{p}.attn.qkv.wq")), c,
                       wt.t(&format!("{p}.attn.qkv.q_bias")), &mut qkv[..tok], &mut xt,
                       &mut wpack);
                linear(&tokbuf, tokens, c, wt.t(&format!("{p}.attn.qkv.wk")), c,
                       &zero_bias, &mut qkv[tok..2 * tok], &mut xt, &mut wpack);
                linear(&tokbuf, tokens, c, wt.t(&format!("{p}.attn.qkv.wv")), c,
                       wt.t(&format!("{p}.attn.qkv.v_bias")), &mut qkv[2 * tok..3 * tok], &mut xt,
                       &mut wpack);
            };
            dmp!(&format!("layers.{s}.blocks.{b}.qkv"), &qkv);
            attention(plan, wt.heads, wt.head_dim, wt.t(&format!("{p}.attn.logit_scale")),
                      wt.t(&format!("{p}.attn.cpb_pre")), &qkv, shift, &mut attn);
            dmp!(&format!("layers.{s}.blocks.{b}.attn_scores_apply"), &attn);
            // The output projection: the reference's WindowAttention ends with
            // `x = self.proj(x)` (nn.Linear, WITH a bias) on the window-major
            // tokens, before window_reverse.
            linear(&attn, tokens, c, wt.t(&format!("{p}.attn.proj.weight")), c,
                   wt.t(&format!("{p}.attn.proj.bias")), &mut tok2, &mut xt, &mut wpack);
            dmp!(&format!("layers.{s}.blocks.{b}.attn"), &tok2);
            scatter(plan, &tok2, shift, &mut acc);
            // norm1 on the attention output, then the residual. The reference
            // normalises the TOKEN layout, but a LayerNorm over the channel axis
            // is per position, so the plane gives the same numbers.
            channel_layer_norm(&acc, wt.t(&format!("{p}.norm1.weight")),
                               wt.t(&format!("{p}.norm1.bias")), &mut tok2, c, hw, LN_EPS);
            dmp!(&format!("layers.{s}.blocks.{b}.norm1"), &tok2);
            add_into(&tok2, &mut res);
            dmp!(&format!("layers.{s}.blocks.{b}.after_attn"), &res);
            // x = x + norm2(mlp(x)): the MLP also reads the raw activation.
            gather(plan, &res, shift, None, &mut tokbuf);
            linear(&tokbuf, tokens, c, wt.t(&format!("{p}.mlp.fc1.weight")), hidden,
                   wt.t(&format!("{p}.mlp.fc1.bias")), &mut mlp, &mut xt, &mut wpack);
            gelu_erf_inplace(&mut mlp);
            linear(&mlp, tokens, hidden, wt.t(&format!("{p}.mlp.fc2.weight")), c,
                   wt.t(&format!("{p}.mlp.fc2.bias")), &mut attn, &mut xt, &mut wpack);
            scatter(plan, &attn, shift, &mut acc);
            channel_layer_norm(&acc, wt.t(&format!("{p}.norm2.weight")),
                               wt.t(&format!("{p}.norm2.bias")), &mut tok2, c, hw, LN_EPS);
            dmp!(&format!("layers.{s}.blocks.{b}.norm2"), &tok2);
            add_into(&tok2, &mut res);
            cur.copy_from_slice(&res);
            dmp!(&format!("layers.{s}.blocks.{b}"), &cur);
        }
        // patch_unembed is a reshape, so the blocks' output is already the image
        // layout: the 3x3 conv, then the 1x1 patch_embed conv, then the stage's
        // residual, all without leaving it.
        conv3x3(&cur, c, plan.hp, plan.wp, wt.t(&cw), c, wt.t(&cb), &mut acc);
        conv1x1(&acc, c, hw, wt.t(&pw), c, wt.t(&pb), &mut cur, &mut ctmp, &mut wpack);
        add_into(&stage, &mut cur);
        dmp!(&format!("layers.{s}"), &cur);
        rss_mark(&format!("after stage {s}"));
    }
    // The stage loop is over and `stage` will not be read again, so it goes back to
    // the pool NOW: the final LayerNorm's plane and then the head's octave planes
    // are what is about to be allocated, and this is the block that pays for them.
    let body = &mut stock.buf;
    stock.pool.push(std::mem::take(&mut stage));

    // 3. The final LayerNorm, patch_unembed, and the body residual.
    let mut nbuf = body.take_block(hw * c);
    channel_layer_norm(&cur, wt.t("norm.weight"), wt.t("norm.bias"), &mut nbuf, c, hw, LN_EPS);
    dmp!("norm", &nbuf);
    rss_mark("after final norm + nbuf");
    conv3x3(&nbuf, c, plan.hp, plan.wp, wt.t("conv_after_body.weight"), c,
            wt.t("conv_after_body.bias"), &mut acc);
    // Same reasoning: `nbuf` is one padded plane and the head is next. It is a
    // `Vec` of another size class than the planes, so it is dropped into the pool
    // by `Drop` here rather than by hand.
    stock.pool.push(nbuf);
    // `x = self.conv_after_body(self.forward_features(x)) + x`, where the `x` on
    // the right is the CONV_FIRST output - not the body's. Adding the body's
    // output to itself is off by exactly one skip connection and looks plausible
    // at every stage in between.
    add_into(&body_res, &mut acc);
    cur.copy_from_slice(&acc);
    dmp!("conv_after_body", &cur);
    rss_mark("body done (conv_after_body)");

    // 4. The reconstruction head, and the denormalisation/crop. The head opens
    //    with `conv_before_upsample.0` - the last layer of the body in the
    //    reference - because that is what lets it free the body's buffers after
    //    reading `cur` once. Everything the body no longer needs is handed back to
    //    the pool HERE, before the head allocates its octave planes.
    body.cur = cur;
    body.acc = acc;
    let owned = ScratchOwned {
        buf: Box::new(Scratch {
            cur: std::mem::take(&mut body.cur),
            res,
            stage,
            acc: std::mem::take(&mut body.acc),
            tok: tokbuf,
            tok2,
            attn,
            qkv,
            mlp,
            xt,
            ctmp,
            zero_bias,
            wpack,
        }),
        pool: std::mem::take(&mut stock.pool),
    };
    // The head hands the scratch back, and the caller stores it: that is what lets
    // the next forward - or the next tile - start from these blocks.
    // The compressed head's OTHER input: the bicubic resample of the padded plane,
    // resized to the output grid and convolved to `feat` channels. It is computed
    // HERE because `padded` lives here - the reference computes the branch before
    // the body and adds it to the head's octaves at the end - and handed to the
    // head, which is the only other place it is needed.
    let bicubic = match wt.upsampler {
        Upsampler::PixelShuffleAux => {
            let feat = wt.t("conv_bicubic.weight").len() / (3 * 9);
            // The PADDED output grid, not the original one. The reference's `H, W`
            // are the dims of the image it was HANDED - and the task wrapper hands
            // it the padded plane - so its `F.interpolate(x, size=(H*scale, W*scale))`
            // runs on the padded geometry and its own `[:H*scale, :W*scale]` crop is
            // a no-op. Resizing to the original size instead shifts every tap.
            let (oh, ow) = (plan.hp * wt.scale, plan.wp * wt.scale);
            let mut resized = vec![0.0f32; 3 * oh * ow];
            bicubic_resize(&padded, 3, plan.hp, plan.wp, oh, ow, &mut resized);
            if let Some(f) = dump.as_deref_mut() {
                f("bicubic_resize", &resized, 3);
            }
            let mut b = vec![0.0f32; feat * oh * ow];
            conv3x3(&resized, 3, oh, ow, wt.t("conv_bicubic.weight"), feat,
                    wt.t("conv_bicubic.bias"), &mut b);
            if let Some(f) = dump.as_deref_mut() {
                f("conv_bicubic", &b, feat);
            }
            rss_mark("after conv_bicubic");
            Some(b)
        }
        _ => None,
    };
    if bicubic.is_some() {
        // `padded` is a plane and the body's blocks are the ones the head is about
        // to need; the bicubic branch has already copied everything it reads.
        drop(padded);
    }
    let (mut stock2, planes, aux) = head(wt, plan, owned, bicubic.as_deref(), dump)?;
    // Every field `forward` took out is emptied, and the blocks that were in them
    // went to the pool (some by hand, some by `free_body`). Put the invariant back
    // before the handle is stored, or the next forward of the SAME size reads
    // empty buffers - see `ScratchOwned::refill`.
    stock2.refill(plan, wt.mlp_ratio);
    *stock = stock2;
    rss_mark("before finish");
    let out = finish(wt, plan, &planes, plan.wp * wt.scale);
    let aux_out = aux.map(|a| finish_aux(wt, plan, &a));
    rss_mark("after finish");
    Ok((out, aux_out))
}

/// The reconstruction head: `planes` is [3][hp*scale][wp*scale], still denormalised.
///
/// THE HEAD OPENS WITH THE BODY'S LAST LAYER, and it takes the scratch by value so
/// it can free what it does not need. `conv_before_upsample.0` reads `cur` once,
/// and every buffer behind `cur` - `res`, `stage`, `body`, the token layouts - is
/// dead from there on. Split into two functions the body would have had to hold all
/// of them alive across the call (`Scratch` is borrowed, so a callee cannot free
/// its fields), which is worth several padded planes of peak RSS for no other
/// reason than where the line between the two functions was drawn.
/// The head. `bicubic` carries the compressed head's pre-upsample branch - the
/// bicubic-resized input, already convolved to `feat` channels on the OUTPUT grid
/// - which `forward` computes because it is the only place the padded input plane
/// is in scope. It is `Some` for `PixelShuffleAux` and `None` for every other
/// head, and the same is true of the returned aux plane.
fn head(wt: &Weights, plan: &Plan, mut owned: ScratchOwned, bicubic: Option<&[f32]>,
        mut dump: Option<Dump>)
    -> Result<(ScratchOwned, Vec<f32>, Option<Vec<f32>>), String> {
    // `scr` reaches the buffers; `owned` is what goes back to the caller. The
    // borrow has to end before the return, hence the scope.
    let scr = &mut *owned.buf;
    let c = plan.c;
    let hw = plan.plane();
    let scale = wt.scale;
    let _ = &scr;
    // `feat` is the width of `conv_before_upsample`, which is the FIRST layer of
    // the pixel-shuffle and nearest+conv heads and does not exist at all in the
    // direct one - so it is read per branch, not here. Reading it here panicked on
    // the lightweight checkpoint, which no earlier fixture reached.
    let mut planes;
    // The compressed head returns a SECOND image: the low-resolution reconstruction
    // it emits beside the upsampled one, at the padded plane's geometry. Every
    // other head leaves this `None`, which is what the fixture's `flags` bit 0 and
    // the backends' aux accessor both key off.
    let mut aux_out: Option<Vec<f32>> = None;
    match wt.upsampler {
        Upsampler::PixelShuffle => {
            let feat = wt.t("conv_before_upsample.0.weight").len() / (c * 9);
            let mut a = vec![0.0f32; feat * hw];
            conv3x3(&scr.cur, c, plan.hp, plan.wp, wt.t("conv_before_upsample.0.weight"), feat,
                    wt.t("conv_before_upsample.0.bias"), &mut a);
            leaky_relu(&mut a, 0.01);
            if let Some(f) = dump.as_deref_mut() {
                f("conv_before_upsample", &a, feat);
            }
            rss_mark("head: after conv_before_upsample (a live)");
            // `cur` is dead: everything the head needs is in `a` now. The rest of
            // the scratch is dead too, and the four token buffers are the biggest
            // of them.
            scr.free_body();
            rss_mark("head: after free_body");
            let (mut h2, mut w2) = (plan.hp, plan.wp);
            for o in 0..wt.upsampler.octaves(scale) {
                let (uw, ub) = (format!("upsample.{}.weight", 2 * o), format!("upsample.{}.bias", 2 * o));
                // The conv writes the shuffled layout directly, so the
                // `4 * feat`-channel intermediate that used to sit between the two
                // is not allocated at all. See `conv3x3_shuffle2`.
                let mut shuf = vec![0.0f32; feat * 4 * h2 * w2];
                conv3x3_shuffle2(&a[..feat * h2 * w2], feat, h2, w2, wt.t(&uw), feat, wt.t(&ub),
                                 &mut shuf);
                if let Some(f) = dump.as_deref_mut() {
                    f(&format!("upsample.{o}"), &shuf, feat);
                }
                a = shuf;
                h2 *= 2;
                w2 *= 2;
                rss_mark(&format!("head: after octave {o} ({}x{})", h2, w2));
            }
            planes = vec![0.0f32; 3 * h2 * w2];
            rss_mark("head: after the final planes alloc");
            conv3x3(&a[..feat * h2 * w2], feat, h2, w2, wt.t("conv_last.weight"), 3,
                    wt.t("conv_last.bias"), &mut planes);
            rss_mark("head: after conv_last");
        }
        Upsampler::PixelShuffleDirect => {
            let out_ch = 3 * scale * scale;
            let mut up = vec![0.0f32; out_ch * hw];
            conv3x3(&scr.cur, c, plan.hp, plan.wp, wt.t("upsample.0.weight"), out_ch,
                    wt.t("upsample.0.bias"), &mut up);
            planes = vec![0.0f32; 3 * plan.hp * scale * plan.wp * scale];
            pixel_shuffle(&up, 3, plan.hp, plan.wp, scale, &mut planes);
        }
        Upsampler::NearestConv => {
            let feat = wt.t("conv_before_upsample.0.weight").len() / (c * 9);
            let mut a = vec![0.0f32; feat * hw];
            conv3x3(&scr.cur, c, plan.hp, plan.wp, wt.t("conv_before_upsample.0.weight"), feat,
                    wt.t("conv_before_upsample.0.bias"), &mut a);
            // The body's activation uses the DEFAULT LeakyReLU slope (0.01); the
            // three convs on the way out use self.lrelu (0.2). Both are in the
            // reference and neither is the other.
            leaky_relu(&mut a, 0.01);
            if let Some(f) = dump.as_deref_mut() {
                f("conv_before_upsample", &a, feat);
            }
            scr.free_body();
            let (mut h2, mut w2) = (plan.hp, plan.wp);
            let mut up = vec![0.0f32; feat * 4 * h2 * w2];
            upsample2x_nearest(&a[..feat * h2 * w2], feat, h2, w2, &mut up);
            h2 *= 2;
            w2 *= 2;
            let mut b = vec![0.0f32; feat * h2 * w2];
            conv3x3(&up, feat, h2, w2, wt.t("conv_up1.weight"), feat, wt.t("conv_up1.bias"), &mut b);
            leaky_relu(&mut b, 0.2);
            let mut up2 = vec![0.0f32; feat * 4 * h2 * w2];
            upsample2x_nearest(&b[..feat * h2 * w2], feat, h2, w2, &mut up2);
            h2 *= 2;
            w2 *= 2;
            let mut c2 = vec![0.0f32; feat * h2 * w2];
            conv3x3(&up2, feat, h2, w2, wt.t("conv_up2.weight"), feat, wt.t("conv_up2.bias"), &mut c2);
            leaky_relu(&mut c2, 0.2);
            let mut hr = vec![0.0f32; feat * h2 * w2];
            conv3x3(&c2, feat, h2, w2, wt.t("conv_hr.weight"), feat, wt.t("conv_hr.bias"), &mut hr);
            leaky_relu(&mut hr, 0.2);
            planes = vec![0.0f32; 3 * h2 * w2];
            conv3x3(&hr, feat, h2, w2, wt.t("conv_last.weight"), 3, wt.t("conv_last.bias"), &mut planes);
        }
        // The compressed head: the classical head with a bicubic shortcut around it
        // and a second output. `conv_before_upsample` feeds TWO branches - the
        // pixel-shuffle octaves that reconstruct the high-resolution image, and a
        // three-channel `conv_aux` that IS the low-resolution image the model also
        // returns - and the bicubic pre-upsample is added to the octaves' before
        // `conv_last`.
        //
        // Note the slopes: `conv_before_upsample` and `conv_after_aux` are
        // `nn.LeakyReLU(inplace=True)` with NO explicit slope, so they are 0.01, the
        // frame's default - not the 0.2 the real-world head's convs use. Reading 0.2
        // off the neighbouring arm is the easy mistake here.
        Upsampler::PixelShuffleAux => {
            let feat = wt.t("conv_before_upsample.0.weight").len() / (c * 9);
            let mut a = vec![0.0f32; feat * hw];
            conv3x3(&scr.cur, c, plan.hp, plan.wp, wt.t("conv_before_upsample.0.weight"), feat,
                    wt.t("conv_before_upsample.0.bias"), &mut a);
            leaky_relu(&mut a, 0.01);
            if let Some(f) = dump.as_deref_mut() {
                f("conv_before_upsample", &a, feat);
            }
            // The aux image comes off the PADDED plane, before any octave - it is
            // 3 channels at `hp x wp`, and the epilogue scales it like the main
            // output. Nothing else in the head reads it.
            let mut aux = vec![0.0f32; 3 * hw];
            conv3x3(&a[..feat * hw], feat, plan.hp, plan.wp, wt.t("conv_aux.weight"), 3,
                    wt.t("conv_aux.bias"), &mut aux);
            if let Some(f) = dump.as_deref_mut() {
                f("conv_aux", &aux, 3);
            }
            let mut x = vec![0.0f32; feat * hw];
            conv3x3(&aux, 3, plan.hp, plan.wp, wt.t("conv_after_aux.0.weight"), feat,
                    wt.t("conv_after_aux.0.bias"), &mut x);
            leaky_relu(&mut x, 0.01);
            // `cur` is dead now: everything the head needs is in `x` and `aux`.
            scr.free_body();
            rss_mark("compressed head: after conv_after_aux");
            let (mut h2, mut w2) = (plan.hp, plan.wp);
            for o in 0..wt.upsampler.octaves(scale) {
                let (uw, ub) = (format!("upsample.{}.weight", 2 * o), format!("upsample.{}.bias", 2 * o));
                let mut shuf = vec![0.0f32; feat * 4 * h2 * w2];
                conv3x3_shuffle2(&x[..feat * h2 * w2], feat, h2, w2, wt.t(&uw), feat, wt.t(&ub),
                                 &mut shuf);
                x = shuf;
                h2 *= 2;
                w2 *= 2;
                rss_mark(&format!("compressed head: after octave {o} ({h2}x{w2})"));
            }
            // `x = self.upsample(x)[:, :, :H*scale, :W*scale] + bicubic[:, :, :H*scale, :W*scale]`
            // - over the whole plane, because the reference's `H, W` are the dims of
            // the image it was HANDED and the task wrapper hands it the PADDED one.
            // So its slice is a no-op here and everything stays on the padded output
            // grid, which is also why `conv_last` runs at that size and the crop to
            // the original resolution happens once, in `finish`, like every other
            // head. Cropping inside the head is the mistake this code was written
            // with first: it made `planes` a different size from every other head's
            // and put the error at the plane's edge.
            let bic = bicubic.expect("PixelShuffleAux needs the bicubic branch (see forward)");
            let mut summed = vec![0.0f32; feat * h2 * w2];
            for ch in 0..feat {
                for y in 0..h2 {
                    for bx in 0..w2 {
                        summed[(ch * h2 + y) * w2 + bx] =
                            x[(ch * h2 + y) * w2 + bx] + bic[(ch * h2 + y) * w2 + bx];
                    }
                }
            }
            planes = vec![0.0f32; 3 * h2 * w2];
            conv3x3(&summed, feat, h2, w2, wt.t("conv_last.weight"), 3, wt.t("conv_last.bias"),
                    &mut planes);
            aux_out = Some(aux);
        }
    }
    // The head's own buffers go back to the pool on drop, but the scratch itself -
    // `cur`, `acc`, the weight pack - is what the caller keeps, so its last use has
    // to be behind us. Naming it here is what ends the borrow.
    let _ = &mut owned.buf;
    rss_mark("head done");
    Ok((owned, planes, aux_out))
}

/// The aux image's epilogue: the same `x / img_range + mean`, and NO CROP.
///
/// The compressed head's second output is emitted on the padded plane and the
/// reference returns it that way - the crop to the original size is applied to
/// the upsampled output only. Cropping this one too would look like tidying and
/// would make the image the wrong size.
///
/// PUBLIC for the same reason `finish` is: the device backend downloads the aux
/// activations and runs this host arithmetic on them, so there is one copy of the
/// epilogue rather than one per backend.
pub fn finish_aux(wt: &Weights, plan: &Plan, aux: &[f32]) -> Vec<f32> {
    let hw = plan.plane();
    let mut out = vec![0.0f32; 3 * hw];
    for ci in 0..3 {
        let src = &aux[ci * hw..(ci + 1) * hw];
        let dst = &mut out[ci * hw..(ci + 1) * hw];
        for (d, v) in dst.iter_mut().zip(src) {
            *d = v / wt.img_range + wt.mean[ci];
        }
    }
    out
}

/// `x / img_range + mean`, then the crop to the original size - which the
/// reference does with a nearest-neighbour-named slice after having upsampled the
/// PADDED plane, so the padded margin is computed and then thrown away.
///
/// The crop is written as "take the first `h*scale` rows and `w*scale` columns of
/// whatever plane is here", never as "the plane is `hp*scale` wide": the two heads
/// hand this function DIFFERENT geometries. The classical head's octaves and
/// `conv_last` run on the padded grid, so the crop is real; the compressed head
/// crops INSIDE the head (the reference's slice is applied before `conv_last`),
/// so its plane is already `h*scale` by `w*scale` and this crop takes everything.
/// Reading the geometry off an assumed padded width is what made this panic.
///
/// `src_w` is the plane's own width, which is the STRIDE its rows are stored
/// with: every head today produces the padded output grid (`plan.wp * scale`),
/// and passing it explicitly is what keeps that a stated fact rather than a
/// coincidence a future head can break silently.
///
/// PUBLIC so the device backend can call it: the epilogue is host arithmetic
/// either way, and a second copy of it would be a second place for the crop to
/// disagree with the padding above it.
pub fn finish(wt: &Weights, plan: &Plan, planes: &[f32], src_w: usize) -> Vec<f32> {
    let scale = wt.scale;
    let (h, w) = (plan.h * scale, plan.w * scale);
    let (oh, ow) = (planes.len() / 3 / src_w, src_w);
    assert!(ow >= w && oh >= h, "finish: a {ow}x{oh} plane cannot contain a {w}x{h} output");
    let mut out = vec![0.0f32; 3 * h * w];
    for ci in 0..3 {
        let plane = &planes[ci * oh * ow..(ci + 1) * oh * ow];
        let dst = &mut out[ci * h * w..(ci + 1) * h * w];
        for y in 0..h {
            for x in 0..w {
                dst[y * w + x] = plane[y * ow + x] / wt.img_range + wt.mean[ci];
            }
        }
    }
    out
}




// ---------------------------------------------------------------------------
// The backend handle, and the library's own checks.
// ---------------------------------------------------------------------------

/// The CPU backend. Buffers are sized on the first forward: a backend is
/// constructed before the image is known, and `--tile auto` can change the
/// geometry between calls, so the scratch is re-sized when it has to be.
pub struct Cpu<'a> {
    wt: &'a Weights,
    mlp_ratio: usize,
    win: usize,
    c: usize,
    scratch: Option<ScratchOwned>,
    plan: Option<Plan>,
    /// The last forward's secondary image, for a head that produces one. Kept on
    /// the handle rather than threaded through `Backend::forward`, whose
    /// single-plane return every existing caller depends on.
    aux: Option<Vec<f32>>,
}

impl<'a> Cpu<'a> {
    pub fn new(wt: &'a Weights) -> Result<Cpu<'a>, String> {
        Ok(Cpu {
            wt,
            mlp_ratio: wt.mlp_ratio,
            win: wt.window,
            c: wt.embed,
            scratch: None,
            plan: None,
            aux: None,
        })
    }
}

impl crate::backend::Backend for Cpu<'_> {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn forward(&mut self, h: usize, w: usize, input: &[f32]) -> Result<Vec<f32>, String> {
        let plan = match self.plan {
            Some(p) if p.h == h && p.w == w => p,
            _ => {
                // A new size: the pooled blocks are reused wherever they are long
                // enough and only the difference is allocated. `--tile auto` calls
                // this repeatedly with different sizes, and doing it this way means
                // the peak is the largest pass rather than the sum of the passes.
                let p = Plan::new(h, w, self.win, self.c);
                let mut fresh = ScratchOwned::new(&p, self.mlp_ratio);
                if let Some(old) = self.scratch.take() {
                    fresh.absorb(old);
                }
                self.scratch = Some(fresh);
                self.plan = Some(p);
                p
            }
        };
        let scr = self.scratch.as_mut().expect("scratch is built with the plan");
        let (planes, aux) = forward(self.wt, &plan, input, scr, None)?;
        self.aux = aux;
        Ok(planes)
    }

    fn aux(&self) -> Option<&[f32]> {
        self.aux.as_deref()
    }
}

/// The library's internal checks, for `--self-test` and `tests/parity.rs`.
///
/// These are the ops whose correctness the fixtures cannot isolate: a gather or a
/// scatter that disagrees with its twin, a pixel shuffle whose two channel orders
/// differ, a conv whose border taps are off by one. A wrong one of those shows up
/// in a fixture as a small numeric difference, indistinguishable from the device's
/// reordering - so they are checked exactly, here, against a naive version.
pub fn selftest() -> Result<(), String> {
    let mut checks = 0;
    let mut rng = 0x12345678u32;
    let mut next = move || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 8) as f32 / 16777216.0) - 0.5
    };

    // conv3x3 against the definition, including the borders.
    {
        let (ci, h, w, co) = (3, 5, 7, 4);
        let inp: Vec<f32> = (0..ci * h * w).map(|_| next()).collect();
        let wgt: Vec<f32> = (0..co * ci * 9).map(|_| next()).collect();
        let bias: Vec<f32> = (0..co).map(|_| next()).collect();
        let mut got = vec![0.0f32; co * h * w];
        conv3x3(&inp, ci, h, w, &wgt, co, &bias, &mut got);
        for o in 0..co {
            for y in 0..h {
                for x in 0..w {
                    let mut acc = bias[o];
                    for i in 0..ci {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let sy = y as isize + ky as isize - 1;
                                let sx = x as isize + kx as isize - 1;
                                if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                                    continue;
                                }
                                acc += wgt[o * ci * 9 + i * 9 + ky * 3 + kx]
                                    * inp[(i * h + sy as usize) * w + sx as usize];
                            }
                        }
                    }
                    let d = (acc - got[o * h * w + y * w + x]).abs();
                    if d > 1e-5 {
                        return Err(format!("conv3x3 differs from its definition at ({x},{y},{o}): {d}"));
                    }
                }
            }
        }
        checks += 1;
    }

    // The head's fused conv against the two ops it replaces: the same rows, written
    // to permuted addresses, so an error here is an interleave error - and an
    // interleave is exactly the kind of mistake that produces a plausible image
    // rather than a crash.
    //
    // THIS USED TO BE BIT FOR BIT, and it cannot be any more. The fused path keeps
    // the engine's row routine, `ky`, `kx`, `ci`; `conv3x3` is the toolkit's now,
    // whose tiled inner loop is `ci`, `ky`, `kx` with the zero taps skipped. Two
    // different orderings of the same 135-term sum differ in the last place of an
    // f32, which is what the 1e-5 below is - the same bound every other op in this
    // function is held to, and three orders inside the fixtures'. What it still
    // catches is every way of getting the permutation wrong, which is what the
    // check is for.
    {
        let (ci, feat, h, w) = (5usize, 3usize, 4usize, 6usize);
        let x: Vec<f32> = (0..ci * h * w).map(|_| next()).collect();
        let wgt: Vec<f32> = (0..4 * feat * ci * 9).map(|_| next()).collect();
        let bias: Vec<f32> = (0..4 * feat).map(|_| next()).collect();
        let mut big = vec![0.0f32; 4 * feat * h * w];
        conv3x3(&x, ci, h, w, &wgt, 4 * feat, &bias, &mut big);
        let mut want = vec![0.0f32; feat * 4 * h * w];
        pixel_shuffle2(&big, 4 * feat, h, w, &mut want);
        let mut got = vec![0.0f32; feat * 4 * h * w];
        conv3x3_shuffle2(&x, ci, h, w, &wgt, feat, &bias, &mut got);
        for i in 0..got.len() {
            let d = (got[i] - want[i]).abs();
            if d > 1e-5 {
                return Err(format!(
                    "conv3x3_shuffle2 differs from conv3x3 + pixel_shuffle2 at {i}: {} vs {} ({d})",
                    got[i], want[i]
                ));
            }
        }
        checks += 1;
    }

    // linear and conv1x1 against the definition - the two matmuls, whose blocked
    // and transposed forms are the one place in this file where the summation
    // order is not the obvious ascending one. The tolerance is f32 rounding on a
    // 60-term sum, three orders inside the fixture's.
    {
        let (rows, ci, co) = (37usize, 60usize, 43usize);
        let x: Vec<f32> = (0..rows * ci).map(|_| next()).collect();
        let wgt: Vec<f32> = (0..co * ci).map(|_| next()).collect();
        let bias: Vec<f32> = (0..co).map(|_| next()).collect();
        let mut xt = vec![0.0f32; rows * ci];
        let mut got = vec![0.0f32; rows * co];
        let mut wp = vec![0.0f32; ci * co];
        linear(&x, rows, ci, &wgt, co, &bias, &mut got, &mut xt, &mut wp);
        for r in 0..rows {
            for o in 0..co {
                let mut acc = bias[o];
                for i in 0..ci {
                    acc += wgt[o * ci + i] * x[r * ci + i];
                }
                let d = (acc - got[r * co + o]).abs();
                if d > 1e-5 {
                    return Err(format!("linear differs from its definition at ({r},{o}): {d}"));
                }
            }
        }
        // conv1x1 is the same loop on an operand that is already channel-major:
        // one op, one check, so a change to the layout one of them assumes cannot
        // pass by breaking both.
        let mut plane = vec![0.0f32; co * rows];
        let mut ctmp = vec![0.0f32; co * rows];
        let mut wp2 = vec![0.0f32; ci * co];
        conv1x1(&xt, ci, rows, &wgt, co, &bias, &mut plane, &mut ctmp, &mut wp2);
        for i in 0..rows * co {
            let (r, o) = (i / co, i % co);
            let mut acc = bias[o];
            for k in 0..ci {
                acc += wgt[o * ci + k] * xt[k * rows + r];
            }
            if (acc - plane[o * rows + r]).abs() > 1e-5 {
                return Err(format!("conv1x1 differs from its definition at ({r},{o})"));
            }
        }
        // BOTH REGISTER TILES, not just the one this CPU selected: the AVX2 worker
        // is what runs here, and the baseline worker is what runs on a machine
        // without AVX2 - so a check that only exercised the dispatched path would
        // leave the other one untested on every machine. The row block is the
        // worker's own MR, so this is the same call shape `matmul_rows` makes.
        for (name, mr, avx2) in [("matmul_avx2", 8usize, true), ("matmul_base", 4, false)] {
            let mut blk = vec![0.0f32; mr * co];
            let nblocks = rows.div_ceil(mr);
            for b in 0..nblocks {
                let take = (rows - b * mr).min(mr);
                let dst = &mut blk[..take * co];
                // SAFETY: the block is exactly MR rows (or the ragged last one),
                // and this CPU has avx2+fma - asserted here rather than assumed,
                // so the check is meaningful on a machine that lacks them.
                unsafe {
                    if avx2 {
                        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
                            return Err("selftest: no avx2+fma, so the AVX2 worker cannot be checked".into());
                        }
                        matmul_avx2(dst, b, &xt, rows, ci, &wp, co, &bias);
                    } else {
                        matmul_base(dst, b, &xt, rows, ci, &wp, co, &bias);
                    }
                }
                for i in 0..take {
                    for o in 0..co {
                        let d = (dst[i * co + o] - got[(b * mr + i) * co + o]).abs();
                        if d > 1e-5 {
                            return Err(format!("{name} differs from linear at ({},{o}): {d}", b * mr + i));
                        }
                    }
                }
            }
        }
        checks += 1;
    }

    // pixel_shuffle2 against the definition of depth-to-space: a channel block
    // becomes a 2x2 neighbourhood at the corresponding position.
    {
        let (c, h, w) = (3, 4, 4);
        let plane: Vec<f32> = (0..4 * c * h * w).map(|_| next()).collect();
        let mut got = vec![0.0f32; c * 4 * h * w];
        pixel_shuffle2(&plane, 4 * c, h, w, &mut got);
        for ch in 0..c {
            for dy in 0..2 {
                for dx in 0..2 {
                    for y in 0..h {
                        for x in 0..w {
                            let want = plane[((ch * 4 + dy * 2 + dx) * h + y) * w + x];
                            let i = (ch * h * 2 + y * 2 + dy) * (w * 2) + x * 2 + dx;
                            if (got[i] - want).abs() > 1e-6 {
                                return Err(format!("pixel_shuffle2 differs at {i}"));
                            }
                        }
                    }
                }
            }
        }
        checks += 1;
    }

    // The parallel channel LayerNorm against the toolkit's sequential twin - the
    // one op in this file that duplicates a toolkit function, so the check is that
    // it is the SAME function, not a similar one.
    {
        let (c, hw) = (17usize, 400usize);
        let x: Vec<f32> = (0..c * hw).map(|_| next()).collect();
        let w: Vec<f32> = (0..c).map(|_| next()).collect();
        let b: Vec<f32> = (0..c).map(|_| next()).collect();
        let mut want = vec![0.0f32; c * hw];
        lg::channel_layer_norm(&x, &w, &b, &mut want, c, hw, LN_EPS);
        let mut got = vec![0.0f32; c * hw];
        channel_layer_norm(&x, &w, &b, &mut got, c, hw, LN_EPS);
        if got != want {
            let at = (0..got.len()).find(|&i| got[i] != want[i]).unwrap();
            return Err(format!(
                "channel_layer_norm is not the toolkit's: {} vs {} at {at}",
                want[at], got[at]
            ));
        }
        checks += 1;
    }

    // gather/scatter are inverses, at both shifts.
    {
        let plan = Plan::new(19, 13, 8, 2);
        let x: Vec<f32> = (0..plan.plane() * plan.c).map(|_| next()).collect();
        for shift in [0usize, 4] {
            let mut tok = vec![0.0f32; plan.tokens() * plan.c];
            gather(&plan, &x, shift, None, &mut tok);
            let mut back = vec![0.0f32; x.len()];
            scatter(&plan, &tok, shift, &mut back);
            if back != x {
                return Err(format!("gather/scatter are not inverses at shift {shift}"));
            }
        }
        checks += 1;
    }

    // The attention kernel against a direct transcription of the reference's
    // numpy: cosine attention, the bias table, and the mask.
    {
        let plan = Plan::new(9, 17, 4, 4);
        let heads = 2;
        let hd = 2;
        let n = plan.n;
        let ls: Vec<f32> = vec![1.5, 0.25];
        let cpb: Vec<f32> = (0..n * n * heads).map(|_| next() * 0.01).collect();
        let qkv: Vec<f32> = (0..3 * plan.tokens() * plan.c).map(|_| next()).collect();
        for shift in [0usize, plan.win / 2] {
            let mut out = vec![0.0f32; plan.tokens() * plan.c];
            attention(&plan, heads, hd, &ls, &cpb, &qkv, shift, &mut out);
            // The direct version: for each (window, head, query), normalise, dot,
            // scale, bias, mask, softmax, and apply.
            let c = plan.c;
            let tok_c = plan.tokens() * c;
            for wi in 0..plan.nw {
                for h in 0..heads {
                    // The reference's own scale: the clamped temperature, and
                    // nothing else (no 1/sqrt(head_dim) in cosine attention).
                    let l = ls[h].min(100f32.ln()).exp();
                    for q in 0..n {
                        let mut logits = vec![0.0f32; n];
                        for k in 0..n {
                            let mut dot = 0.0;
                            let mut qn = 0.0;
                            let mut kn = 0.0;
                            for d in 0..hd {
                                let qv = qkv[plan.token(wi, q) * c + h * hd + d];
                                let kv = qkv[tok_c + plan.token(wi, k) * c + h * hd + d];
                                qn += qv * qv;
                                kn += kv * kv;
                            }
                            qn = qn.sqrt().max(1e-12);
                            kn = kn.sqrt().max(1e-12);
                            for d in 0..hd {
                                dot += qkv[plan.token(wi, q) * c + h * hd + d] / qn
                                    * (qkv[tok_c + plan.token(wi, k) * c + h * hd + d] / kn);
                            }
                            let mut s = dot * l + cpb[(q * n + k) * heads + h];
                            if shift != 0 {
                                let (rq, cq) = plan.region(wi, q, shift);
                                let (rk, ck) = plan.region(wi, k, shift);
                                if rq != rk || cq != ck {
                                    s -= 100.0;
                                }
                            }
                            logits[k] = s;
                        }
                        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let ex: Vec<f32> = logits.iter().map(|v| (v - mx).exp()).collect();
                        let sum: f32 = ex.iter().sum();
                        for d in 0..hd {
                            let mut acc = 0.0;
                            for k in 0..n {
                                acc += ex[k] / sum * qkv[2 * tok_c + plan.token(wi, k) * c + h * hd + d];
                            }
                            let got = out[plan.token(wi, q) * c + h * hd + d];
                            if (acc - got).abs() > 2e-5 {
                                return Err(format!(
                                    "attention differs from its definition at window {wi} head {h} \
                                     query {q} dim {d} shift {shift}: {} vs {}",
                                    acc, got
                                ));
                            }
                        }
                    }
                }
            }
        }
        checks += 1;
    }

    // The upsampling heads' resampling ops, against their definitions.
    {
        let (c, h, w, scale) = (2usize, 3usize, 3usize, 2usize);
        let x: Vec<f32> = (0..c * scale * scale * h * w).map(|_| next()).collect();
        let mut got = vec![0.0f32; c * (h * scale) * (w * scale)];
        pixel_shuffle(&x, c, h, w, scale, &mut got);
        for ch in 0..c {
            for dy in 0..2 {
                for dx in 0..2 {
                    for y in 0..h {
                        for xx in 0..w {
                            let want = x[((ch * 4 + dy * 2 + dx) * h + y) * w + xx];
                            let i = (ch * h * 2 + y * 2 + dy) * (w * 2) + xx * 2 + dx;
                            if (got[i] - want).abs() > 1e-6 {
                                return Err(format!("pixel_shuffle differs at {i}"));
                            }
                        }
                    }
                }
            }
        }
        let y0: Vec<f32> = (0..c * h * w).map(|_| next()).collect();
        let mut up = vec![0.0f32; c * 4 * h * w];
        upsample2x_nearest(&y0, c, h, w, &mut up);
        for ch in 0..c {
            for y in 0..h {
                for xx in 0..w {
                    let v = y0[(ch * h + y) * w + xx];
                    let o = (ch * h * 2 + y * 2) * (w * 2) + xx * 2;
                    if up[o] != v || up[o + 1] != v || up[o + w * 2] != v || up[o + w * 2 + 1] != v {
                        return Err(format!("upsample2x_nearest differs at ({xx},{y})"));
                    }
                }
            }
        }
        checks += 1;
    }

    println!("cpu selftest: {checks} op families checked against their definitions");
    Ok(())
}