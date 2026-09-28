//! A backend, and the tiling the two of them share.
//!
//! The trait exists so that the CLI, the tile loop and the fixture checker are
//! written once: `Cpu` and `Gpu` differ only in where the arithmetic happens, and
//! a second copy of the loop above them would be a second place for the padding
//! convention and the crop to disagree.
//!
//! TILING IS NOT EXACT, AND HOW INEXACT IT IS HAS TO BE MEASURED. The network's
//! receptive field is local: window attention never mixes tokens from different
//! windows, and the only operators that cross a window boundary are the 3x3 convs
//! (conv_first, each stage's `conv`, conv_after_body) and the reconstruction convs.
//! A tile therefore differs from the whole image only within a few pixels of its
//! edge - and only because the reference's padding rule (reflect, to the next
//! window multiple) is applied to the TILE. Two consequences the engine has to
//! live with rather than hide:
//!
//! * `--tile` is for images that do not fit in memory, and the tile is derived
//!   from the budget, not chosen to be pretty. `tests/tiling.rs` measures the
//!   whole-image difference for a given tile size and margin so the number in the
//!   README is a measurement, not a claim.
//! * The tile and the margin are rounded to window multiples, so the geometry
//!   inside a tile is the geometry the whole image would have had at that
//!   offset - the windows line up; only the padded margin differs.

use crate::image::Image;
use crate::plan::Plan;
use crate::weights::{Upsampler, Weights};

pub trait Backend {
    fn name(&self) -> &'static str;

    /// One whole-image forward pass. `input` is [3][h][w] in [0,1] (NOT
    /// mean-adjusted); the result is [3][h*scale][w*scale] in [0,1] after the
    /// denormalisation.
    fn forward(&mut self, h: usize, w: usize, input: &[f32]) -> Result<Vec<f32>, String>;

    /// The secondary image the last `forward` produced, if its head makes one.
    ///
    /// Only the compressed_sr head does: it returns the low-resolution
    /// reconstruction beside the upsampled one, on the PADDED plane's geometry.
    /// This is a separate accessor rather than a second element of `forward`'s
    /// return so that every existing caller - and the tiled path, which calls
    /// `forward` once per tile - keeps the signature it was written against.
    /// `None` for every other head, which is what tells the caller not to write
    /// a second file.
    fn aux(&self) -> Option<&[f32]> {
        None
    }
}

/// The reference's preprocessing: pad to the next window multiple, subtract the
/// mean, scale by img_range - and the reverse at the end.
pub struct Pre {
    pub plan: Plan,
    pub h: usize,
    pub w: usize,
}

impl Pre {
    pub fn new(wt: &Weights, h: usize, w: usize) -> Result<Pre, String> {
        if h < wt.window || w < wt.window {
            return Err(format!(
                "{}x{} is smaller than one {}x{} window. The reference pads to the NEXT window \
                 multiple by concatenating the image with its own mirror, which for an image \
                 shorter than a window truncates the pad and then falls through to a second, \
                 different padding rule this engine does not implement.",
                h, w, wt.window, wt.window
            ));
        }
        Ok(Pre { plan: Plan::new(h, w, wt.window, wt.embed), h, w })
    }

    /// Mean-adjusted input, in the network's units, for the padded plane the
    /// backend expects. This is the ONLY place the mean is subtracted, matching
    /// the reference's single `x = (x - self.mean) * self.img_range`.
    pub fn adjust(&self, wt: &Weights, img: &[f32]) -> Vec<f32> {
        let hw = self.h * self.w;
        let mut out = img.to_vec();
        for c in 0..3 {
            let m = wt.mean[c];
            for v in out[c * hw..(c + 1) * hw].iter_mut() {
                *v = (*v - m) * wt.img_range;
            }
        }
        out
    }
}

/// Run `backend` over an image in tiles of `tile` pixels, taking `pad` pixels of
/// context around each and discarding them. `tile == 0` is one pass over the whole
/// image, which is the only exact mode.
pub fn run_tiled(
    wt: &Weights,
    backend: &mut dyn Backend,
    img: &Image,
    tile: usize,
    pad: usize,
) -> Result<Image, String> {
    let scale = wt.scale;
    if tile == 0 {
        let pre = Pre::new(wt, img.h, img.w)?;
        let adjusted = pre.adjust(wt, &img.data);
        let out = backend.forward(img.h, img.w, &adjusted)?;
        return Ok(Image { w: img.w * scale, h: img.h * scale, data: out });
    }

    let win = wt.window;
    let tile = (tile / win).max(1) * win;
    // ROUNDED UP, and never below one window. Up rather than down because the
    // pad is context, and a request for 20 pixels of it must not silently become
    // 16 - the whole point of the pad is that more of it is more accurate. Zero
    // and anything below a window mean one window, not none: a tile with no
    // context is the one setting that is never what the user wanted.
    let pad = pad.div_ceil(win).max(1) * win;
    let mut out = Image::new(img.w * scale, img.h * scale);
    let mut ty = 0;
    while ty < img.h {
        let mut tx = 0;
        while tx < img.w {
            // The tile plus its context, clamped to the image. The context is
            // what the tile's own reflect padding will stand in for, so it is
            // taken from the image wherever the image has it.
            let y0 = ty.saturating_sub(pad);
            let x0 = tx.saturating_sub(pad);
            let y1 = (ty + tile + pad).min(img.h);
            let x1 = (tx + tile + pad).min(img.w);
            let (th, tw) = (y1 - y0, x1 - x0);
            let mut sub = vec![0.0f32; 3 * th * tw];
            for c in 0..3 {
                for y in 0..th {
                    let src = (c * img.h + y0 + y) * img.w + x0;
                    sub[(c * th + y) * tw..(c * th + y) * tw + tw]
                        .copy_from_slice(&img.data[src..src + tw]);
                }
            }
            let pre = Pre::new(wt, th, tw)?;
            let adjusted = pre.adjust(wt, &sub);
            let res = backend.forward(th, tw, &adjusted)?;
            // Copy the tile's own region, dropping the context on each side.
            let (oy0, ox0) = (ty - y0, tx - x0);
            let cw = (tx + tile).min(img.w) - tx;
            let ch = (ty + tile).min(img.h) - ty;
            let (rh, rw) = (th * scale, tw * scale);
            // THE COPY IS ON THE OUTPUT GRID, so it runs over `ch * scale` rows -
            // one per OUTPUT row of the tile, not one per input row. Iterating
            // `0..ch` and multiplying the row by `scale` writes only every
            // `scale`-th output row and leaves the rest of the buffer at zero,
            // which is a black-and-striped image rather than a slightly wrong one.
            // `dst` starts at the tile's first output row and `src` at the tile's
            // first row inside the padded result, both counted in output pixels.
            for c in 0..3 {
                for y in 0..ch * scale {
                    let dst = (c * out.h + ty * scale + y) * out.w + tx * scale;
                    let src = (c * rh + oy0 * scale + y) * rw + ox0 * scale;
                    out.data[dst..dst + cw * scale]
                        .copy_from_slice(&res[src..src + cw * scale]);
                }
            }
            tx += tile;
        }
        ty += tile;
    }
    Ok(out)
}

/// The tile size that keeps a forward pass inside `budget` bytes, rounded down to
/// a window multiple - the inverse of the memory check, so `--tile auto` cannot
/// disagree with the number the engine would have complained about.
pub fn auto_tile(wt: &Weights, budget: u64, h: usize, w: usize, pad: usize, gpu: bool) -> usize {
    let win = wt.window;
    let per_px = per_pixel_floats(wt, gpu);
    let pad = pad.div_ceil(win).max(1) * win;
    // THE IMAGE'S SIZE HAS TO BE IN HERE, not just the tile's. An earlier version
    // only asked whether a 1024-pixel tile fit the budget - which it always does,
    // at 18 MB - and so returned 1024 for a 640x640 image, a tile larger than the
    // image itself: an effectively untiled pass needing 7 GB under a 1 GB budget.
    let whole = crop_side(h, w, 0, pad, win) as u64;
    if per_px * whole * whole * 4 <= budget && launches(whole, gpu) {
        return 0;
    }
    let mut t = (1024.max(win) / win) * win;
    loop {
        let s = crop_side(h, w, t, pad, win) as u64;
        if (per_px * s * s * 4 <= budget && launches(s, gpu)) || t == win {
            return t;
        }
        t = (t / 2 / win).max(1) * win;
    }
}

/// Can a forward over a crop `crop_side` reports as this many pixels a side be
/// LAUNCHED?
///
/// A budget question is not a launch question and both have to be asked: on a
/// 60-wide checkpoint the lightweight x2 fits 1000 pixels in 2.1 GB and cannot
/// launch 1016, because the token grid runs out of `grid.y` blocks. `crop_side`
/// reports the longer side of a square, which is what every other calculation
/// here treats the crop as, so the square's area is the area the launch sees.
fn launches(side: u64, gpu: bool) -> bool {
    !gpu || side * side <= GPU_MAX_PLANE_PIXELS
}

/// The activation bytes a run of `tile` over an `h`x`w` image will allocate, the
/// figure the banner prints: the whole padded plane for one pass, or the largest
/// tile plus its context. `tile == 0` is one pass.
pub fn activation_bytes(wt: &Weights, h: usize, w: usize, tile: usize, pad: usize, gpu: bool) -> u64 {
    let per_px = per_pixel_floats(wt, gpu);
    let side = crop_side(h, w, tile, pad, wt.window) as u64;
    per_px * side * side * 4
}

/// The largest padded plane, in PIXELS, that a device forward can be LAUNCHED on.
///
/// NOT A MEMORY LIMIT, and the reason a small checkpoint on a large card still
/// cannot take an arbitrarily large image in one pass. The token matmuls are
/// launched as `(ceil(co/16), ceil(tokens/16), 1)` and a grid dimension is capped
/// at 65535 blocks, so a forward can address at most `65535 * 16` tokens. One
/// padded PIXEL is one token - a window is `win * win` tokens and the plane is
/// `nw * nw` windows of them - so that is also the largest plane in pixels.
///
/// Past it the launch fails with `CUDA_ERROR_INVALID_VALUE`, which is not an
/// out-of-memory error and therefore NOT something `run_with_fallback` retries:
/// this has to be prevented rather than survived, which is what `auto_tile`'s
/// `launches` test and the CLI's own check are for. Measured on the lightweight
/// x2, whose `819` floats a padded pixel ask for very little memory indeed: 1000
/// pixels (a 1008 plane, 1016064 tokens, `grid=(4,63504,1)`) runs, and 1016 (a
/// 1024 plane, 1048576 tokens, `grid=(4,65536,1)`) fails - one window of side,
/// and 16 tokens, apart. That pair is what pins the rule to the token count
/// rather than to anything about pixels.
pub const GPU_MAX_PLANE_PIXELS: u64 = 65535 * 16;

/// The padded plane a crop of `tile` pixels plus `pad` of context runs on, in
/// pixels a side - the ONE place this geometry is computed, so `--tile auto` and
/// the banner cannot disagree about it.
///
/// Two roundings happen to the crop before the network sees it and both are the
/// reference's, not a convention of this function: the tile and the context are
/// rounded to window multiples (in `run_tiled`), and `Pre` then pads the crop to
/// the NEXT window multiple plus a whole window - `(n / win + 1) * win`, never
/// `ceil(n / win) * win`, which is the padding the reference actually does. The
/// crop is clamped to the image, so `tile + 2 * pad` above the image's own size is
/// not a larger plane.
///
/// `tile == 0` is the whole image and takes the same rule, which is why the
/// whole-image test in `auto_tile` is `crop_side(h, w, 0, ..)`: an earlier version
/// used `h.max(w) + 2 * pad` there - the crop rule with no crop - and so called a
/// 648x648 image 712 wide, 18% of overestimate that pushed it into tiles under a
/// budget the exact pass fits in.
pub fn crop_side(h: usize, w: usize, tile: usize, pad: usize, win: usize) -> usize {
    let (th, tw) = if tile == 0 {
        (h, w)
    } else {
        let t = (tile / win).max(1) * win;
        let pad = pad.div_ceil(win).max(1) * win;
        ((t + 2 * pad).min(h), (t + 2 * pad).min(w))
    };
    let plane = |n: usize| (n / win + 1) * win;
    plane(th).max(plane(tw))
}

/// Floats of activation per PADDED pixel, for the banner and for `--tile auto`.
///
/// THIS IS A BUDGET, SO IT MUST NOT BE OPTIMISTIC, and it is written out term by
/// term from the two backends' buffer inventories rather than gathered from the
/// structs - because the way it has gone wrong is always the same way: a buffer
/// whose SIZE or LIFETIME is not what the struct list suggests.
///
/// THREE THINGS ARE EASY TO GET WRONG, and all three were wrong here at one time.
///
/// * THE HEAD'S BUFFERS ARE AT THE OCTAVE'S RESOLUTION, not the padded plane's.
///   Octave `o` writes `4 * feat` channels at `2^o` times the padded resolution,
///   which is `4 * feat * 4^o` floats per PADDED pixel, and the previous octave's
///   output is still live while it does - so the pixel-shuffle head peaks at
///   `feat * scale^2` (the last shuffle) plus `feat * scale^2 / 4` (the one
///   before), i.e. 1280 for the classical x4 head. Counting `feat + 4 * feat`
///   instead, as if the octave were the padded plane, gives 368 and understates
///   the head by 3.5x - which is what made `--tile auto` choose a whole-image pass
///   at 512 that allocated 4094 MiB while the banner claimed 1166.
/// * THE BODY NEEDS THE TOKEN BUFFERS; THE HEAD DOES NOT. `Scratch::free_body`
///   runs between them, so the body's peak is the one that includes `tok`,
///   `tok2`, `attn`, `qkv`, `mlp` and `xt`, and the head's is the one that does
///   not. The body is the larger of the two by a wide margin (2880 floats a
///   padded pixel against 900 + the head, for a 180-wide model).
/// * THE DEVICE FREES NOTHING. `Gpu`'s `Acts` is one allocation for the whole
///   forward, and there is no device counterpart of `free_body`: the head's
///   buffers are additive with every body buffer, so the CPU's two regimes
///   collapse into one and the device's figure is the larger of the two by about a
///   fifth for a 180-wide model (3671 floats a padded pixel against 3111). So the
///   CALLER SAYS WHICH BACKEND IT BUILT - `gpu` - rather than this guessing a
///   maximum, because the maximum on the host would tile a 1000x1000 image that
///   measurably fits in one exact pass.
///
/// The checkpoint, the weight pack and the allocator's own granularity are NOT
/// here: the first two are constants (`gpu::uploaded_bytes` for the device) and
/// the third is a fact about the allocator, not about the geometry. Against
/// `nvidia-smi` on the four published checkpoints at 512x512 the sum of this
/// figure, the upload and the driver's own baseline lands within a few percent of
/// the measured peak, on the low side - which is the side a budget must err on,
/// and the reason `run_with_fallback` exists at all.
pub fn per_pixel_floats(wt: &Weights, gpu: bool) -> u64 {
    let c = wt.embed as u64;
    let mlp = c * wt.mlp_ratio as u64;
    let scale = wt.scale as u64;
    let r2 = scale * scale;
    // `Scratch`, at the moment the body is widest: cur, res, stage, acc and ctmp;
    // the body's own skip (`body_res`); tok, tok2 and attn; qkv; mlp and xt.
    //
    // PLUS ONE MORE PLANE: `nbuf`, the body's staging block for the final
    // `channel_layer_norm` before `conv_after_body`. It comes from the pool rather
    // than from a field of `Scratch`, which is why the inventory above left it out
    // - and leaving it out makes the host figure 6.6% optimistic at every size,
    // which is the one direction a budget must not be wrong in.
    //
    // WHAT IDENTIFIED IT was a peak-RSS fit failing and then an in-process phase
    // trace (`SWIN2SR_DEBUG_RSS`, which prints `VmRSS` at the marks in
    // `cpu::forward` and `cpu::head`). At 512x512 on the classical x4 the curve is
    // 2795 MiB when `forward` opens - the whole scratch is already built - then
    // +189 MiB at `conv_first` (`cur`, its clone `body_res` and the padded copy),
    // +8 a stage for the six stages, and then ONE STEP OF EXACTLY ONE c=180 PLANE,
    // +190 MiB, at `nbuf`; the head's `a` puts a `feat`=64 plane (+66 MiB) on top
    // of that and the peak is there. `free_body` then drops RSS by 2.1 GB, and
    // that is the point: everything in the inventory is live when `nbuf` is
    // allocated, so it is additive and not a freed block counted twice. Two
    // explanations were refuted on the way here and are worth not re-deriving: it
    // is not `pad_reflect`'s row staging (3 floats a pixel, far too small), and it
    // is not glibc holding freed transient blocks - forcing `mmap` above 4 KB
    // moves RSS by 4 MiB at 512 pixels, and the phase trace shows the step is a
    // live allocation rather than a plateau.
    let cpu_body = 6 * c + 3 * c + 3 * c + 2 * mlp + c;
    // `Gpu`'s `Acts`: cur, res, stage, acc and body; tok, tok2 and attn; qkv; mlp.
    // It is never freed, so it is live during the head as well.
    let gpu_acts = 5 * c + 3 * c + 3 * c + mlp;
    // `conv_before_upsample`'s width, read from the checkpoint: the two heads that
    // have that layer take `feat` from it, and the direct head has neither.
    let feat = |c: u64| wt.t("conv_before_upsample.0.weight").len() as u64 / (c * 9);

    let head = match wt.upsampler {
        Upsampler::PixelShuffle | Upsampler::PixelShuffleAux => {
            let feat = feat(c);
            // Octave `o`'s shuffled output is `4 * feat` channels at `2^o` times
            // the padded resolution; the one before it is still live. The final
            // conv then reads the last octave and writes `3 * scale^2`.
            let mut prev = feat;
            let mut peak = 0;
            for o in 0..wt.upsampler.octaves(wt.scale) as u32 {
                let shuf = 4 * feat * 4u64.pow(o);
                peak = peak.max(prev + shuf);
                prev = shuf;
            }
            peak.max(prev + 3 * r2)
        }
        Upsampler::NearestConv => {
            let feat = feat(c);
            // Two nearest-upsample + conv steps. At step `s` the upsampled buffer
            // (`4 * feat` channels at `2^s`) and the conv's output (`feat` at
            // `2^(s+1)`) are both live, so the last step is `16*feat + 16*feat` -
            // the largest allocation in this head by a wide margin.
            let mut peak = 0;
            for s in 0..2u32 {
                let up = 4 * feat * 4u64.pow(s);
                let next = feat * 4u64.pow(s + 1);
                peak = peak.max(up + next);
            }
            // `conv_hr`'s output and the final 3-plane image.
            peak.max(feat * 4 + 3 * r2)
        }
        // One conv to `3 * scale^2` planes and the shuffle of it into the output.
        Upsampler::PixelShuffleDirect => 2 * 3 * r2,
    };

    // The padded input copy and the output image, live in both regimes.
    let io = 3 + 3 * r2;
    // ASKED FOR THE BACKEND THAT IS GOING TO RUN. The two figures are not
    // interchangeable: the device's is the larger (3671 floats a padded pixel
    // against 3111 for the 180-wide model), and using it on the host would tile a
    // 1000x1000 image that measurably fits in one exact pass. The caller knows
    // which backend it built; this is that fact, and nothing here guesses.
    if gpu {
        gpu_acts + head + io
    } else {
        cpu_body.max(5 * c + head) + io
    }
}
