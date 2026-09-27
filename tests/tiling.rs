//! Tiling: what it is for, and what it must not do.
//!
//! `--tile` exists so an image whose activations do not fit on the device can
//! still be processed. The tile loop is the one piece of this engine that is
//! arithmetic-free - it crops, runs the network, and pastes - so it is exactly the
//! piece where a mistake produces a whole-image error rather than a numerical one,
//! and exactly the piece the golden fixtures cannot see (they run one pass).
//!
//! TWO PROPERTIES ARE CHECKED HERE, and they are different in kind:
//!
//! * **Every output pixel is written, and written from the right place.** A
//!   constant input is the sharpest probe: the whole-image result is constant to
//!   the last bit, so any pixel left at zero, double-written, or taken from the
//!   wrong row shows up as a difference no tolerance can hide. This is not
//!   hypothetical - an earlier version of the loop iterated the tile's INPUT rows
//!   while indexing both ends on the OUTPUT grid, writing one row in `scale` and
//!   leaving the rest of the buffer zero. The fixtures all passed; the image was
//!   black-striped.
//! * **The error from cutting the receptive field falls as the context grows.**
//!   This one is a measurement, and the numbers are in the README. The pad has to
//!   be proportional to the tile, because the shift of `win/2` on the shifted
//!   blocks is a window-wide hole no absolute pad can cover at any tile size.
//!
//! The checkpoint used is the LIGHTWEIGHT one: a 60-wide, 4-stage model that runs
//! a 32x32 image in about a second. The classical x4 would take ~16 s per pass,
//! and these tests take several passes each.

mod common;

use common::weights;
use swin2sr::backend::{
    activation_bytes, auto_tile, crop_side, run_tiled, GPU_MAX_PLANE_PIXELS,
};
use swin2sr::cpu::Cpu;
use swin2sr::image::Image;
use swin2sr::weights::Weights;

/// A plane filled by `f(c, y, x)`.
fn image(h: usize, w: usize, f: impl Fn(usize, usize, usize) -> f32) -> Image {
    let mut data = vec![0.0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                data[(c * h + y) * w + x] = f(c, y, x);
            }
        }
    }
    Image { w, h, data }
}

/// One run of the tile loop over `img`. `tile == 0` is the whole image.
fn run(wt: &Weights, img: &Image, tile: usize, pad: usize) -> Image {
    let mut backend = Cpu::new(wt).expect("construct the CPU backend");
    run_tiled(wt, &mut backend, img, tile, pad).expect("run the tile loop")
}

/// The largest absolute difference between two results, in [0,1] units.
fn worst(a: &Image, b: &Image) -> f32 {
    assert_eq!(a.w, b.w);
    assert_eq!(a.h, b.h);
    a.data.iter().zip(&b.data).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max)
}

/// A constant image, i.e. one the network's whole-image output is constant for.
fn flat(h: usize, w: usize, v: f32) -> Image {
    image(h, w, |_, _, _| v)
}

/// Every pixel a tiled run writes, and where it came from.
///
/// A constant input is the sharpest probe available without a second reference
/// implementation: the network's answer for it is a nearly-constant field (it is
/// position dependent - the cosine attention's bias is - so "nearly" is as far as
/// this goes), which means any pixel left at ZERO, or taken from the wrong place,
/// stands out as a difference no tolerance can hide. An earlier version of the
/// loop iterated the tile's INPUT rows while indexing both ends on the OUTPUT
/// grid: it wrote one row in `scale`, left the rest of the buffer at zero, and
/// passed every fixture in `parity.rs`. Three quarters of the output was black.
#[test]
fn tiled_output_covers_the_image_exactly() {
    let Some(wt) = weights("lightweight-x2") else { return };
    let img = flat(32, 32, 0.55);
    let (whole, lo, hi) = one_pass(&wt, &img);
    assert!(lo > 0.1, "the flat probe is only useful if the true answer is far from zero");

    for (tile, pad) in [(8, 0), (8, 8), (16, 0), (16, 16), (24, 8), (32, 8)] {
        let got = run(&wt, &img, tile, pad);
        let zeros = got.data.iter().filter(|v| **v == 0.0).count();
        assert_eq!(
            zeros, 0,
            "tile {tile} pad {pad}: {zeros} output pixels are exactly zero where the whole-image \
             answer is {lo}..{hi}; those were never written, which is what a tile loop that \
             indexes its crop on the wrong grid leaves behind"
        );
        let d = worst(&got, &whole);
        assert!(
            d <= 1e-2,
            "tile {tile} pad {pad}: the flat probe differs from one pass by {d:.3e}; with no \
             structure to lose, the only source of an error this size is the mosaic"
        );
    }
}

/// The result of one whole-image pass, plus its minimum and maximum.
fn one_pass(wt: &Weights, img: &Image) -> (Image, f32, f32) {
    let out = run(wt, img, 0, 0);
    let (lo, hi) = out.data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &p| {
        (lo.min(p), hi.max(p))
    });
    (out, lo, hi)
}

/// `--tile-pad` is rounded UP to a whole window, and never down to zero.
///
/// The rule is `(pad / win).max(1) * win` in `run_tiled`: a pad smaller than one
/// window is not a pad at all - the network's own reflect padding reaches beyond
/// it - so asking for 0 context gets one window, and asking for 9 gets 16. This
/// test pins the rule down by observing which requests give identical results,
/// which matters because the CLI reports the pad the user typed, not the one used.
#[test]
fn the_pad_rounds_up_to_a_whole_window() {
    let Some(wt) = weights("lightweight-x2") else { return };
    let win = wt.window;
    let img = image(16, 16, |c, y, x| {
        let t = (c * 256 + y * 16 + x) as f32 * 0.618_034;
        t - t.floor()
    });
    let one = run(&wt, &img, 8, win);
    // 0, 1 and `win - 1` all mean "one window".
    for asked in [0, 1, win - 1, win] {
        let got = run(&wt, &img, 8, asked);
        assert_eq!(
            worst(&got, &one),
            0.0,
            "--tile-pad {asked} must behave as one window ({win}); the loop rounds the pad up \
             to a window multiple and clamps it to at least one"
        );
    }
    // `win + 1` means TWO windows, not one: the loop rounds the pad UP to a
    // window multiple rather than down, so a request for more context than a
    // window gets more than a window. The image has to be big enough for the two
    // answers to differ at all - on an image no larger than one tile, every pad
    // at or above a window is clamped to the whole image and the assertion would
    // hold whatever the rounding was.
    let big = image(4 * win, 4 * win, |c, y, x| {
        let t = (c * 1024 + y * 32 + x) as f32 * 0.618_034;
        t - t.floor()
    });
    let two = run(&wt, &big, win, 2 * win);
    assert_eq!(
        worst(&run(&wt, &big, win, win + 1), &two),
        0.0,
        "--tile-pad {} must round UP to two windows ({})",
        win + 1,
        2 * win
    );
    assert!(
        worst(&run(&wt, &big, win, win), &two) > 0.0,
        "one window of context must not already be two on a {}x{} image - otherwise the \
         rounding is untested",
        big.h,
        big.w
    );
}

/// A tile that covers the whole image must be the whole-image pass, bit for bit.
///
/// This is the boundary of the loop: with one tile and no context to take, the
/// crop is the image and the paste is the result. If a tile size at or above the
/// image is not identical to one pass, then the loop's geometry is wrong in a way
/// that has nothing to do with the receptive field - and `--tile <large>` would
/// silently degrade a small image.
#[test]
fn a_tile_covering_the_image_is_one_pass() {
    let Some(wt) = weights("lightweight-x2") else { return };
    let img = image(24, 24, |c, y, x| ((c * 7 + y * 5 + x * 3) % 251) as f32 / 255.0);
    let whole = run(&wt, &img, 0, 0);
    for (tile, pad) in [(24, 0), (24, 8), (32, 8), (48, 16)] {
        let got = run(&wt, &img, tile, pad);
        assert_eq!(
            worst(&got, &whole),
            0.0,
            "tile {tile} pad {pad} on a 24x24 image: the single tile IS the image, so this must \
             be the same run as one pass"
        );
    }
}

/// `--tile auto` has two criteria, and the launch ceiling is not the budget.
///
/// ON A CARD BIG ENOUGH FOR ANYTHING, A TILE IS STILL FORCED. The token matmuls
/// launch as `(ceil(c_out / 16), ceil(tokens / 16), 1)` and `grid.y` is capped at
/// 65535 blocks, so a forward can address `65535 * 16` tokens - one padded pixel
/// each - and no amount of free memory lifts that. `auto_tile` has to count it
/// (`backend::launches`), because the failure it prevents is
/// `CUDA_ERROR_INVALID_VALUE`, which `run_with_fallback` deliberately does not
/// retry: a budget-only auto would choose a whole pass that cannot be launched
/// and the run would die with a message naming a kernel rather than a limit.
///
/// The budget here is 64 GiB, so memory can never be the reason for the answer:
/// anything but the whole image is the launch rule speaking.
#[test]
fn auto_tile_counts_the_launch_ceiling_not_only_the_budget() {
    let Some(wt) = weights("lightweight-x2") else { return };
    let budget = 64u64 << 30;
    let big = 1200;
    let plane = |tile: usize| crop_side(big, big, tile, 32, wt.window) as u64;

    // The whole image is inside the budget and outside the grid.
    assert!(
        activation_bytes(&wt, big, big, 0, 32, true) <= budget,
        "this test is about the launch ceiling; the budget must not bind first"
    );
    assert!(
        plane(0) * plane(0) > GPU_MAX_PLANE_PIXELS,
        "a {big}x{big} image pads to a {}x{} plane, which has to be past the {}-token ceiling \
         for this test to say anything",
        plane(0),
        plane(0),
        GPU_MAX_PLANE_PIXELS
    );

    let t = auto_tile(&wt, budget, big, big, 32, true);
    assert_ne!(t, 0, "one pass does not fit the grid, so `auto` must not choose it");
    assert!(
        plane(t) * plane(t) <= GPU_MAX_PLANE_PIXELS,
        "`auto` chose a {t} tile, whose {}x{} plane is still past the ceiling it was supposed to \
         respect",
        plane(t),
        plane(t)
    );

    // THE HOST HAS NO GRID TO RUN OUT OF. The same call with `gpu = false` must
    // take the whole image, or the ceiling is being applied to runs it cannot
    // affect - and a host run of a 1200-pixel image is a few gigabytes, not a
    // launch.
    assert_eq!(
        auto_tile(&wt, budget, big, big, 32, false),
        0,
        "the launch ceiling is a device rule; a host run of this size fits its budget"
    );
}

/// Cutting the receptive field costs accuracy, and only context buys it back.
///
/// The measured shape of that tradeoff, on a 32x32 image and with `win = 8`:
/// `--tile 16` with one window of context leaves ~0.08 of 1.0 on the tile seams,
/// two windows brings it to 0, and no amount of context at `--tile 8` does as
/// well as a larger tile with the same context - the pad has to scale with the
/// tile. The README's table is this measurement at a size where the numbers are
/// not swamped by one tile covering the image.
#[test]
fn context_reduces_the_tiling_error() {
    let Some(wt) = weights("lightweight-x2") else { return };
    let win = wt.window;
    let img = image(32, 32, |c, y, x| {
        let t = (c * 3 + y * 11 + x * 29) as f32 * 0.618_034;
        t - t.floor()
    });
    let whole = run(&wt, &img, 0, 0);

    let thin = worst(&run(&wt, &img, 2 * win, win), &whole);
    let thick = worst(&run(&wt, &img, 2 * win, 2 * win), &whole);
    assert!(
        thin > thick,
        "one window of context ({thin:.3e}) must be worse than two ({thick:.3e}); if it is not, \
         the pad is not reaching the graph at all"
    );
    assert!(
        thick <= 0.02,
        "two windows of context around a 16-pixel tile left {thick:.3e} of error; the README \
         documents this at 8-bit levels"
    );
}
