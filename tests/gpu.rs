//! Both backends, on the SAME image, at a width the fixtures cannot reach.
//!
//! WHY THIS EXISTS. `tests/parity.rs` runs every golden fixture through `Cpu`
//! only, and the fixtures are 37x29 - so `cargo test` never executes the device
//! graph at all, and there was nothing in the suite that would have failed when
//! `ss_conv3x3_shuffle2` staged its shared rows with two writer threads per
//! address. That race is silent below one warp and wrong above it: the host sizes
//! the block as `ceil(padded_width / 4)`, so a padded plane up to 128 is a single
//! warp and agrees, and everything wider diverges by up to 255 8-bit levels. The
//! fixtures are 37 wide (plane 40, block 10) and the classical-x4 head's second
//! octave is 80 (block 20), both inside one warp - which is exactly why every
//! test passed while every real image was wrong.
//!
//! SO THE PROBE HAS TO BE A WIDTH, NOT A FIXTURE. This runs the same deterministic
//! input through both backends at a size above the boundary and holds them to a
//! fraction of one 8-bit level, the same standard the README quotes for the two
//! backends against the reference. No golden file is needed: the CPU path is
//! verified against the published network independently, so agreement between the
//! two backends is a real check rather than a shared-mistake check.
//!
//! It is skipped without a converted checkpoint (see `tests/parity.rs` for why),
//! and it needs a device the CUDA build can open.


// CPU-only builds have no device to compare against, and `swin2sr::gpu` does not
// exist there. The whole file is gated rather than each test, so a
// `--no-default-features` run has an empty binary here instead of a compile error.
#![cfg(feature = "cuda")]

mod common;

use common::weights;
use swin2sr::backend::{Backend, Pre};
use swin2sr::cpu::Cpu;
use swin2sr::weights::Weights;

/// The worst disagreement allowed between the two backends, in [0,1] units.
///
/// ONE 8-BIT LEVEL IS 1/255 = 3.9e-3, and the two backends reach 1 level at these
/// sizes because a value that lands within 1e-6 of a rounding boundary flips.
/// The measured figure is a single level and never more, so this sits at a
/// level and a half: it cannot pass the race (which is 98-116 levels at the
/// sizes below) and it cannot fail on rounding.
const TOL: f32 = 6e-3;

fn image(h: usize, w: usize) -> Vec<f32> {
    // Deterministic, smooth, and with all three channels different: a sine plane
    // cannot manufacture a large disagreement out of a small one the way a
    // saturation boundary can, so a failure here is the network and not rounding.
    let mut v = Vec::with_capacity(3 * h * w);
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let t = ((x as f32) * 0.31 + (y as f32) * 0.17 + c as f32 * 1.7).sin();
                v.push(0.5 + 0.45 * t);
            }
        }
    }
    v
}

fn run<B: Backend>(
    wt: &Weights, b: &mut B, h: usize, w: usize,
) -> Vec<f32> {
    let pre = Pre::new(wt, h, w).expect("at least one window across");
    let adjusted = pre.adjust(wt, &image(h, w));
    b.forward(h, w, &adjusted).expect("forward")
}

fn agree(wt: &Weights, h: usize, w: usize) -> (f32, usize) {
    let mut cpu = Cpu::new(wt).expect("the CPU backend");
    let want = run(wt, &mut cpu, h, w);
    let mut gpu = swin2sr::gpu::Gpu::new(wt).expect("open the device");
    let got = run(wt, &mut gpu, h, w);
    assert_eq!(want.len(), got.len(), "the two backends returned different sizes");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for i in 0..want.len() {
        let d = (want[i] - got[i]).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

/// The compressed head, on both backends, INCLUDING its second output.
///
/// This is the only head whose device path differs from the CPU one in KIND rather
/// than in kernel: the bicubic resample runs on the HOST (there is no device kernel
/// for it) and the octaves' output is combined with a separate `lg_add`, so the two
/// backends reach the same numbers by visibly different routes - which is exactly
/// when a comparison between them is worth having. The aux is compared at the
/// padded plane's own size, since that is the geometry the head emits.
#[test]
fn the_compressed_head_agrees_with_its_aux() {
    let Some(wt) = weights("compressed-x4") else { return };
    let (worst, at) = agree(&wt, 64, 64);
    eprintln!("compressed x4 64x64 (plane 72, head octaves 72/144/288): max |diff| {worst:.3e} at {at}");
    assert!(
        worst <= TOL,
        "the two backends disagree by {worst:.3e} at {at} ({:.0} 8-bit levels)",
        worst * 255.0
    );

    let mut cpu = Cpu::new(&wt).expect("the CPU backend");
    run(&wt, &mut cpu, 64, 64);
    let aux_cpu = cpu.aux().expect("the compressed head produces an aux").to_vec();
    let mut gpu = swin2sr::gpu::Gpu::new(&wt).expect("open the device");
    run(&wt, &mut gpu, 64, 64);
    let aux_gpu = gpu.aux().expect("the compressed head produces an aux").to_vec();
    assert_eq!(aux_cpu.len(), aux_gpu.len(), "the two aux planes are different sizes");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (a, b)) in aux_cpu.iter().zip(&aux_gpu).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    eprintln!("compressed x4 aux 40x40: max |diff| {worst:.3e} at {at}");
    assert!(worst <= TOL, "the two aux planes disagree by {worst:.3e} at {at}");
}

/// THE REGRESSION GUARD: a padded plane whose head octaves are wider than one
/// warp. 64x64 pads to 72, and the classical-x4 head runs its fused conv at 72 and
/// at 144 - block 18 and block 36 - so the second one is a multi-warp block.
#[test]
fn the_backends_agree_above_the_warp_boundary() {
    let Some(wt) = weights("classical-x4") else { return };
    let (worst, at) = agree(&wt, 64, 64);
    eprintln!("classical x4 64x64 (plane 72, head octaves 72/144): max |diff| {worst:.3e} at {at}");
    assert!(
        worst <= TOL,
        "the two backends disagree by {worst:.3e} at {at}, which is {:.0} 8-bit levels - a \
         kernel that stages shared memory with more than one writer per address looks exactly \
         like this, and it is invisible below one warp (a padded plane up to 128)",
        worst * 255.0
    );
}

/// The same check just under the boundary, so a fix that merely moves the failure
/// is not mistaken for agreement.
#[test]
fn the_backends_agree_below_the_warp_boundary() {
    let Some(wt) = weights("classical-x4") else { return };
    let (worst, at) = agree(&wt, 56, 56);
    eprintln!("classical x4 56x56 (plane 64, head octaves 64/128): max |diff| {worst:.3e} at {at}");
    assert!(worst <= TOL, "the two backends disagree by {worst:.3e} at {at}");
}

/// A wider case again, because the block keeps growing with the width: 128x128
/// pads to 136 and the octaves are 136 and 272, so the block is 34 and 68 threads.
#[test]
fn the_backends_agree_at_a_wide_plane() {
    let Some(wt) = weights("classical-x4") else { return };
    let (worst, at) = agree(&wt, 128, 128);
    eprintln!("classical x4 128x128 (plane 136, head octaves 136/272): max |diff| {worst:.3e} at {at}");
    assert!(worst <= TOL, "the two backends disagree by {worst:.3e} at {at}");
}

/// The x2 checkpoint reaches the boundary at a different place - one octave, at
/// the plane width itself - so it is a separate cell of the same rule and not a
/// duplicate of the tests above.
#[test]
fn the_x2_backends_agree_above_the_warp_boundary() {
    let Some(wt) = weights("classical-x2") else { return };
    let (worst, at) = agree(&wt, 128, 128);
    eprintln!("classical x2 128x128 (plane 136, head octave 136): max |diff| {worst:.3e} at {at}");
    assert!(worst <= TOL, "the two backends disagree by {worst:.3e} at {at}");
}
