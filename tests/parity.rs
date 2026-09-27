//! Golden-fixture parity: the published PyTorch network against this engine.
//!
//! The fixture in `tests/data/` was produced by `tools/make_fixture.py` from the
//! release checkpoint with a seeded input, so this test does not need torch, a
//! network, or an image: it is `checkpoint + fixture` in, a number out.
//!
//! THE TOLERANCE IS A MEASUREMENT, NOT A GUESS. Two independent f32 evaluations of
//! a 36-block network - torch's blocked matmuls and this engine's loops - cannot
//! agree to the last bit; every 3x3 conv in the residual path amplifies what the
//! one before it left behind. What matters is the size of the disagreement, and
//! the numbers to hold to are: an exact transcription lands at 1e-6 on the final
//! image, and any single wrong tap, transposed layout or missing skip lands at
//! 1e-2 or worse. `TOL` sits between the two with an order of magnitude either
//! side, so it fails loudly on a bug and never on rounding.
//!
//! These tests need the converted checkpoints, which are NOT in the repository:
//! each is ~85 MiB and reproducible from the published `.pth` with
//! `tools/convert.py` in a minute. They live in the family's shared `models/`
//! directory next to the `.pth` files, like every other engine's - see
//! `convert_all.sh`, which does the whole set. A missing checkpoint SKIPS (it does
//! not fail): a fresh clone has no weights, and a test that cannot run is not a
//! failing test.

mod common;

use common::{models_dir, repo, weights};
use swin2sr::backend::{Backend, Pre};
use swin2sr::cpu::Cpu;
use swin2sr::fixture::Fixture;
use swin2sr::weights::Weights;

/// The worst absolute difference allowed on the final image, in [0,1] units.
const TOL: f32 = 2e-3;

fn cpu_forward(wt: &Weights, f: &Fixture) -> Vec<f32> {
    let pre = Pre::new(wt, f.h, f.w).expect("the fixture is at least one window across");
    let adjusted = pre.adjust(wt, &f.input);
    let mut backend = Cpu::new(wt).expect("construct the CPU backend");
    backend.forward(f.h, f.w, &adjusted).expect("run the CPU forward")
}

/// The classical x4 checkpoint against its fixture: the end-to-end check, and the
/// one the README's accuracy table is measured with.
#[test]
fn classical_x4_matches_the_reference() {
    let Some(wt) = weights("classical-x4") else { return };
    let f = Fixture::load(repo("tests/data/classical_x4.bin")).expect("load the fixture");
    assert_eq!(f.scale, wt.scale);
    assert_eq!(f.win, wt.window);
    let got = cpu_forward(&wt, &f);
    let (worst, at, mean) = f.compare(&got);
    let (y, x, c) = f.locate(at);
    eprintln!("classical x4: max {worst:.3e} at (x{x},y{y},c{c}), mean {mean:.3e}");
    assert!(
        worst <= TOL,
        "max |diff| {worst:.3e} > {TOL:.1e} at (x{x}, y{y}, c{c}); \
         the reference has {:+.6} there and this engine produced {:+.6}",
        f.expected[at],
        got[at]
    );
}

/// The same engine, a different checkpoint: x2 with a different depth and a
/// different head. A weights-shape mistake shows up here and nowhere else.
#[test]
fn classical_x2_matches_the_reference() {
    let Some(wt) = weights("classical-x2") else { return };
    let f = Fixture::load(repo("tests/data/classical_x2.bin")).expect("load the fixture");
    assert_eq!(f.scale, wt.scale);
    let got = cpu_forward(&wt, &f);
    let (worst, at, _) = f.compare(&got);
    assert!(worst <= TOL, "x2: max |diff| {worst:.3e} at {at} > {TOL:.1e}");
}

/// The real-world head (`nearest+conv`, with its two LeakyReLU slopes) is a
/// different code path from the pixel-shuffle head above.
#[test]
fn realworld_x4_matches_the_reference() {
    let Some(wt) = weights("realworld-x4") else { return };
    let f = Fixture::load(repo("tests/data/realworld_x4.bin")).expect("load the fixture");
    let got = cpu_forward(&wt, &f);
    let (worst, at, _) = f.compare(&got);
    assert!(worst <= TOL, "real-world x4: max |diff| {worst:.3e} at {at} > {TOL:.1e}");
}

/// The lightweight model is a different width, depth and head
/// (`pixelshuffledirect`, one conv to 3*scale^2 then a shuffle).
#[test]
fn lightweight_x2_matches_the_reference() {
    let Some(wt) = weights("lightweight-x2") else { return };
    let f = Fixture::load(repo("tests/data/lightweight_x2.bin")).expect("load the fixture");
    let got = cpu_forward(&wt, &f);
    let (worst, at, _) = f.compare(&got);
    assert!(worst <= TOL, "lightweight x2: max |diff| {worst:.3e} at {at} > {TOL:.1e}");
}

/// The engine's own library checks. Cheap, and they isolate the ops a fixture can
/// only see in aggregate.
#[test]
fn the_op_library_checks_itself() {
    swin2sr::cpu::selftest().expect("the CPU library's internal checks");
}

/// A checkpoint whose header describes a different architecture than its tensors
/// must be REJECTED, not run. This is the failure mode that otherwise appears as a
/// wrong image after ten minutes of computation.
#[test]
fn a_mismatched_checkpoint_is_refused() {
    let Some(wt) = weights("classical-x4") else { return };
    // The header of the real file, with `depths` claiming one more stage than the
    // tensors provide. `Weights::load` validates every per-block tensor, so the
    // missing stage is what it should notice.
    let src = models_dir().join("swin2sr-classical-x4.safetensors");
    let bytes = std::fs::read(&src).expect("read the checkpoint");
    let mut tampered = bytes.clone();
    // The header is JSON at a fixed offset; patching it in place keeps the byte
    // length and therefore the offsets, which is exactly how a stale or
    // hand-edited container looks.
    let needle = b"6,6,6,6,6,6";
    let at = find(&tampered, needle).expect("the depths field is in the header");
    tampered[at..at + needle.len()].copy_from_slice(b"6,6,6,6,6,7");
    let tmp = std::env::temp_dir().join("swin2sr-tampered.safetensors");
    std::fs::write(&tmp, &tampered).expect("write the tampered copy");
    let err = match Weights::load(&tmp) {
        Ok(_) => panic!("a checkpoint with a stage its header promises but does not contain must fail"),
        Err(e) => e,
    };
    assert!(
        err.contains("layers.5.residual_group.blocks.6"),
        "the error should name the tensor that is missing, not just fail: {err}"
    );
    let _ = wt;
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}
