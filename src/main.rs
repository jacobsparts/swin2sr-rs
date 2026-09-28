//! swin2sr - Swin2SR image restoration as one binary.
//!
//!   swin2sr -m model.safetensors -i in.png -o out.png
//!   swin2sr -m model.safetensors -i in.png --tile 256 --device cpu
//!   swin2sr -m model.safetensors --verify tests/data/classical_x4.bin
//!
//! See README.md for the measured numbers behind `--tile` and `--device`.
use std::time::Instant;

use swin2sr::backend::{
    activation_bytes, auto_tile, crop_side, run_tiled, Backend, Pre, GPU_MAX_PLANE_PIXELS,
};
use swin2sr::cpu::Cpu;
use swin2sr::fixture::Fixture;
#[cfg(feature = "cuda")]
use swin2sr::gpu;
use swin2sr::image;
use swin2sr::weights::{Upsampler, Weights};

const USAGE: &str = "\
swin2sr - Swin2SR image restoration (super-resolution)

    swin2sr -m <weights.safetensors> -i <in.png> -o <out.png> [options]

    -m, --model <path>    converted .safetensors checkpoint (see tools/convert.py)
    -i, --input <path>    input PNG, or - for stdin (default: stdin)
    -o, --output <path>   output PNG, or - for stdout (default: stdout)
        --device <dev>    cpu or gpu (default: gpu when built with the `cuda`
                          feature and a driver is there, else cpu)
        --tile <n>        process in tiles of n pixels a side; 0 (default) is one
                          pass, which is the only exact mode; `auto` picks the
                          largest window-aligned tile that fits the budget
        --tile-pad <n>    context kept around each tile, rounded up to a window
                          multiple (default 32 = 4 windows)
        --verify <fixture> run the golden fixture instead of an image: report the
                          worst difference against the reference and exit non-zero
                          if it exceeds --tol
        --tol <f>         tolerance for --verify (default 2e-3, see README)
        --aux <path>      ALSO write the head's second output image, for the
                          compressed_sr checkpoint (one pass only: it is produced
                          at the padded plane, which tiling does not preserve)
    -q, --quiet           no progress output
        --cuda-selftest   compare each CUDA kernel against its CPU twin and exit
        --self-test       run the CPU library's internal checks and exit
        --list-weights    print the checkpoint's architecture and tensor names
        --info            print what this binary and this machine can do, and exit
    -h, --help            this text
    -V, --version         print the version";

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("{USAGE}");
        std::process::exit(0);
    }
    let mut model: Option<String> = None;
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut device: Option<String> = None;
    let mut tile = 0usize;
    let mut tile_pad = 32usize;
    let mut verify: Option<String> = None;
    let mut aux_out: Option<String> = None;
    let mut tol = 2e-3f32;
    let mut quiet = false;
    let mut selftest = false;
    let mut cuda_selftest = false;
    let mut list_weights = false;
    let mut info = false;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let val = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_else(|| die(&format!("{a} needs a value")))
        };
        match a {
            "-m" | "--model" => model = Some(val(&mut i)),
            "-i" | "--input" => input = Some(val(&mut i)),
            "-o" | "--output" => output = Some(val(&mut i)),
            "--device" => device = Some(val(&mut i)),
            "--tile" => {
                let v = val(&mut i);
                tile = if v == "auto" { usize::MAX } else {
                    v.parse().unwrap_or_else(|_| die("--tile takes a number or `auto`"))
                };
            }
            "--tile-pad" => tile_pad = val(&mut i).parse().unwrap_or_else(|_| die("--tile-pad takes a number")),
            "--verify" => verify = Some(val(&mut i)),
            "--aux" => aux_out = Some(val(&mut i)),
            "--tol" => tol = val(&mut i).parse().unwrap_or_else(|_| die("--tol takes a number")),
            "-q" | "--quiet" => quiet = true,
            "--cuda-selftest" => cuda_selftest = true,
            "--self-test" => selftest = true,
            "--list-weights" => list_weights = true,
            "--info" => info = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            "-V" | "--version" => {
                println!("swin2sr {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            other => die(&format!("unknown argument `{other}` (--help)")),
        }
        i += 1;
    }

    if info {
        print_info();
        return;
    }
    if selftest {
        match swin2sr::cpu::selftest() {
            Ok(()) => {
                println!("self-test: ok");
                return;
            }
            Err(e) => die(&e),
        }
    }

    let model = model.unwrap_or_else(|| die("--model is required (--help)"));
    let wt = match Weights::load(&model) {
        Ok(w) => w,
        Err(e) => die(&e),
    };

    if list_weights {
        print_weights(&wt);
        return;
    }

    // The device choice: explicit, or the best available.
    let asked = match device.as_deref() {
        Some("gpu") => Some(true),
        Some("cpu") => Some(false),
        Some(other) => die(&format!("--device takes cpu or gpu, not `{other}`")),
        None => None,
    };
    if asked == Some(true) && !cfg!(feature = "cuda") {
        die("this binary was built without the `cuda` feature; use --device cpu");
    }
    // THE DEFAULT DEVICE FALLS BACK, THE EXPLICIT ONE DOES NOT.
    //
    // Every engine in this family has to run on a machine with no CUDA driver -
    // a laptop, a container without `--gpus`, a card whose driver did not load -
    // and "the GPU is the default" must not mean "the engine is dead there". So
    // the default probes ONCE, before the banner, before the tile is chosen and
    // before anything is allocated, and a probe that fails makes this a CPU run.
    //
    // `--device gpu` typed by hand is a different request and stays fatal: the
    // user named the device, and a run that quietly went to the CPU would take
    // minutes where they expected seconds, with nothing to say why. That is the
    // failure this engine's original design was protecting against, and it is
    // still protected - it is just no longer the default's behaviour.
    //
    // The note is printed even under `-q`, like the `--tile 0` fallback: `-q`
    // silences progress, not a change in what the run IS.
    let mut want_gpu = asked.unwrap_or(cfg!(feature = "cuda"));
    if want_gpu && asked.is_none() {
        #[cfg(feature = "cuda")]
        if let Err(e) = lightgpu::vm::device() {
            want_gpu = false;
            eprintln!(
                "note: no usable CUDA device ({}) - running on the CPU. Pass --device gpu to \
                 make this an error instead.",
                e
            );
        }
    }

    if cuda_selftest {
        #[cfg(feature = "cuda")]
        {
            match swin2sr::gpu::selftest(&wt) {
                Ok(()) => {
                    println!("cuda selftest: ok");
                    return;
                }
                Err(e) => die(&e),
            }
        }
        #[cfg(not(feature = "cuda"))]
        die("--cuda-selftest needs a build with the `cuda` feature");
    }

    if let Some(path) = verify {
        let f = match Fixture::load(&path) {
            Ok(f) => f,
            Err(e) => die(&e),
        };
        std::process::exit(verify_fixture(&wt, &f, want_gpu, tol, quiet));
    }

    // ---------------------------------------------------------------------
    // The image path.
    // ---------------------------------------------------------------------
    let input = input.unwrap_or_else(|| "-".into());
    let img = match input.as_str() {
        "-" => {
            let stdin = std::io::stdin();
            match image::load_rgb_stream(stdin.lock()) {
                Ok(i) => i,
                Err(e) => die(&format!("stdin: {e}")),
            }
        }
        path => match image::load_rgb(path) {
            Ok(i) => i,
            Err(e) => die(&e),
        },
    };

    let plan = match Pre::new(&wt, img.h, img.w) {
        Ok(p) => p.plan,
        Err(e) => die(&e),
    };
    // `--aux` asks for the head's second image, which exists only for a single
    // pass and only for one head. Both facts are knowable HERE, before the image is
    // read and before anything is allocated - so the refusals belong here rather
    // than after a PNG has already been written, which is what a check at the end
    // would do. (`--tile auto` can still fall back to a smaller tile; the write
    // below keeps its own check for that case.)
    if aux_out.is_some() {
        if wt.upsampler != Upsampler::PixelShuffleAux {
            die(&format!(
                "--aux: the {} head has one output; only the compressed_sr head \
                 (pixelshuffle_aux) produces a second image",
                wt.upsampler.name()
            ));
        }
        if tile != 0 {
            die(&format!(
                "--aux: the second image exists only for a single pass, and --tile {tile} would \
                 compute it per tile. Use --tile 0 (the default)."
            ));
        }
    }
    let budget = memory_budget(want_gpu, &wt);
    let tile = if tile == usize::MAX {
        auto_tile(&wt, budget, img.h, img.w, tile_pad, want_gpu)
    } else {
        tile
    };
    // A LAUNCH LIMIT, WHICH `--tile auto` CANNOT FIX FOR A TILE THE USER TYPED.
    // Past `GPU_MAX_PLANE_SIDE` the token matmuls cannot be launched at all: the
    // driver's error names a kernel and not the constraint, and the run has
    // already read and padded the image by then. The check belongs here, before
    // anything is allocated, and it is the device's question - a host run has no
    // grid to run out of.
    if want_gpu {
        let side = crop_side(img.h, img.w, tile, tile_pad, wt.window) as u64;
        if side * side > GPU_MAX_PLANE_PIXELS {
            die(&format!(
                "a {side}x{side} padded plane is past what a device launch can address: {} \
                 tokens is `grid.y`'s 65535 blocks of 16 rows and one padded pixel is one \
                 token, so the plane has at most {} pixels a side. This is a launch limit \
                 rather than a memory one, and a smaller tile is the only fix - `--tile auto` \
                 counts it, a `--tile` you typed is not second-guessed",
                GPU_MAX_PLANE_PIXELS,
                (GPU_MAX_PLANE_PIXELS as f64).sqrt() as u64,
            ));
        }
    }
    if !quiet {
        let need = activation_bytes(&wt, img.h, img.w, tile, tile_pad, want_gpu);
        eprintln!(
            "swin2sr {} : {} {}x{} -> {}x{} at x{} (padded {}x{}, {} window{})",
            wt.task,
            wt.upsampler.name(),
            img.w,
            img.h,
            img.w * wt.scale,
            img.h * wt.scale,
            wt.scale,
            plan.wp,
            plan.hp,
            plan.nw,
            if plan.nw == 1 { "" } else { "s" },
        );
        eprintln!(
            "  backend {} | activations {:.1} MiB{} | tile {}{}",
            if want_gpu { "cuda" } else { "cpu" },
            need as f64 / (1024.0 * 1024.0),
            if tile == 0 { "" } else { " per tile" },
            if tile == 0 {
                "whole image".to_string()
            } else {
                // The EFFECTIVE tile and pad, rounded the way `run_tiled` rounds
                // them: a banner that echoed the request would say `pad 0` while
                // running with a whole window of context.
                let win = wt.window;
                format!(
                    "{} (pad {})",
                    (tile / win).max(1) * win,
                    tile_pad.div_ceil(win).max(1) * win
                )
            },
            if tile != 0 && tile < img.h.max(img.w) {
                " - tiled results differ slightly from one pass (see README)"
            } else {
                ""
            }
        );
    }

    let t0 = Instant::now();
    let mut backend: Box<dyn Backend + '_> = if want_gpu {
        #[cfg(feature = "cuda")]
        {
            match swin2sr::gpu::Gpu::new(&wt) {
                Ok(g) => Box::new(g),
                Err(e) => die(&e),
            }
        }
        #[cfg(not(feature = "cuda"))]
        unreachable!()
    } else {
        Box::new(match Cpu::new(&wt) {
            Ok(c) => c,
            Err(e) => die(&e),
        })
    };
    let (out, ran) = match run_with_fallback(&wt, backend.as_mut(), &img, tile, tile_pad, quiet)
    {
        Ok(o) => o,
        Err(e) => die(&e),
    };
    let dt = t0.elapsed();
    #[cfg(feature = "cuda")]
    if want_gpu {
        // Only meaningful for the device backend, and a no-op without
        // `SWIN2SR_PROFILE_GPU`; the sync is what makes the per-kernel event
        // times comparable to `dt`, and it is a sync the process was about to
        // earn anyway when it wrote the PNG.
        if lightgpu::vm::sync().is_ok() {
            swin2sr::cuda::Profile::report(dt.as_secs_f32() * 1e3);
        }
    }
    if !quiet {
        // The banner above named the tile the run WOULD have used; if the
        // fallback shrank it, say so here rather than leaving the two lines to
        // disagree about what happened. The note itself is on stderr already, so
        // this is the pointer to it and not a second copy.
        if ran != tile {
            eprintln!(
                "  (ran with a {ran}x{ran} tile, not the {tile} the banner above chose - see the \
                 note on stderr)"
            );
        }
        // The rate is reported against the INPUT, which is what the user asked
        // for; at one decimal a small image reads as `0.0 Mpx/s`, so the precision
        // follows the magnitude.
        let rate = (img.w * img.h) as f64 / 1e6 / dt.as_secs_f64().max(1e-9);
        let rate = if rate >= 0.1 { format!("{rate:.1}") } else { format!("{rate:.3}") };
        eprintln!("  {:.2}s ({rate} Mpx/s in)", dt.as_secs_f64());
    }

    let rgb = out.to_rgb8();
    match output.as_deref() {
        None | Some("-") => {
            let stdout = std::io::stdout();
            if let Err(e) = image::save_rgb_stream(stdout.lock(), out.w, out.h, &rgb) {
                die(&e);
            }
        }
        Some(path) => {
            if let Err(e) = image::save_rgb(path, out.w, out.h, &rgb) {
                die(&e);
            }
            if !quiet {
                eprintln!("  wrote {path}");
            }
        }
    }

    // The head's second image, when one was asked for. It lives on the PADDED
    // plane and is only well defined for a single pass: a tiled run computes it
    // per tile, and the last tile's plane is not the image's. So a tiled request is
    // REFUSED rather than answered with something that looks like a thumbnail of
    // the wrong region.
    if let Some(path) = aux_out.as_deref() {
        // The head and the requested tile were checked before the run; this is the
        // one case that check cannot cover, `--tile auto` deciding after the fact
        // that the whole image does not fit.
        if ran != 0 {
            die(&format!(
                "--aux: the second image exists only for a single pass, and this run used a \
                 {ran}x{ran} tile. Re-run with --tile 0 (or --tile auto on an image that fits)."
            ));
        }
        let aux = backend
            .aux()
            .unwrap_or_else(|| die("--aux: the backend produced no second image"));
        let (aw, ah) = (plan.wp, plan.hp);
        if aux.len() != 3 * aw * ah {
            die(&format!(
                "--aux: expected {} values for a {aw}x{ah} plane, the backend produced {}",
                3 * aw * ah,
                aux.len()
            ));
        }
        // The same quantisation the main image gets, at the aux plane's own size.
        let img_aux = image::Image { w: aw, h: ah, data: aux.to_vec() };
        let rgb_aux = img_aux.to_rgb8();
        if let Err(e) = image::save_rgb(path, aw, ah, &rgb_aux) {
            die(&e);
        }
        if !quiet {
            eprintln!("  wrote {path} ({aw}x{ah} from the compressed head)");
        }
    }
}

/// The activation budget a run gets before tiling becomes mandatory.
///
/// WHAT THIS IS FOR. `--tile auto` has to pick a tile without running anything,
/// and a run that is going to be too large should be tiled BEFORE it allocates
/// rather than after it dies - especially on the CPU, where an allocation
/// failure is an abort and not an error the process can catch. So the budget is
/// the memory a run can have, and it has to be honest about two things a device
/// and a host disagree about.
///
/// * THE DEVICE'S FREE VRAM IS NOT THE WHOLE STORY. `cuMemGetInfo` answers for
///   the card, not for this process: the checkpoint's own tensors, once
///   `Gpu::new` has uploaded them, are already spent, and a budget that ignores
///   them would let a 74 MiB checkpoint's upload push an "auto" tile over the
///   edge. `gpu::uploaded_bytes` is exactly that cost. The margin on top is for
///   the allocator's own fragmentation and for the two buffers whose size the
///   per-pixel count does not model to the byte (the padded host copy and the
///   pixel-shuffle intermediates).
/// * THE HOST'S BUDGET IS NOT FIXED. A 1 GiB constant - what this used to be -
///   is both too small on a 32 GB machine and too large on a loaded one; the
///   kernel's own `MemAvailable` is the number that answers "may I allocate
///   this", because it counts what can be had without swapping and without
///   pushing the machine into a fight with whatever else is running.
///
/// The CPU and GPU figures are deliberately not the same shape: on the device
/// the failure is `cuMemAlloc` returning out-of-memory and the engine can RETRY
/// smaller (see `run_with_fallback`), while on the host there is nothing to
/// catch, so the only defence is to stay under `MemAvailable` - and to stay well
/// under it, because `MemAvailable` is a number that is already falling while
/// this runs.
fn memory_budget(gpu: bool, wt: &Weights) -> u64 {
    // The checkpoint is only read by the CUDA branch below, where its upload is
    // spent whatever the tile is. Naming it in the CPU build as well keeps ONE
    // signature for the two builds rather than two, which is what the call site
    // wants - it passes the flag and the checkpoint without asking which build it
    // is in. Without this the pure-Rust build warns about an unused parameter.
    #[cfg(not(feature = "cuda"))]
    let _ = wt;
    if gpu {
        #[cfg(feature = "cuda")]
        {
            if let Ok((free, total)) = lightgpu::vm::vram() {
                let free = free as u64;
                let total = total as u64;
                // What the card can hold for us: free now, or (if the driver's
                // free figure is already pessimistic because of a foreign
                // process) all of it less a margin for the desktop.
                let usable = free.min(total.saturating_sub(total / 16));
                // The upload is spent regardless of the tile, so it comes off the
                // top; the margin covers fragmentation and the allocator's own
                // per-buffer rounding.
                let reserved = gpu::uploaded_bytes(wt) + (usable / 16).max(32 << 20);
                return usable.saturating_sub(reserved).max(32 << 20);
            }
        }
        256 << 20
    } else {
        match available_ram() {
            Some(avail) => {
                // Half of what the kernel says it can hand over: the engine's own
                // peak includes the input and output images, the weight mapping
                // and rayon's per-thread stacks, none of which are in
                // `per_pixel_floats`, and a machine with 10 GB "available" that is
                // asked for 10 GB is a machine that swaps.
                (avail / 2).max(256 << 20)
            }
            // No /proc: a conservative fixed figure, which is the old behaviour.
            None => 1 << 30,
        }
    }
}

/// `MemAvailable` from /proc/meminfo, or `None` where there is no /proc.
///
/// NOT `MemFree`: free memory is what is not being used for anything, and on a
/// machine with a warm page cache it is the smaller and less useful number -
/// here 9.7 GB free against 19.0 GB available, on a machine that would happily
/// allocate 15. What the allocator can actually get is the available figure,
/// which counts reclaimable page cache with the file-backed weights mapped into
/// it.
fn available_ram() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn print_info() {
    println!("swin2sr {}", env!("CARGO_PKG_VERSION"));
    println!("  built with cuda: {}", cfg!(feature = "cuda"));
    #[cfg(feature = "cuda")]
    {
        match lightgpu::vm::device() {
            Ok(d) => println!("  device: {} (sm_{}{})", d.name, d.cc_major, d.cc_minor),
            Err(e) => println!("  device: unavailable ({e})"),
        }
    }
}

fn print_weights(w: &Weights) {
    println!("task {} upsampler {} scale {} window {}", w.task, w.upsampler.name(), w.scale, w.window);
    println!(
        "embed {} heads {} head dim {} mlp ratio {} depths {:?}",
        w.embed, w.heads, w.head_dim, w.mlp_ratio, w.depths
    );
    println!("img_range {} mean {:?}", w.img_range, w.mean);
    println!("{} tensors, {:.1} MiB", w.names.len(), w.bytes as f64 / (1024.0 * 1024.0));
    for n in &w.names {
        println!("  {n}");
    }
}

fn backend_for<'a>(want_gpu: bool, wt: &'a Weights) -> Box<dyn Backend + 'a> {
    if want_gpu {
        #[cfg(feature = "cuda")]
        {
            match swin2sr::gpu::Gpu::new(wt) {
                Ok(g) => Box::new(g),
                Err(e) => die(&e),
            }
        }
        #[cfg(not(feature = "cuda"))]
        unreachable!()
    } else {
        Box::new(Cpu::new(wt).unwrap_or_else(|e| die(&e)))
    }
}

/// Run the image, and when the device runs out of memory, try again smaller.
///
/// WHY THIS EXISTS. `--tile auto` predicts, and a prediction can be wrong: the
/// driver's free figure moves while the run allocates, the desktop takes a share,
/// and the engine's own per-pixel model is a model. Before this, such a run died
/// with `cuMemAlloc failed: CUDA_ERROR_OUT_OF_MEMORY` while the fix - one smaller
/// tile - was a retry the program could have done itself.
///
/// THE ONE THING THIS MUST NOT DO IS RETRY THE WRONG ERROR. A wrong launch, an
/// illegal address or a failed kernel are bugs, and running them again at half
/// the size would turn "this build is broken" into "this build is slow and then
/// broken". `is_out_of_memory` matches the one class of failure that a smaller
/// tile can fix.
///
/// THE CPU IS NOT RETRIED, because it cannot be: its allocations are `Vec`s, and
/// a failed `Vec` allocation aborts the process rather than returning an error.
/// The host's protection is the budget (`memory_budget`), applied before the run
/// by `auto_tile`; this loop is the device's protection, which is possible only
/// because `cuMemAlloc` reports failure.
///
/// `--tile 0` IS THE ONLY EXACT MODE, so a fallback out of it is announced on
/// stderr even under `-q`: a user who asked for an exact pass and got a tiled one
/// has to be told, or the flag `--tile 0` quietly means "0 unless it does not
/// fit" and the difference that the README measured becomes invisible.
///
/// Returns the image and the tile size that produced it, which is what the rate
/// line and the caller's own reporting want.
fn run_with_fallback(
    wt: &Weights,
    backend: &mut dyn Backend,
    img: &image::Image,
    tile: usize,
    pad: usize,
    quiet: bool,
) -> Result<(image::Image, usize), String> {
    let mut tile = tile;
    // A hundred steps is not a real limit - the tile halves and the floor is one
    // window, so this cannot loop - but it stops a hypothetical non-shrinking
    // update from hanging the program in the dark.
    for _ in 0..64 {
        match run_tiled(wt, backend, img, tile, pad) {
            Ok(out) => return Ok((out, tile)),
            // `tile == 0` IS NOT A SMALL TILE. Written as `tile > wt.window` this
            // guard excludes the single most important case - the whole-image pass
            // that did not fit - because `0 > 8` is false, so `--tile 0` and a
            // whole-image `--tile auto` died with the driver's message and no retry.
            // Every other tile above one window took the branch, which is why the
            // bug only showed at the sizes where auto chose the whole image.
            Err(e) if is_out_of_memory(&e) && (tile > wt.window || tile == 0) => {
                // HALVE, ALWAYS - never re-ask `auto_tile`.
                //
                // THE BUDGET THAT JUST FAILED IS NOT EVIDENCE FOR ITSELF. Asking
                // `auto_tile` again with the same budget returns `0` again - it is
                // saying the whole pass fits, which is exactly what the driver has
                // just denied - and the retry then has nowhere to go. So a failed
                // whole-image pass falls back to HALF THE IMAGE, rounded down to a
                // window multiple: the engine's own model is wrong here and the
                // only trustworthy direction is down.
                let next = if tile == 0 {
                    (img.h.max(img.w) / 2 / wt.window).max(1) * wt.window
                } else {
                    (tile / 2 / wt.window).max(1) * wt.window
                };
                // `tile == 0` is not a small tile, it is "no tiling at all", so
                // `next >= tile` is true for EVERY positive `next` and would abort
                // the first, largest, most useful retry. The step is only illegal
                // when it does not actually shrink: at or below one window there is
                // nothing left to give.
                let shrinking = if tile == 0 { next != 0 } else { next < tile };
                if !shrinking || next == 0 {
                    return Err(format!(
                        "{e}\n(a {}-pixel tile is the smallest this engine will run, and it did \
                         not fit either)",
                        wt.window
                    ));
                }
                // ANNOUNCED EVEN UNDER `-q`. `--tile 0` means "one exact pass",
                // and a run that quietly becomes a tiled one has changed what the
                // user asked for rather than how fast it happened; the note about
                // a tile the user already chose is progress output and takes `-q`
                // like the rest.
                if tile == 0 {
                    eprintln!(
                        "note: {} - retrying with a {}x{} tile (--tile 0 is the only exact mode; \
                         this run is now tiled, and tiled results differ slightly from one pass - \
                         see README)",
                        e.trim_end_matches('.'),
                        next,
                        next,
                    );
                } else if !quiet {
                    eprintln!("note: {} - retrying with a {next}x{next} tile", e.trim_end_matches('.'));
                }
                tile = next;
            }
            Err(e) => return Err(e),
        }
    }
    Err(format!("gave up after 64 tile reductions at tile {tile}"))
}

/// Is this failure one a smaller tile can fix?
///
/// `lightgpu` builds every error as `<what> failed: <name>` (`ffi::chk`), so the
/// driver's own name is in the message: `cuMemAlloc failed:
/// CUDA_ERROR_OUT_OF_MEMORY` is the one that is ours to retry. Everything else -
/// a launch failure, an illegal address, a missing kernel - is a bug, and a
/// smaller tile would only make it fail slower.
fn is_out_of_memory(err: &str) -> bool {
    err.contains("OUT_OF_MEMORY") || err.contains("out of memory")
}

/// Run a golden fixture and report the difference. Returns the process exit code.
fn verify_fixture(wt: &Weights, f: &Fixture, want_gpu: bool, tol: f32, quiet: bool) -> i32 {
    if f.c != 3 {
        die(&format!("the fixture has {} channels, this engine restores RGB", f.c));
    }
    if f.scale != wt.scale {
        die(&format!("the fixture is scale {}, the checkpoint is x{}", f.scale, wt.scale));
    }
    if f.win != wt.window {
        die(&format!("the fixture was made with window {}, the checkpoint uses {}", f.win, wt.window));
    }
    let mut backend = backend_for(want_gpu, wt);
    let t0 = Instant::now();
    let pre = match Pre::new(wt, f.h, f.w) {
        Ok(p) => p,
        Err(e) => die(&e),
    };
    let adjusted = pre.adjust(wt, &f.input);
    let got = match backend.forward(f.h, f.w, &adjusted) {
        Ok(g) => g,
        Err(e) => die(&e),
    };
    let dt = t0.elapsed();
    let (worst, at, mean) = f.compare(&got);
    let (y, x, c) = f.locate(at);
    if !quiet {
        println!(
            "{} {}: {}x{} -> {}x{}, {:.2}s",
            backend.name(),
            wt.task,
            f.w,
            f.h,
            f.w * f.scale,
            f.h * f.scale,
            dt.as_secs_f64()
        );
    }
    println!("  max |diff| {worst:.6} at (x{x}, y{y}, c{c})  mean |diff| {mean:.8}  tol {tol:.1e}");
    let ok_main = worst <= tol;
    if !ok_main {
        println!("  FAIL - the reference and this engine disagree by more than the tolerance");
        println!("  expected {:+.6}, got {:+.6}", f.expected[at], got[at]);
    }
    // The compressed head returns a SECOND image, and a fixture made from it
    // carries a second plane. Both directions are errors worth reporting: a
    // backend that produces one where the fixture has none is writing a file the
    // reference never produced, and the reverse is a head that quietly dropped
    // half its output.
    let mut ok_aux = true;
    match (f.aux.as_deref(), backend.aux()) {
        (None, None) => {}
        (Some(want), Some(aux_got)) => {
            let (ah, aw) = f.aux_plane();
            if aux_got.len() != want.len() {
                println!("  aux: {} values, the fixture has {}", aux_got.len(), want.len());
                ok_aux = false;
            } else {
                let mut aw_worst = 0.0f32;
                let mut aw_at = 0usize;
                let mut aw_mean = 0.0f64;
                for (i, (a, b)) in want.iter().zip(aux_got).enumerate() {
                    let d = (a - b).abs();
                    aw_mean += d as f64;
                    if d > aw_worst {
                        aw_worst = d;
                        aw_at = i;
                    }
                }
                let aw_mean = (aw_mean / want.len() as f64) as f32;
                // The aux plane is on the PADDED grid, so its index is a plane
                // index and not the output pixel `locate` would give.
                let plane = ah * aw;
                println!(
                    "  aux {aw}x{ah}: max |diff| {aw_worst:.6} at (x{}, y{}, c{})  mean |diff| {aw_mean:.8}",
                    aw_at % aw,
                    (aw_at % plane) / aw,
                    aw_at / plane
                );
                ok_aux = aw_worst <= tol;
                if !ok_aux {
                    println!("  aux FAIL - expected {:+.6}, got {:+.6}", want[aw_at], aux_got[aw_at]);
                }
            }
        }
        (Some(want), None) => {
            println!(
                "  aux: the fixture has a {}x{} aux plane and this backend produced none",
                f.aux_plane().1,
                f.aux_plane().0
            );
            let _ = want;
            ok_aux = false;
        }
        (None, Some(aux_got)) => {
            println!("  aux: this backend produced {} values the fixture does not have", aux_got.len());
            ok_aux = false;
        }
    }
    if ok_main && ok_aux {
        println!("  PASS");
        0
    } else {
        1
    }
}
