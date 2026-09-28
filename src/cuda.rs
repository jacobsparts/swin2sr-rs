//! The CUDA half of the engine: the two fatbins, the kernel-name sets, and the
//! launch plumbing.
//!
//! WHICH MODULE A KERNEL LIVES IN IS A PROPERTY OF THE NAME, and the two modules
//! are disjoint: `lightgpu/cuda/kernels.cu` (toolkit) and `cuda/swin2sr.cu` (this
//! engine). `module_of` asks the project module first and then the toolkit, so a
//! kernel promoted from one to the other is a one-line change here. A name in
//! NEITHER fails at construction, where the message can say which of the two
//! lists is wrong.
use lightgpu::vm::{Args, DevBuf, Event, Launch, Module};
use std::cell::RefCell;

/// The toolkit's kernels this graph launches. Every one of them is a name the
/// toolkit defines for anybody, and this list is what `build.rs` compiles into the
/// toolkit fatbin's `--entries`, so a name here that the toolkit does not define
/// fails the build rather than the first forward pass.
///
/// COUNTING THE ROLES RATHER THAN THE NAMES, because that is what makes the list
/// reviewable: `lg_conv3x3_winograd` is conv_first, every stage's `conv`, the
/// body's `conv_after_body` AND the reconstruction head's convs - this model is
/// mostly 3x3 convolutions, and F(4,3) computes 36 products per 16 outputs where
/// the direct kernel computes 144. `lg_conv3x3s1p1` is here as well, and is
/// launched by NOTHING: it is the direct form of the same op, and
/// `--cuda-selftest` uses it as the second opinion on the winograd one, because a
/// transform this engine got wrong would otherwise only show up as a slightly
/// wrong image. `lg_linear` is q, k, v, the attention output projection and both
/// MLP layers, as a shared-memory tiled matmul. `lg_channel_layer_norm` is the
/// top-level `patch_embed.norm` and every block's `norm1`/`norm2`, and the final
/// `norm`. `lg_lrelu` is the head's activations at their two slopes. `lg_add` is
/// every residual.
///
/// `lg_copy` is the one name here that is not a graph op: it is the stage input's
/// copy into `res`, which the CPU backend does with `copy_from_slice` and the
/// device does with a kernel.
///
/// `lg_conv1x1` and `lg_upsample2x_nearest` are the other two: the 1x1
/// `patch_embed.proj` convs at both levels, and the real-world head's 2x upsample.
/// This file used to carry its own copies of both (`ss_conv1x1`,
/// `ss_upsample2x_nearest`) - the same kernels under different names, which is the
/// duplication the promotion test exists to prevent. The selftest is what makes
/// swapping them safe: it compares the LAUNCHED kernel against the CPU function
/// the graph calls (`src/cpu.rs::conv1x1`), not against a toolkit twin.
pub const TOOLKIT_KERNELS: &[&str] = &[
    "lg_conv3x3s1p1",
    "lg_conv3x3_winograd",
    "lg_conv1x1",
    "lg_conv1x1_rb",
    "lg_linear",
    "lg_linear_rb",
    "lg_upsample2x_nearest",
    "lg_channel_layer_norm",
    "lg_add",
    "lg_gelu_erf",
    "lg_lrelu",
    "lg_copy",
];

/// This engine's own kernels, from `cuda/swin2sr.cu`. The same names `build.rs`
/// validates against the source, in both directions.
pub const PROJECT_KERNELS: &[&str] = &[
    "ss_window_gather",
    "ss_window_scatter",
    "ss_attention",
    "ss_conv3x3_shuffle2",
];

const TOOLKIT_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/swin2sr_toolkit.fatbin"));
const PROJECT_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/swin2sr_project.fatbin"));

pub struct Cuda {
    toolkit: Module,
    project: Module,
}

impl Cuda {
    /// Load both fatbins and resolve every kernel the graph can launch, so a name
    /// that is in neither module fails here rather than at the first forward pass.
    pub fn new() -> Result<Cuda, String> {
        let toolkit = Module::load(TOOLKIT_FATBIN)?;
        let project = Module::load(PROJECT_FATBIN)?;
        for k in PROJECT_KERNELS {
            if !project.has(k) {
                return Err(format!(
                    "kernel `{k}` is not in the project fatbin - is it listed in build.rs's \
                     PROJECT_KERNELS and defined in cuda/swin2sr.cu?"
                ));
            }
        }
        for k in TOOLKIT_KERNELS {
            if !toolkit.has(k) {
                return Err(format!(
                    "kernel `{k}` is not in the toolkit fatbin - is it in lightgpu's \
                     src/ops/mod.rs NAMES and cuda/kernels.cu?"
                ));
            }
        }
        Ok(Cuda { toolkit, project })
    }

    /// The module a kernel lives in. Project first, then toolkit.
    pub fn module_of(&self, name: &str) -> &Module {
        if self.project.has(name) {
            &self.project
        } else {
            &self.toolkit
        }
    }

    pub fn run(&self, name: &str, launch: Launch, args: &mut Args) -> Result<(), String> {
        // PER-KERNEL TIMING, off unless asked for. The device pipeline is one
        // stream with no host sync until the end, so the only honest way to say
        // where the time goes is to bracket each launch with events and to read
        // the elapsed times once, after the last one - which is exactly what
        // `lightgpu::vm::Event` exists for. Two Event records per launch, no
        // synchronise per launch: the extra device work is a pair of timestamp
        // writes, and it is what makes "the stage convolution is N% of the run"
        // a measurement rather than an inference from the wall clock.
        let prof = Profile::active();
        let start = match prof {
            Some(_) => Some(Event::new()?),
            None => None,
        };
        let end = match prof {
            Some(_) => Some(Event::new()?),
            None => None,
        };
        if let Some(e) = &start {
            e.record()?;
        }
        args.launch(self.module_of(name), name, launch)?;
        if let (Some(p), Some(s), Some(e)) = (prof, start, end) {
            e.record()?;
            // NOT read here: `cuEventElapsedTime` wants two COMPLETED events, and
            // the point of this profiler is that nothing synchronises between
            // launches. The pair is stashed and drained in `report`, after the one
            // sync the process already earns at the end of the forward.
            p.push(name, s, e, launch.grid, launch.block);
        }
        Ok(())
    }

    pub fn buf(&self, n: usize) -> Result<DevBuf, String> {
        DevBuf::zeros(n * std::mem::size_of::<f32>())
    }

    pub fn upload(&self, v: &[f32]) -> Result<DevBuf, String> {
        DevBuf::from_host(v)
    }
}

/// Grid for `n` elements at `block` threads, as one dimension.
pub fn grid_for(n: usize, block: usize) -> (u32, u32, u32) {
    (((n + block - 1) / block) as u32, 1, 1)
}

/// Per-kernel device times, collected by [`Cuda::run`] when `SWIN2SR_PROFILE_GPU`
/// is set, and printed once by [`Profile::report`].
///
/// The collection is per-kernel-NAME rather than per-call-site, because that is
/// the question a reader of this engine actually has: how much of a forward pass
/// is the stage convolution, how much is attention, how much is the five `lg_add`
/// launches. A call-site breakdown would say `conv3x3` six times over.
pub struct Profile {
    /// One row per launch, in launch order: the name as launched, the events that
    /// bracket it, and the grid/block it ran with (which is what makes two
    /// launches of the same kernel at different shapes tell themselves apart in
    /// the report).
    pending: RefCell<Vec<(String, Event, Event, (u32, u32, u32), (u32, u32, u32))>>,
    /// (row label, milliseconds, launches) - accumulated by `report`.
    times: RefCell<Vec<(String, f32, usize)>>,
}

thread_local! {
    static PROFILE: RefCell<Option<Profile>> = const { RefCell::new(None) };
}

impl Profile {
    /// The active profile, or `None` unless `SWIN2SR_PROFILE_GPU` is set. The
    /// environment is read once, so a profiled run differs from an unprofiled one
    /// only in that it records events.
    fn active() -> Option<&'static Profile> {
        thread_local! {
            static ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        }
        let on = ON.with(|c| match c.get() {
            Some(v) => v,
            None => {
                let v = std::env::var_os("SWIN2SR_PROFILE_GPU").is_some();
                c.set(Some(v));
                v
            }
        });
        if !on {
            return None;
        }
        PROFILE.with(|p| {
            if p.borrow().is_none() {
                *p.borrow_mut() = Some(Profile {
                    pending: RefCell::new(Vec::new()),
                    times: RefCell::new(Vec::new()),
                });
            }
        });
        // The profile is a thread-local that lives for the process and is only
        // ever replaced by another `active()` on the same thread, so handing out
        // a `'static` view of it here is what lets `run` hold it across a launch
        // without borrowing the thread-local for the whole call.
        PROFILE.with(|p| {
            let slot = p.borrow();
            let r: &Profile = slot.as_ref().expect("just installed");
            Some(unsafe { &*(r as *const Profile) })
        })
    }

    fn push(&self, name: &str, start: Event, end: Event, grid: (u32, u32, u32), block: (u32, u32, u32)) {
        self.pending.borrow_mut().push((name.to_string(), start, end, grid, block));
    }

    /// Read every pending elapsed time and fold it into the per-kernel table.
    fn drain(&self) {
        let pending = std::mem::take(&mut *self.pending.borrow_mut());
        let mut t = self.times.borrow_mut();
        for (name, start, end, grid, block) in pending {
            let ms = start.elapsed_ms(&end).unwrap_or(0.0);
            let label = format!("{name} [{},{},{}]x[{},{},{}]", grid.0, grid.1, grid.2,
                                block.0, block.1, block.2);
            match t.iter_mut().find(|(n, _, _)| *n == label) {
                Some(row) => {
                    row.1 += ms;
                    row.2 += 1;
                }
                None => t.push((label, ms, 1)),
            }
        }
    }

    /// Print the table, heaviest first, with each kernel's share of the device
    /// time. Called once, after the forward and after a sync.
    pub fn report(total_wall_ms: f32) {
        // The caller has synchronised by now (main.rs does it before calling
        // this), so every pending pair has both events complete and the elapsed
        // times are readable.
        let rows = PROFILE.with(|p| {
            p.borrow().as_ref().map(|pr| {
                pr.drain();
                let mut v = pr.times.borrow().clone();
                v.sort_by(|a, b| b.1.total_cmp(&a.1));
                v
            })
        });
        let Some(rows) = rows else { return };
        let sum: f32 = rows.iter().map(|r| r.1).sum();
        eprintln!("device profile: {} kernel kinds, {:.1} ms of device time, {:.1} ms wall",
                  rows.len(), sum, total_wall_ms);
        eprintln!("  {:>9}  {:>6}  {:>5}  {}", "ms", "share", "calls", "kernel");
        for (name, ms, calls) in rows {
            eprintln!("  {:>9.2}  {:>5.1}%  {:>5}  {}", ms, 100.0 * ms / sum.max(1e-6), calls, name);
        }
    }
}
