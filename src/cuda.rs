//! The CUDA half of the engine: the two fatbins, the kernel-name sets, and the
//! launch plumbing.
//!
//! WHICH MODULE A KERNEL LIVES IN IS A PROPERTY OF THE NAME, and the two modules
//! are disjoint: `lightgpu/cuda/kernels.cu` (toolkit) and `cuda/swin2sr.cu` (this
//! engine). `module_of` asks the project module first and then the toolkit, so a
//! kernel promoted from one to the other is a one-line change here. A name in
//! NEITHER fails at construction, where the message can say which of the two
//! lists is wrong.
use lightgpu::vm::{Args, DevBuf, Launch, Module};

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
    "lg_linear",
    "lg_upsample2x_nearest",
    "lg_channel_layer_norm",
    "lg_add",
    "lg_gelu_erf",
    "lg_lrelu",
    "lg_copy",
];

/// This engine's own kernels, from `cuda/swin2sr.cu`. The same four names
/// `build.rs` validates against the source, in both directions.
pub const PROJECT_KERNELS: &[&str] = &[
    "ss_window_gather",
    "ss_window_scatter",
    "ss_attention",
    "ss_pixel_shuffle2",
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
        args.launch(self.module_of(name), name, launch)
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
