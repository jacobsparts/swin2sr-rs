//! Compiles this engine's kernels into two modules: the shared `lightgpu`
//! toolkit's `cuda/kernels.cu` (TOOLKIT_KERNELS) and this project's own
//! `cuda/swin2sr.cu` (PROJECT_KERNELS). Each gets its own fatbin and its own
//! `--entries` list, and `src/cuda.rs` loads them as separate modules, so neither
//! can shadow a name in the other.
//!
//! Both lists are checked against the source they are compiled from before nvcc
//! runs, in BOTH directions: a name that is listed but not defined fails the
//! build, and so does a name that is defined but not listed - the latter would be
//! pruned from the fatbin and fail at launch instead.

/// The toolkit's kernels this engine launches. Everything else the graph does is
/// in PROJECT_KERNELS, which is the split lightgpu's promotion test asks for: a
/// kernel stays here only if another model family could call it unchanged.
const TOOLKIT_KERNELS: &[&str] = &[
    // conv_first, every stage's `conv`, conv_after_body, and the whole
    // reconstruction head: this model is mostly 3x3 convolutions.
    //
    // Both 3x3 kernels are here. `lg_conv3x3_winograd` is what the graph
    // launches (F(4,3): 36 products per 16 outputs against the direct kernel's
    // 144); `lg_conv3x3s1p1` is launched by nothing and is kept because
    // `--cuda-selftest` checks the winograd one against it - a transform this
    // engine got wrong is otherwise only visible as a slightly wrong image.
    //
    // `lg_conv1x1` and `lg_upsample2x_nearest` are here because the graph calls
    // them: this file used to carry its own `ss_conv1x1` and
    // `ss_upsample2x_nearest` for the same two ops.
    "lg_conv3x3s1p1",
    "lg_conv3x3_winograd",
    // The 1x1 patch_embed convs, at both levels: one thread per output element,
    // `c_in` FMAs each.
    "lg_conv1x1",
    // q, k, v and the attention output projection, plus the MLP's two layers:
    // all four are [tokens][c] x [c][c]. A 16x16 shared-memory tiled matmul; the
    // per-thread `ss_linear` this replaced was 60% of a forward and 10x slower.
    "lg_linear",
    // The real-world head's nearest-neighbour 2x upsample.
    "lg_upsample2x_nearest",
    // `self.norm` at the end of the body: nn.LayerNorm over the channel axis.
    "lg_channel_layer_norm",
    // The residual adds after each block and around the body and the stages.
    "lg_add",
    // The erf GELU in the MLP. nn.GELU's default, not the tanh approximation.
    "lg_gelu_erf",
    // The reconstruction head's activations.
    "lg_lrelu",
    "lg_copy",
];

/// This engine's own kernels, in `cuda/swin2sr.cu`.
///
/// The window attention is the architecture, not a generic op: the score/softmax/
/// apply fusion, the shift folded into the gather index, the mask computed from
/// coordinates, and the [N*N][heads] relative-position table are all this model's
/// terms. lightgpu's CONVENTIONS.md already records that the Swin window
/// gather/scatter pair lives in a consumer (`rmbg-rs`), for the same reason.
const PROJECT_KERNELS: &[&str] = &[
    "ss_window_gather",
    "ss_window_scatter",
    "ss_attention",
    "ss_pixel_shuffle2",
    "ss_conv3x3_shuffle2",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/swin2sr.cu");

    // `cargo build --no-default-features` is the pure-Rust CPU build: it must not
    // need nvcc, and `src/cuda.rs` (which includes the fatbins) is not compiled.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate lightgpu's cuda/kernels.cu (set LA_GPU_DIR to override)");
    let toolkit = toolkit.to_string_lossy().into_owned();

    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit - did it move into cuda/swin2sr.cu?"
        );
    }
    let src = std::fs::read_to_string("cuda/swin2sr.cu").expect("read cuda/swin2sr.cu");
    let defined = lightgpu_build::kernel_names_in(&src);
    for k in PROJECT_KERNELS {
        assert!(
            defined.iter().any(|d| d == k),
            "`{k}` is not defined in cuda/swin2sr.cu (it has {})",
            defined.join(", ")
        );
    }
    for d in &defined {
        assert!(
            PROJECT_KERNELS.contains(&d.as_str()),
            "cuda/swin2sr.cu defines `{d}`, which PROJECT_KERNELS does not list - \
             it would be pruned from the fatbin and fail at launch"
        );
    }

    lightgpu_build::fatbin_modules(&[
        lightgpu_build::Source {
            path: &toolkit,
            out_name: "swin2sr_toolkit.fatbin",
            entries: Some(TOOLKIT_KERNELS),
        },
        lightgpu_build::Source {
            path: "cuda/swin2sr.cu",
            out_name: "swin2sr_project.fatbin",
            entries: Some(PROJECT_KERNELS),
        },
    ]);
}
