//! Swin2SR as a self-contained engine: classical and real-world image
//! super-resolution on a PNG, nothing else.
//!
//! The layers exist so the two backends can be read against each other:
//!
//! * [`plan`] - the padded geometry every stage shares (windows, tokens, shifts).
//! * [`cpu`] - the CPU backend: plain Rust, no device, one function per op
//!   (`--device cpu`). A fallback for a machine with no usable GPU, held to the
//!   same standard as the device backend and optimised to the same end - both are
//!   judged against the published PyTorch output by `--verify` and `tests/parity.rs`.
//! * [`gpu`] - the CUDA backend, built from the toolkit kernels plus
//!   `cuda/swin2sr.cu` (`--device gpu`).
//! * [`weights`] - the checkpoint, validated against the architecture it claims.
//! * [`image`] - PNG in, PNG out.
//!
//! Neither backend is the standard the other is judged by. The reference is the
//! published PyTorch implementation: both backends are required to land closer to
//! it than two of its own runs under different summation orders would. They are
//! not required to be bit-identical to each other, and `--cuda-selftest` exists
//! only to catch a kernel that does not do what the graph says - it cannot catch a
//! graph both backends transcribed wrongly, because both would be wrong together.
//!
//! `--cuda-selftest` is not a substitute for `tests/gpu.rs` either, and that gap
//! had teeth: the selftest compares each KERNEL against its CPU twin at shapes the
//! test chooses, and it passed a fused head kernel whose shared-memory staging had
//! two threads writing one address - because every shape it uses produces the same
//! numbers on both sides of the race. What caught it was running the whole GRAPH
//! on both backends at a width the golden fixtures never reach.

pub mod backend;
pub mod cpu;
pub mod fixture;
pub mod image;
pub mod plan;
pub mod weights;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod gpu;
