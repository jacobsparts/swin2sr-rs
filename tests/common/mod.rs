//! Shared helpers for the integration tests.
//!
//! The converted checkpoints are ~85 MiB each and are NOT in the repository: they
//! are reproducible from the published `.pth` with `tools/convert.py` in a minute
//! (`./convert_all.sh` does the whole set). They live in the family's shared
//! `models/` directory next to the `.pth` files, like every other engine's, and
//! `SWIN2SR_MODELS` points the tests at a different one. A test that needs a
//! checkpoint SKIPS when it is missing rather than failing: a fresh clone has no
//! weights, and a test that cannot run is not a failing test.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use swin2sr::weights::Weights;

/// A file inside this engine's repository.
pub fn repo(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(name)
}

/// The directory the converted checkpoints live in: the family's shared `models/`
/// next to this engine, overridable for a checkout that keeps them elsewhere.
pub fn models_dir() -> PathBuf {
    match std::env::var_os("SWIN2SR_MODELS") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the repo's parent")
            .join("models"),
    }
}

/// The checkpoint for `task`, or `None` (with a message) when it has not been
/// converted. Every test that needs weights goes through this.
pub fn weights(task: &str) -> Option<Weights> {
    let candidate = models_dir().join(format!("swin2sr-{task}.safetensors"));
    if !candidate.exists() {
        eprintln!(
            "skipping: {} is not present. Create it with:\n    \
             python3 tools/convert.py /path/to/<checkpoint>.pth {}\n\
             (or run ./convert_all.sh, which converts every released checkpoint the \
             fixtures use)",
            candidate.display(),
            candidate.display()
        );
        return None;
    }
    Some(Weights::load(&candidate).expect("load the converted checkpoint"))
}
