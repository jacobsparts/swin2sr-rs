//! The golden fixtures: input and expected output, as the reference produced them.
//!
//! `tools/make_fixture.py` writes these from the published PyTorch network, and
//! `--verify` / `tests/parity.rs` read them. The format is deliberately trivial (a
//! header of u32s and two f32 planes) so a fixture can be checked with `xxd`, and
//! the errors below are the ones a truncated or half-written file would produce.
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"SW2F";

/// `flags` bit 0: an aux plane follows `expected`.
///
/// The compressed_sr head returns TWO images - the restored one and a
/// lower-resolution `aux` at the PADDED plane's size - so the fixture has to be
/// able to carry the second one. A flag rather than a version bump because a
/// fixture without it is byte-for-byte what it always was, and the field it uses
/// was already in the header, written as zero and never read.
pub const FLAG_AUX: u32 = 1;

pub struct Fixture {
    pub version: u32,
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub scale: usize,
    pub win: usize,
    pub flags: u32,
    pub input: Vec<f32>,
    pub expected: Vec<f32>,
    /// The head's second output, at the PADDED plane's geometry, for checkpoints
    /// whose head produces one. `None` for every other head.
    pub aux: Option<Vec<f32>>,
}

impl Fixture {
    /// The padded plane the aux output lives on, `(h/win + 1) * win` per axis -
    /// the same padding `Plan::new` applies.
    pub fn aux_plane(&self) -> (usize, usize) {
        ((self.h / self.win + 1) * self.win, (self.w / self.win + 1) * self.win)
    }
}

fn u32le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn f32s(b: &[u8], off: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| f32::from_le_bytes([b[off + 4 * i], b[off + 4 * i + 1], b[off + 4 * i + 2], b[off + 4 * i + 3]])).collect()
}

impl Fixture {
    pub fn load(path: impl AsRef<Path>) -> Result<Fixture, String> {
        let path = path.as_ref();
        let b = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        if b.len() < 36 || &b[..4] != MAGIC {
            return Err(format!("{}: not a swin2sr fixture (want 4-byte magic SW2F)", path.display()));
        }
        let (version, h, w, c, scale, win, flags) =
            (u32le(&b, 4), u32le(&b, 8) as usize, u32le(&b, 12) as usize, u32le(&b, 16) as usize,
             u32le(&b, 20) as usize, u32le(&b, 24) as usize, u32le(&b, 28));
        if version != 1 {
            return Err(format!("{}: fixture version {version}, this engine reads 1", path.display()));
        }
        if flags & !FLAG_AUX != 0 {
            return Err(format!(
                "{}: unknown fixture flags {:#x} - this engine knows only bit 0 ({FLAG_AUX:#x}, \
                 an aux plane follows)",
                path.display(),
                flags
            ));
        }
        let (ah, aw) = if flags & FLAG_AUX != 0 {
            ((h / win + 1) * win, (w / win + 1) * win)
        } else {
            (0, 0)
        };
        let need = 36 + 4 * (h * w * c + h * scale * w * scale * c + ah * aw * c);
        if b.len() != need {
            return Err(format!(
                "{}: {need} bytes expected for {h}x{w}x{c} at scale {scale}{}, {} present",
                path.display(),
                if flags & FLAG_AUX != 0 { format!(" with a {ah}x{aw} aux plane") } else { String::new() },
                b.len()
            ));
        }
        let aux_at = 36 + 4 * (h * w * c + h * scale * w * scale * c);
        Ok(Fixture {
            version,
            h,
            w,
            c,
            scale,
            win,
            flags,
            input: f32s(&b, 36, h * w * c),
            expected: f32s(&b, 36 + 4 * h * w * c, h * scale * w * scale * c),
            aux: if flags & FLAG_AUX != 0 { Some(f32s(&b, aux_at, ah * aw * c)) } else { None },
        })
    }

    /// The worst absolute difference, and where it is - the two numbers a parity
    /// report needs, since "max diff 0.31" and "max diff 0.31 at one pixel in the
    /// top-left corner" call for different responses.
    pub fn compare(&self, got: &[f32]) -> (f32, usize, f32) {
        assert_eq!(got.len(), self.expected.len(), "backend returned the wrong number of pixels");
        let mut worst = 0.0f32;
        let mut at = 0usize;
        let mut mean = 0.0f64;
        for (i, (a, b)) in self.expected.iter().zip(got.iter()).enumerate() {
            let d = (a - b).abs();
            mean += d as f64;
            if d > worst {
                worst = d;
                at = i;
            }
        }
        (worst, at, (mean / got.len() as f64) as f32)
    }

    /// The value at `index` in the expected tensor, as (y, x, channel) - for the
    /// message that accompanies a failed comparison.
    pub fn locate(&self, idx: usize) -> (usize, usize, usize) {
        let (oh, ow) = (self.h * self.scale, self.w * self.scale);
        let plane = oh * ow;
        let c = idx / plane;
        let rem = idx % plane;
        (rem / ow, rem % ow, c)
    }
}
