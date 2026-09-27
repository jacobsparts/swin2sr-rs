//! The geometry of one forward pass, computed once and shared by both backends.
//!
//! Everything here follows the reference's `main_test_swin2sr.py`, including the
//! part that surprises people: the input is padded to the NEXT window multiple
//! (`(h / win + 1) * win`, never `ceil(h / win) * win`), which for an exact
//! multiple adds a FULL window row and column - 1080 rows become 1088 - and then
//! by reflecting the whole plane rather than its edge rows, so the added columns
//! are a mirror of the image's FIRST 8 columns, not a repeat of its last ones.
//! Both are load-bearing: the crop at the end hides the geometry, but the numbers
//! inside it depend on every bit of it.
//!
//! The window shift is the other half. A shifted block is fed `torch.roll(x, -shift)`,
//! so a window's token (i, j) reads the image at `(wh*win + i + shift) % hp` - and
//! the attention mask is built from the position in the ROLLED plane (`wh*win + i`),
//! not from the image coordinate the token reads. `Plan::index` and `Plan::region`
//! are those two different mappings, and keeping them apart is what makes shifted
//! windows correct.
/// The padded geometry of one image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The input image, before padding.
    pub h: usize,
    pub w: usize,
    /// The padded plane every stage runs on. Both are multiples of `win`.
    pub hp: usize,
    pub wp: usize,
    pub win: usize,
    /// Windows along each axis, and their product.
    pub nwh: usize,
    pub nww: usize,
    pub nw: usize,
    /// Tokens in one window, `win * win`.
    pub n: usize,
    /// One token, `c` floats.
    pub c: usize,
}

impl Plan {
    pub fn new(h: usize, w: usize, win: usize, c: usize) -> Plan {
        // The reference's padding, exactly: enough to reach the next multiple of
        // the window size, which is at least one whole window in each direction.
        let hp = (h / win + 1) * win;
        let wp = (w / win + 1) * win;
        let nwh = hp / win;
        let nww = wp / win;
        Plan { h, w, hp, wp, win, nwh, nww, nw: nwh * nww, n: win * win, c }
    }

    pub fn tokens(&self) -> usize {
        self.nw * self.n
    }

    pub fn plane(&self) -> usize {
        self.hp * self.wp
    }

    /// The default shift for block `b`: the reference alternates `win/2` for odd
    /// blocks and 0 for even ones (`SwinTransformerBlock`'s `shift_size`).
    pub fn shift_for(&self, b: usize) -> usize {
        if b % 2 == 1 {
            self.win / 2
        } else {
            0
        }
    }

    /// The image coordinate a window token reads, for a block shifted by `shift`.
    ///
    /// `wi` is the flat window index (`wh * nww + ww`), `t` the flat token index
    /// (`i * win + j`) in the store's window-major layout.
    #[inline]
    pub fn index(&self, wi: usize, t: usize, shift: usize) -> (usize, usize) {
        let (wh, ww) = (wi / self.nww, wi % self.nww);
        let (i, j) = (t / self.win, t % self.win);
        let a = (wh * self.win + i + shift) % self.hp;
        let b = (ww * self.win + j + shift) % self.wp;
        (a, b)
    }

    /// The region (0, 1 or 2 per axis) a window token belongs to, for the shift
    /// mask. Note the UNWRAPPED coordinate: the mask is defined on the plane as it
    /// is rolled, before the modulo above.
    #[inline]
    pub fn region(&self, wi: usize, t: usize, shift: usize) -> (u8, u8) {
        let (wh, ww) = (wi / self.nww, wi % self.nww);
        let (i, j) = (t / self.win, t % self.win);
        (region_of(wh * self.win + i, self.hp, self.win, shift),
         region_of(ww * self.win + j, self.wp, self.win, shift))
    }

    /// Flat token index of (window, token) in the store's window-major layout.
    #[inline]
    pub fn token(&self, wi: usize, t: usize) -> usize {
        wi * self.n + t
    }
}

/// The reference's `h_slices`: `[0, -win)`, `[-win, -shift)`, `[-shift, None)`.
#[inline]
pub fn region_of(a: usize, p: usize, win: usize, shift: usize) -> u8 {
    if a + win < p {
        0
    } else if a + shift < p {
        1
    } else {
        2
    }
}

/// Padding by reflection, in the reference's order and its two steps.
///
/// `torch.cat([x, flip(x)], dim)[.., :h + pad]`: the first `h` entries are the
/// image, and entries `h..h+pad` are `x[h + pad - 1]`, `x[h + pad - 2]`, ...: a
/// MIRROR of the image's own rows read backwards. Input and output are
/// `[c][h][w]` planes.
pub fn pad_reflect(x: &[f32], c: usize, h: usize, w: usize, hp: usize, wp: usize) -> Vec<f32> {
    // Rows first, on the full width, then columns on the padded height: this is
    // the order the reference's two `torch.cat` calls impose.
    let mut rows = vec![0.0f32; c * hp * w];
    for ch in 0..c {
        for y in 0..hp {
            // The reflection is of the whole plane, so row index is `hp - 1 - y`
            // while that stays inside the image, and the image's first row after.
            let sy = if y < h { y } else { 2 * h - 1 - y.min(2 * h - 1) };
            rows[(ch * hp + y) * w..(ch * hp + y) * w + w]
                .copy_from_slice(&x[(ch * h + sy) * w..(ch * h + sy) * w + w]);
        }
    }
    let mut out = vec![0.0f32; c * hp * wp];
    for ch in 0..c {
        for y in 0..hp {
            for bx in 0..wp {
                let sx = if bx < w { bx } else { 2 * w - 1 - bx.min(2 * w - 1) };
                out[(ch * hp + y) * wp + bx] = rows[(ch * hp + y) * w + sx];
            }
        }
    }
    out
}

#[derive(Debug)]
pub enum PlanError {
    TooSmall { h: usize, w: usize, win: usize },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::TooSmall { h, w, win } => write!(
                f, "{h}x{w} is smaller than one {win}x{win} window. The reference pads to the NEXT \
                   window multiple by concatenating the image with its own mirror, which for an \
                   image shorter than a window truncates the pad and then pads again with \
                   F.pad(reflect) - a second, different rule this engine does not implement. \
                   Images at least one window wide in each direction do not reach it."
            ),
        }
    }
}
