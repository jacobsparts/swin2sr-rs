//! The checkpoint: an mmap'd `.safetensors` plus the architecture it describes.
//!
//! THE FILE SAYS WHAT IT IS. The published Swin2SR checkpoints are
//! indistinguishable by tensor names - classical and real-world SR share every
//! layer name and differ only in the reconstruction head - so `tools/convert.py`
//! writes the architecture into the container's `__metadata__` and this module
//! reads it back and validates it against the tensors that are actually present.
//! A checkpoint converted for a different task fails here, with a message that
//! names the field that disagrees, instead of at the first misshapen matmul.
//!
//! Tensors are handed out as borrowed `&[f32]` slices of the mapping: the engine
//! never copies a weight, on either backend (`Gpu` uploads them once).
use std::collections::BTreeMap;
use std::path::Path;

use lightgpu::safetensors;

/// Which reconstruction head the checkpoint ends with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Upsampler {
    /// `Conv2d(feat, 4*feat, 3x3) + PixelShuffle(2)`, once per octave. Classical SR.
    PixelShuffle,
    /// One `Conv2d(feat, 4*out, 3x3) + PixelShuffle(scale)`. Lightweight SR.
    PixelShuffleDirect,
    /// Nearest-neighbour x2 + conv, twice. Real-world SR.
    NearestConv,
    /// Classical, with a second head that also emits the pre-upsample image. Compressed SR.
    PixelShuffleAux,
}

impl Upsampler {
    fn parse(s: &str) -> Result<Upsampler, String> {
        Ok(match s {
            "pixelshuffle" => Upsampler::PixelShuffle,
            "pixelshuffledirect" => Upsampler::PixelShuffleDirect,
            "nearest+conv" => Upsampler::NearestConv,
            "pixelshuffle_aux" => Upsampler::PixelShuffleAux,
            other => return Err(format!(
                "upsampler `{other}` is not one this engine implements (pixelshuffle, \
                 pixelshuffledirect, nearest+conv, pixelshuffle_aux)"
            )),
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            Upsampler::PixelShuffle => "pixelshuffle",
            Upsampler::PixelShuffleDirect => "pixelshuffledirect",
            Upsampler::NearestConv => "nearest+conv",
            Upsampler::PixelShuffleAux => "pixelshuffle_aux",
        }
    }

    /// The reconstruction convs are applied `scale/2` times for the pixel-shuffle
    /// head and once for the direct one; nearest+conv always runs twice.
    pub fn octaves(&self, scale: usize) -> usize {
        match self {
            Upsampler::PixelShuffle | Upsampler::PixelShuffleAux => {
                (scale as f32).log2().round() as usize
            }
            Upsampler::PixelShuffleDirect => 1,
            Upsampler::NearestConv => 2,
        }
    }
}

/// The processed image the network sees: `(x - mean) * img_range`.
pub const RGB_MEAN: [f32; 3] = [0.4488, 0.4371, 0.4040];

pub struct Weights {
    file: safetensors::File,
    pub task: String,
    pub upsampler: Upsampler,
    pub scale: usize,
    pub window: usize,
    pub embed: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub mlp_ratio: usize,
    /// Blocks per RSTB stage, one entry per stage.
    pub depths: Vec<usize>,
    pub img_range: f32,
    pub mean: [f32; 3],
    /// Every tensor name in the file, in the file's own order (for `--list-weights`).
    pub names: Vec<String>,
    pub bytes: u64,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Weights, String> {
        let path = path.as_ref();
        let file = safetensors::File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let mut meta: BTreeMap<String, String> = BTreeMap::new();
        for k in ["task", "upsampler", "scale", "window_size", "embed_dim", "num_heads",
                  "mlp_ratio", "depths", "img_range", "rgb_mean"] {
            let v = file
                .metadata_get(k)
                .ok_or_else(|| format!(
                    "{}: no `{k}` in __metadata__ - convert the checkpoint with tools/convert.py, \
                     which records the architecture; a plain state_dict cannot be checked",
                    path.display()))?;
            meta.insert(k.to_string(), v.to_string());
        }
        let num = |k: &str| -> Result<usize, String> {
            meta[k].parse().map_err(|e| format!("{}: `{}` = {:?}: {}", path.display(), k, meta[k], e))
        };
        let depths = meta["depths"]
            .split(',')
            .map(|s| s.trim().parse::<usize>().map_err(|e| format!("depths: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        let mean: Vec<f32> = meta["rgb_mean"]
            .split(',')
            .map(|s| s.trim().parse::<f32>().map_err(|e| format!("rgb_mean: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        if mean.len() != 3 {
            return Err(format!("rgb_mean has {} entries, expected 3", mean.len()));
        }

        let w = Weights {
            task: meta["task"].clone(),
            upsampler: Upsampler::parse(&meta["upsampler"])?,
            scale: num("scale")?,
            window: num("window_size")?,
            embed: num("embed_dim")?,
            heads: num("num_heads")?,
            head_dim: num("embed_dim")? / num("num_heads")?,
            mlp_ratio: num("mlp_ratio")?,
            depths,
            img_range: meta["img_range"].parse().map_err(|e| format!("img_range: {e}"))?,
            mean: [mean[0], mean[1], mean[2]],
            names: file.order().to_vec(),
            bytes: file.len() as u64,
            file,
        };
        w.validate(path)?;
        Ok(w)
    }

    /// Check the file against the architecture it claims. These are the mistakes
    /// that otherwise surface as a wrong image rather than an error: a checkpoint
    /// paired with the wrong task, a stale conversion, a hand-edited header.
    fn validate(&self, path: &Path) -> Result<(), String> {
        let embed = self.embed;
        let mut want: Vec<(String, Vec<usize>)> = vec![
            ("conv_first.weight".into(), vec![embed, 3, 3, 3]),
            ("conv_after_body.weight".into(), vec![embed, embed, 3, 3]),
            ("norm.weight".into(), vec![embed]),
            ("patch_embed.proj.weight".into(), vec![embed, embed, 1, 1]),
            ("patch_embed.norm.weight".into(), vec![embed]),
        ];
        // The two pixel-shuffle heads end in `conv_last`; the direct one's last
        // layer is the single conv that produces all 3*scale^2 planes at once, so
        // asking for `conv_last` would reject a perfectly good checkpoint. Which
        // head is in the file is in the header, and this is where that matters.
        if self.upsampler != Upsampler::PixelShuffleDirect {
            want.push(("conv_last.bias".into(), vec![3]));
        }
        for (name, shape) in want {
            let got = self
                .file
                .shape(&name)
                .map_err(|e| format!("{}: {name}: {e}", path.display()))?
                .to_vec();
            if got != shape {
                return Err(format!(
                    "{}: {name} is {got:?}, the {}-stage/{embed}-wide architecture in the header \
                     implies {shape:?}",
                    path.display(),
                    self.depths.len()
                ));
            }
        }
        for (l, b) in self.depths.iter().enumerate().flat_map(|(l, d)| (0..*d).map(move |b| (l, b))) {
            let p = format!("layers.{l}.residual_group.blocks.{b}");
            for (name, shape) in [
                (format!("{p}.attn.qkv.wq"), vec![embed, embed]),
                (format!("{p}.attn.qkv.wv"), vec![embed, embed]),
                (format!("{p}.attn.proj.weight"), vec![embed, embed]),
                (format!("{p}.mlp.fc1.weight"), vec![self.mlp_ratio * embed, embed]),
                (format!("{p}.mlp.fc2.weight"), vec![embed, self.mlp_ratio * embed]),
                (format!("{p}.attn.cpb_pre"), vec![self.window * self.window * self.window * self.window, self.heads]),
            ] {
                let got = self
                    .file
                    .shape(&name)
                    .map_err(|e| format!("{}: {name}: {e}", path.display()))?
                    .to_vec();
                if got != shape {
                    return Err(format!("{}: {name} is {got:?}, expected {shape:?}", path.display()));
                }
            }
        }
        // The compressed head's three extra convs, which nothing else in the file
        // implies: the bicubic pre-upsample conv (3 -> 64), the auxiliary
        // reconstruction (64 -> 3) and the conv that lifts the auxiliary image
        // back to `feat` before it is added to the octave output.
        if self.upsampler == Upsampler::PixelShuffleAux {
            let feat = self
                .file
                .shape("conv_before_upsample.0.weight")
                .map_err(|e| format!("{}: conv_before_upsample.0.weight: {e}", path.display()))?[0];
            for (name, shape) in [
                ("conv_bicubic.weight".to_string(), vec![feat, 3, 3, 3]),
                ("conv_bicubic.bias".to_string(), vec![feat]),
                ("conv_aux.weight".to_string(), vec![3, feat, 3, 3]),
                ("conv_aux.bias".to_string(), vec![3]),
                ("conv_after_aux.0.weight".to_string(), vec![feat, 3, 3, 3]),
                ("conv_after_aux.0.bias".to_string(), vec![feat]),
            ] {
                let got = self
                    .file
                    .shape(&name)
                    .map_err(|e| format!("{}: {name}: {e}", path.display()))?
                    .to_vec();
                if got != shape {
                    return Err(format!(
                        "{}: {name} is {got:?}, the compressed head implies {shape:?}",
                        path.display()
                    ));
                }
            }
        }
        if self.upsampler == Upsampler::PixelShuffleDirect {
            let out = self.scale * self.scale * 3;
            let got = self.file.shape("upsample.0.weight").map_err(|e| e.to_string())?.to_vec();
            if got != vec![out, self.embed, 3, 3] {
                return Err(format!(
                    "{}: upsampler pixelshuffledirect implies upsample.0.weight {out}x{}x3x3, found {got:?}",
                    path.display(),
                    self.embed
                ));
            }
        }
        Ok(())
    }

    /// A weight tensor. Every name this engine asks for is validated at load, so a
    /// miss here is a programming error, not bad input.
    #[inline]
    pub fn t(&self, name: &str) -> &[f32] {
        self.file.f32(name).unwrap_or_else(|e| panic!("weight `{name}`: {e}"))
    }

    /// A tensor's shape, for the code that has to budget device memory rather
    /// than read values (`gpu::uploaded_bytes`). Every name asked about here was
    /// validated at load, so a miss is a programming error.
    #[inline]
    pub fn shape(&self, name: &str) -> &[usize] {
        self.file.shape(name).unwrap_or_else(|e| panic!("weight `{name}`: {e}"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.file.contains(name)
    }

    /// The number of multiply-adds the 3x3 convs and matmuls cost at `h`x`w`,
    /// for the run header. Attention is counted separately (`flops_attention`).
    pub fn flops_conv(&self, h: usize, w: usize, scale: usize) -> u64 {
        let hw = (h * w) as u64;
        let c = self.embed as u64;
        let blocks: usize = self.depths.iter().sum();
        let mut f = hw * 3 * c * 9;                    // conv_first
        f += hw * c * c * 9 * blocks as u64;           // each block's RSTB 3x3
        f += hw * c * c * blocks as u64;               // ...and its 1x1 patch_embed
        f += hw * c * c * 9;                           // conv_after_body
        for _ in 0..blocks {
            f += hw * c * c * 3;                       // qkv (one fused [3C][C])
            f += hw * c * c;                           // attn proj
            f += 2 * hw * c * (self.mlp_ratio as u64) * c;
        }
        // The reconstruction head, at the pre-upsampler resolution per octave.
        let feat = 64u64;
        match self.upsampler {
            Upsampler::PixelShuffle => {
                f += hw * c * feat * 9;
                let mut cur = hw;
                for _ in 0..self.upsampler.octaves(scale) {
                    f += cur * feat * 4 * feat * 9;
                    cur *= 4;
                }
                f += cur * feat * 3 * 9;
            }
            Upsampler::PixelShuffleDirect => {
                f += hw * c * 3 * (scale as u64).pow(2) * 9;
            }
            // The compressed head is the pixel-shuffle head plus three small
            // convs; counting only the shared part would understate the cost of
            // the very head whose bicubic pre-upsample is the extra work.
            Upsampler::PixelShuffleAux => {
                f += hw * c * feat * 9;
                let mut cur = hw;
                for _ in 0..self.upsampler.octaves(scale) {
                    f += cur * feat * 4 * feat * 9;
                    cur *= 4;
                }
                f += cur * feat * 3 * 9;
                // conv_bicubic runs on the OUTPUT grid (3 -> feat), conv_aux and
                // conv_after_aux on the padded plane.
                f += cur * 3 * feat * 9;
                f += hw * feat * 3 * 9;
                f += hw * 3 * feat * 9;
            }
            Upsampler::NearestConv => {}
        }
        f
    }

    /// Multiply-adds in window attention: `q.k^T` and `attn.v` per block.
    pub fn flops_attention(&self, h: usize, w: usize) -> u64 {
        let tokens = ((h + self.window - 1) / self.window * self.window) as u64
            * ((w + self.window - 1) / self.window * self.window) as u64;
        let win = (self.window * self.window) as u64;
        let c = self.embed as u64;
        let blocks: usize = self.depths.iter().sum();
        blocks as u64 * (tokens * win * c * 2 + tokens * win * c)
    }
}
