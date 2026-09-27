# swin2sr

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).
The family also includes [rmbg-rs](https://github.com/jacobsparts/rmbg-rs),
[realesrgan-rs](https://github.com/jacobsparts/realesrgan-rs),
[nafnet-rs](https://github.com/jacobsparts/nafnet-rs),
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs),
[locate-anything-rs](https://github.com/jacobsparts/locate-anything-rs),
[maxim-rs](https://github.com/jacobsparts/maxim-rs),
[scunet-rs](https://github.com/jacobsparts/scunet-rs) and
[ifan-rs](https://github.com/jacobsparts/ifan-rs), all built on the
[lightgpu toolkit](https://github.com/jacobsparts/lightgpu).

[Swin2SR](https://github.com/mv-lab/swin2sr) image restoration - classical and
real-world super-resolution - as a single self-contained binary. Feed it a PNG,
get back a larger PNG. No Python, PyTorch, ONNX Runtime, or CUDA toolkit needed
at runtime.

```
swin2sr -m swin2sr-classical-x4.safetensors -i small.png -o large.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. `--device cpu|gpu` picks one; there is no silent
  fallback, so a GPU run that cannot start says so instead of quietly taking
  minutes.
* 1.89 MB binary with the CUDA backend, 1.23 MB without. The direct
  dependencies are `png`, `rayon` and the toolkit, and the whole CPU-only tree is
  20 crates including this one (`cargo tree --no-default-features`).
* All four released checkpoints, converted from the official `.pth` files:
  classical x2 and x4, real-world x4, and the lightweight x2. See Choosing a
  checkpoint below.
* **Byte-level agreement with the published network.** On a seeded input, every
  checkpoint reproduces PyTorch to within 1-8e-6 of 1.0 - two to three orders of
  magnitude inside the 8-bit level - on both backends.
* Tiling for images that do not fit in VRAM, with the fidelity cost measured
  rather than hidden (see Tiling below).

## Why this exists

The reference Swin2SR implementation is a Python program. Getting a
super-resolved image out of it means a Python environment, a PyTorch install
matched to the right CUDA, and a `.pth` pickle that only `torch.load` can read.

This is the same network, rewritten as an engine:

* **One file to ship.** A 1.89 MB binary and one `.safetensors` checkpoint. The
  converter needs `torch` and `numpy`; the engine needs neither.
* **The checkpoint is a data file, not a pickle.** It is converted once into the
  standard `.safetensors` container and read straight out of the mapping by the
  shared `lightgpu` reader - no deserialisation, no per-run parse. The
  architecture constants live in the file header and every tensor is
  shape-checked against them at load, so a checkpoint that does not match its own
  description is refused rather than run.
* **Composes in a pipeline.** Input and output default to stdin and stdout, and
  `-` names either stream, so
  `swin2sr -m model.safetensors -i in.png | display -` works. All progress goes
  to stderr.
* **Tiling that reports what it costs.** `--tile` makes an image that does not
  fit run anyway, and the README carries the measured difference from an
  untiled run rather than the word "approximate".
* **An op set with a boundary.** The general kernels - the winograd and direct
  3x3 convolutions, the 1x1 convolution, the tiled linear, nearest-neighbour
  upsampling, channel LayerNorm, erf GELU, LeakyReLU, residual add, copy - are the
  [lightgpu](https://github.com/jacobsparts/lightgpu) toolkit's, and the graph
  calls the toolkit for every one of them. This engine keeps four kernels of its
  own: the two window index maps, the attention, and the head's pixel shuffle -
  what the window-versus-image duality and Swin's shifted-window score/softmax/
  apply with a relative-position bias table force onto a consumer, and what the
  toolkit has no CPU twin or forward form for. `build.rs` checks both lists
  against the source in both directions, so a kernel that is defined but not
  listed fails the build instead of disappearing from the fatbin.

## Build

```sh
cargo build --release
# CPU only, no CUDA toolkit or driver needed at build time either:
cargo build --release --no-default-features
```

The default build needs `nvcc` (set `NVCC=` if it is not on `PATH`) and produces
one binary with both backends. The kernels are compiled for `sm_61`, `sm_75`,
`sm_80` and compute capability 8.0 PTX, so the GPU path runs on Pascal (GTX
10-series) through Ampere, and on anything newer through the PTX.

`lightgpu` is a normal Cargo dependency on
[its repository](https://github.com/jacobsparts/lightgpu), so a clone of this
project builds on its own.

## Get the weights

The converted checkpoints are **not** in the repository: each is 19-82 MiB. They
are attached to the [releases](https://github.com/jacobsparts/swin2sr-rs/releases)
(see Download above), so downloading one from there is enough for a restored
image. The tests skip (with a message naming the script below) until a checkpoint
is put in the family's `models/` directory or one is built here.

To build them from the official `.pth` files instead:

```sh
./convert_all.sh
```

which downloads the four upstream checkpoints if they are not already in the
family's `models/` directory, converts them, and regenerates the golden fixtures
in `tests/data`. The tests skip (with a message naming this script) until it has
been run; nothing else needs it.

The released `.pth` files are these, from the upstream
[releases](https://github.com/mv-lab/swin2sr/releases/tag/v0.0.1):

| checkpoint | upstream file | converted size |
|---|---|---|
| `swin2sr-classical-x4` | `Swin2SR_ClassicalSR_X4_64.pth` | 81.6 MiB |
| `swin2sr-classical-x2` | `Swin2SR_ClassicalSR_X2_64.pth` | 81.0 MiB |
| `swin2sr-realworld-x4` | `Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth` | 80.9 MiB |
| `swin2sr-lightweight-x2` | `Swin2SR_Lightweight_X2_64.pth` | 19.2 MiB |

All four are attached to the [releases](https://github.com/jacobsparts/swin2sr-rs/releases)
alongside the binaries, so there is no need to run the script unless you want to
build them yourself.

The converted files remain subject to the upstream Apache-2.0 terms; see Licence
and attribution below.

## Convert a checkpoint

```sh
python3 tools/convert.py Swin2SR_ClassicalSR_X4_64.pth swin2sr-classical-x4.safetensors
```

The converter needs `torch` and `numpy`; the engine needs neither. It writes the
standard `.safetensors` container and records the architecture (`task`,
`upsampler`, `scale`, `window_size`, `embed_dim`, `num_heads`, `mlp_ratio`,
`depths`, `img_range`, `rgb_mean`) in the header, from which the engine takes its
geometry and validates every tensor shape.

It also does three things beyond copying, each of which removes work from the
engine's inner loop:

* **Fuses the attention QKV projection.** The reference builds `qkv` with one
  `Linear` and then applies the key-less `q_bias` / `v_bias` through a
  `torch.cat`. The converter splits the one `[3C][C]` matrix into
  `attn.qkv.wq/wk/wv` and keeps `q_bias` / `v_bias` separate, so the engine's
  three projections are three `lg_linear` calls with no copy between them.
* **Folds the first row of `relative_coords_table` out of `cpb_mlp`.** That
  table's second column is a constant 1.0, so the `Linear(2, 512)` is
  `W[:,0] * table[:,0] + W[:,1]`. The converter precomputes the resulting
  `[N*N][heads]` bias as `{block}.attn.cpb_pre`, which is what the attention
  then reads - one float per (i, j, head) instead of a small matmul, a gather and
  a sigmoid per block per call.
* **Infers the head from the tensor names, not from a flag.** The released
  checkpoint filenames are not a reliable description of the architecture, and a
  wrong guess used to be invisible: the weight shapes still load and the image
  still looks plausible. The one that used to slip through is the compressed
  model, which shares the whole classical head and adds a bicubic branch beside
  it - so the extra branch is checked for first, and a checkpoint that has it is
  labelled `pixelshuffle_aux` and refused by the engine rather than run through
  the classical head.

`--task` and `--scale` are still accepted and are used for the fixture header;
the tensors win where they disagree, and the disagreement is printed.

## Download

Prebuilt binaries and all four converted checkpoints are attached to the
[releases](https://github.com/jacobsparts/swin2sr-rs/releases) as plain files,
with no archive: download what you need and run it.

| asset | contents | notes |
|---|---|---|
| `swin2sr-linux-x86_64` | CPU + CUDA | x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+); the GPU path needs a compute capability 6.1+ GPU, `--device cpu` runs the pure-Rust path anywhere |
| `swin2sr-linux-x86_64-cpu-only` | CPU only | same build with the CUDA feature off, so it uses the CPU and `--device gpu` is refused with a reason rather than falling back |
| the four `swin2sr-*.safetensors` | the converted checkpoints | 19-82 MiB each; see Choosing a checkpoint |

```sh
chmod +x swin2sr-linux-x86_64
./swin2sr-linux-x86_64 -m swin2sr-realworld-x4.safetensors -i in.png -o out.png
```

The `chmod` is not decoration: a download does not carry the executable
bit through, and a binary that has lost it fails with `Permission denied`
before it can print anything.

## Run

```sh
swin2sr -m swin2sr-classical-x4.safetensors -i in.png -o out.png
swin2sr -m swin2sr-realworld-x4.safetensors -i in.png -o out.png --device cpu
swin2sr -m swin2sr-lightweight-x2.safetensors -i in.png -o out.png --tile 64 --tile-pad 32
```

| flag | meaning |
| --- | --- |
| `-m, --model <path>` | converted `.safetensors` checkpoint (required) |
| `-i, --input <path>` | input PNG, or `-` (default: stdin) |
| `-o, --output <path>` | output PNG, or `-` (default: stdout) |
| `--device cpu\|gpu` | which backend to use (default: gpu when built with the `cuda` feature, else cpu) |
| `--tile <n\|auto>` | process in tiles of n pixels a side; `0` (default) is one whole-image pass |
| `--tile-pad <n>` | context kept around each tile, rounded up to a window multiple (default 32) |
| `--verify <fixture>` | run a golden fixture instead of an image and report the worst difference against the reference |
| `--tol <f>` | the tolerance `--verify` holds to (default 2e-3) |
| `-q, --quiet` | no progress output |
| `--cuda-selftest` | compare every CUDA kernel against its CPU twin and exit |
| `--self-test` | run the CPU op library's internal checks and exit |
| `--list-weights` | print the checkpoint's architecture and tensor names |
| `--info` | print what this binary and this machine can do |
| `-h, --help`, `-V, --version` | usage, version |

**`--device` does not fall back.** If the binary was built with the CUDA feature
it uses the GPU by default, and a GPU that cannot be brought up is an error, not
a quieter path. On a machine with no CUDA driver, build with
`--no-default-features` and the binary has the CPU path only.

Development builds (`cargo build --release --features dev`) add `--raw <h>x<w>`,
`--raw-out` and `--dump`, which run the network on a generated plane of floats
and write every intermediate activation. That is how a divergence is located by
stage rather than by bisecting an image, and it is how `tools/compare.py` is
driven. A release binary neither accepts nor advertises them.

## Choosing a checkpoint

The file decides the task and the scale; there is no flag for either.

| checkpoint | task | scale | params | what it is for |
|---|---|---|---|---|
| `swin2sr-classical-x4` | `classical_sr` | x4 | 12.2 M | bicubic-degraded input, the usual benchmark setting |
| `swin2sr-classical-x2` | `classical_sr_x2` | x2 | 12.1 M | as above, x2 |
| `swin2sr-realworld-x4` | `real_sr` | x4 | 12.0 M | **photographs.** Trained on BSRGAN degradations, so it handles JPEG and real sensor damage |
| `swin2sr-lightweight-x2` | `lightweight_sr` | x2 | 1.0 M | a 60-wide, 4-stage model: ~4x smaller and ~10x faster, visibly softer |

Use `realworld-x4` on a photograph and `classical-x4` on an image that was
cleanly downsampled. The classical models assume their input really is a bicubic
downsample; a JPEG's blocking is off their training distribution. The upstream
`compressed_sr` checkpoint is **not** supported: its head reconstructs a
low-resolution auxiliary image through a bicubic branch this engine does not
implement, and it is refused with that reason rather than approximated.

## Accuracy

Every checkpoint is checked against the *published* network -
`tools/network_swin2sr.py`, copied verbatim from upstream - by running both on
the same seeded 37x29 input and comparing the full output plane. A CPU and a GPU
backend that agree with each other prove only that they share a mistake, so the
reference is the printed network and never this engine.

| checkpoint | backend | max abs diff | mean abs diff | against a 2e-3 tolerance |
|---|---|---|---|---|
| classical x4 | CPU | 3.0e-6 | 5.1e-7 | pass |
| classical x4 | CUDA | 8.0e-6 | 9.0e-7 | pass |
| classical x2 | CPU | 3.0e-6 | 5.5e-7 | pass |
| classical x2 | CUDA | 6.0e-6 | 9.8e-7 | pass |
| real-world x4 | CPU | 2.0e-6 | 3.2e-7 | pass |
| real-world x4 | CUDA | 6.0e-6 | 5.5e-7 | pass |
| lightweight x2 | CPU | 1.0e-6 | 2.3e-7 | pass |
| lightweight x2 | CUDA | 5.0e-6 | 4.5e-7 | pass |

The differences are f32 rounding, not an algorithmic gap: two independent f32
evaluations of a 36-block network - torch's blocked matmuls and this engine's
loops - cannot agree to the last bit, and every 3x3 convolution in the residual
path amplifies what the one before it left behind. The numbers to read them
against: an exact transcription lands at ~1e-6 on the final image, and any single
wrong tap, transposed layout or missing skip lands at 1e-2 or worse. `--tol`
defaults to 2e-3, an order of magnitude either side of that gap.

Reproduce any row with:

```sh
$ swin2sr -m swin2sr-classical-x4.safetensors --verify tests/data/classical_x4.bin --device cpu
cpu classical_sr: 29x37 -> 116x148, 0.10s
  max |diff| 0.000003 at (x91, y64, c2)  mean |diff| 0.00000051  tol 2.0e-3
  PASS
```

Those are f32 numbers. The end-to-end claim is a PNG: take the same 8-bit input
the reference would read, run both through the network, and compare the files.
Every one of the mismatching pixels is off by exactly one 8-bit level, and there
are a handful of them:

| checkpoint | pixels differing from the reference, of the total | worst |
|---|---|---|
| classical x4 | 2 / 51504 CPU, 7 / 51504 CUDA | 1 level |
| classical x2 | 0 / 12876 CPU, 1 / 12876 CUDA | 1 level |
| real-world x4 | 3 / 51504 CPU, 8 / 51504 CUDA | 1 level |
| lightweight x2 | 0 / 12876 CPU, 0 / 12876 CUDA | none |

They sit exactly where the tolerance predicts: a value has to land within
`tol * 255` = 0.001 of an 8-bit rounding boundary to flip, and 0.06-0.22% of the
values in these outputs are that close to one.

`tools/compare.py` goes further and diffs the two implementations stage by
stage, against a `--dump` from a development build, naming each stage after its
torch module path. It is how the transcription errors in this engine were found
rather than guessed at. Two stages remain above the tolerance and both are
explained: the last stage of the body sits at 3.9e-3 from f32 accumulation over
36 blocks, which is 2.7e-6 in the final image, and `conv_after_body` differs from
a per-`Conv2d` hook because the body's residual skip is inside that module.

## Tiling

`--tile` exists so an image whose activations do not fit in memory can still be
processed. Each tile is cropped from the input with `--tile-pad` pixels of
context on every side, the network runs on that crop, and only the tile's own
region is pasted into the output; at the image border the crop shrinks rather
than extending past the edge.

**Tiling is a memory tradeoff, not an exact mode**, and the cost is not a small
constant: a tile boundary cuts the network's receptive field, and Swin2SR's is
large - 6 stages of 6 windows-attention blocks, with the shifted blocks reaching
a whole window sideways.

Measured on a 120x160 input with `classical-x4`, against the same run untiled,
on 8-bit output levels. The CPU and CUDA figures agree to the last digit:

| tile | pad | max diff | mean diff | pixels off by >2 levels |
|---|---|---|---|---|
| 32 | 8 | 35 | 0.8744 | 10.28% |
| 32 | 16 | 11 | 0.1740 | 0.15% |
| 32 | 32 | 1 | 0.0042 | 0.00% |
| 64 | 8 | 34 | 0.4090 | 4.33% |
| 64 | 16 | 11 | 0.0797 | 0.05% |
| 64 | 32 | 1 | 0.0021 | 0.00% |
| 64 | 64 | 1 | 0.0000 | 0.00% |
| 128 | 8 | 23 | 0.1421 | 1.47% |
| 128 | 16 | 11 | 0.0261 | 0.02% |
| 128 | 32 | 1 | 0.0004 | 0.00% |
| 256 | any | 0 | 0.0000 | 0.00% |

**The pad has to be proportional to the tile.** One window of context is not
enough at any tile size - `tile 32 pad 8` and `tile 128 pad 8` are both visibly
wrong - because the shifted blocks move the window by half its width and an
absolute pad cannot cover a hole that scales with the tile. As a rule, pad by a
quarter to a half of the tile; anything at or above the image is exact, since the
single tile then is the image.

`--tile-pad` is rounded **up** to a whole window and never below one, so
`--tile-pad 0` and `--tile-pad 8` are the same request and mean "one window of
context", not "none".

`--tile auto` picks one whole-image pass when the image fits the budget and
otherwise the largest window-aligned tile that fits. Neither side of that
comparison is a constant:

* **The budget** is the memory a run may have. On the GPU it is the card's free
  VRAM less the checkpoint's own upload (49 MiB for the classical x4 - the x4
  file is 82 MiB but a third of it is the reference's fused qkv and its
  relative-position tables, which this engine never uploads) and a sixteenth for
  fragmentation; on the CPU it is **half of `MemAvailable`**, not a
  fixed figure - a 1 GiB constant, which is what this used to be, is both too
  small on a 32 GB machine and too large on a loaded one.
* **The estimate** is the activations this engine's own two layouts hold, per
  padded pixel: 3111 floats on the host and 3671 on the device for the 180-wide
  classical x4, from `backend::per_pixel_floats`, which is written term by term
  against the buffers rather than guessed. The two differ because the host frees
  the token buffers before the head (`Scratch::free_body`) and the device does
  not.

One pass is the right default when the image fits, and **`--tile 0`, one pass, is
the only exact mode.** The banner prints the tile it chose and the figure it
sized it against.

Where that lands, measured on this machine rather than derived: `--tile auto`
picks one pass for `classical-x4` to just under 700 pixels, and which side of 688
it lands on depends on the machine rather than on the checkpoint. The host's
criterion is `MemAvailable / 2`, so it moves: at 11.8 GB available that is a
5.9 GiB budget, which covers the 5617 MiB estimate at 680 pixels but not the
5749 at 688 - and both were observed directly, as a whole-image banner line at
680 and a 512 tile at 688. The device's criterion is the card's own free VRAM,
which moves the same way for the same reason: on an idle 8 GB card it picks one
pass to 704 pixels (7099 MiB estimated against 8111 free) and tiles above that,
and with 3.3 GB of the card held by another process it picks a 256 tile at the
same 704-pixel image, which is the check doing its job rather than failing.

If the estimate is nevertheless wrong - the driver's free figure drops while the
run allocates, or a foreign process takes the memory first - a `cuMemAlloc`
failure is retried at half the tile, down to one window, rather than being fatal.
A fallback out of `--tile 0` is announced on stderr even under `-q`: a run that
silently stops being exact has changed what was asked for.

### Memory

Peak RSS, whole process, fitted over four to seven input sizes from 128x128 to
1000x1000. It is linear in the area of the plane the network actually runs on - the
input rounded up to a whole window, plus one more window of padding, which is what
`Pre` builds - with a residual under 0.5 MiB:

| checkpoint | base | per padded pixel | at 128x128 | at 256x256 |
|---|---|---|---|---|
| classical x4 | 51 MiB | 12.2 KB | 271 MiB | 878 MiB |
| lightweight x2 | 6 MiB | 4.0 KB | 79 MiB | 281 MiB |

Those slopes are `backend::per_pixel_floats`, floored: the fitted values are
12.25 KB and 4.14 KB, under 2.3% above the model, with the model the lower of the
two at every size measured. So the budget `--tile auto` uses and the table above
are the same arithmetic, and the budget is on the safe side of it.

Every term of that model is a named buffer. The one that is not a field of
`Scratch` is `nbuf`, the staging block for the final `channel_layer_norm` before
`conv_after_body`, and it is worth a whole padded plane - 186 MB of the 3.29 GB
peak at 512x512. The curve, from `SWIN2SR_DEBUG_RSS=1` (an `eprintln` of `VmRSS`
at each phase of `cpu::forward` and `cpu::head`, off unless the variable is set):
2795 MB when the forward opens with the whole scratch already built, +189 MB at
`conv_first` (the padded input copy plus the `body_res` clone of it - one padded
plane and a bit), +8 MB a stage through the six stages, then the single-plane step
at `nbuf`, then the head's 64-wide `a` on top of that, and the peak.
`Scratch::free_body` then drops the process by 2.1 GB, which is why the head's
planes are not additive with the body's on the host - and are on the device, where
there is no `free_body`.

For comparison, PyTorch on the same machine is 557 MB + 9.7 KB for `classical-x4`
and 473 MB + 4.8 KB for `lightweight-x2`: it pays a ~500 MB interpreter and
oneDNN cost and then about half the per-pixel slope, so the engine is the lighter
of the two below roughly 200x200 and heavier above it.

The host slope is what `--tile auto` is sized against on `--device cpu`; the
device's is a different quantity with its own slope (below), which is why the two
budgets in the tiling section are not the same shape.

**Device memory is a different quantity and a different fit.** The card holds the
whole forward in ONE allocation - there is no device counterpart of the host's
`free_body`, so the head's octave buffers are additive with every body buffer
instead of replacing them - and that is most of why the model's device figure is
larger per padded pixel than its host one despite the host keeping five planes at
once (3671 floats against 3111 for the classical x4, and both are the CPU's
structural sum plus the head). The peak `nvidia-smi memory.used` over a run is
linear in the padded plane, and the base of that line is the driver's own context
plus the checkpoint's upload:

| checkpoint | fitted base | fitted per padded pixel | model | recorded peak at 512x512 |
|---|---|---|---|---|
| classical x4 | 164-168 MiB | 14.9 KB (3811 floats) | 3671 | 4094 MiB |
| classical x2 | 169 MiB | 11.2 KB (2856 floats) | 2675 | 3104 MiB |
| realworld x4 | 171 MiB | 18.9 KB (4836 floats) | 4439 | 5152 MiB |
| lightweight x2 | 123-127 MiB | 3.4 KB (866 floats) | 819 | 1016 MiB |

Each fit is over four to five sizes from 256 to 768 pixels (`classical-x4`
256/384/448/512, `classical-x2` 256/384/512/608/640, `realworld-x4`
256/384/448/512, `lightweight-x2` 512/648/768), with residuals of a few MiB. The
model is 3.7-8.2% BELOW every one of the four fitted slopes - 0.96, 0.94, 0.92 and
0.95 of the fit - which is where a budget wants to be and is the same side as the
host's. The one it describes worst is `realworld-x4`, consistent with that being
the only head that doubles the plane twice through separate convolutions.

Measured ceilings for one exact pass, from below: the largest size that RUNS is
the ceiling, because a failed `cuMemAlloc` records a peak below its own demand. On
memory that is `classical-x4` 704 pixels (736 asks for more than the card has
free), `realworld-x4` 640 and `classical-x2` 864; `realworld-x4` at 648 is
therefore already past its ceiling and is the one row above that only exists
because the fallback retried it.

There is a SECOND ceiling that has nothing to do with memory, and on a light
checkpoint it comes first. The token matmuls launch as
`(ceil(c_out / 16), ceil(tokens / 16), 1)` and a grid dimension is capped at 65535
blocks, so no launch can address more than `65535 * 16` tokens - and one padded
pixel is one token, so 1048560 pixels is the widest one-pass plane any device can
be asked for, whatever the card's size. The lightweight x2 reaches it first: 1000
pixels (a 1008 plane, 1016064 tokens) runs in 2.1 GB, and 1016 (a 1024 plane,
1048576 tokens) fails with `cuLaunchKernel(lg_linear) failed:
CUDA_ERROR_INVALID_VALUE grid=(4,65536,1)` - `grid.y` is `rows / 16` and there are
not 65535 blocks of 16 tokens. That failure is not an allocation failure, so
`run_with_fallback` does not and should not retry it: `--tile auto` counts it
(`backend::launches`) and a `--tile` typed by hand is refused before anything is
allocated, with the token count in the message rather than the driver's opaque
one.

The fused pixel-shuffle head is what makes the two classical rows fit at all: it
used to materialise a `4 * feat`-channel intermediate over the whole padded plane,
and the two classical checkpoints were unrunnable above 480 pixels wide before the
kernel's own column split.

A `512x512` run is 295936 padded pixels, a `648x648` one 495616 - the input
rounded up to a window plus a window of padding, the same plane the host table's
slope is quoted against.

## Performance

On an i7-class machine (24 threads) with a GTX 1080 (Pascal, sm_61), whole image,
one process including PNG IO and checkpoint load:

| checkpoint | input | CPU | CUDA |
|---|---|---|---|
| classical x4 | 128x128 -> 512x512 | 1.96 s | 1.16 s |
| classical x4 | 256x256 -> 1024x1024 | 8.70 s | 4.12 s |
| lightweight x2 | 128x128 -> 256x256 | 0.36 s | 0.17 s |
| lightweight x2 | 256x256 -> 512x512 | 1.52 s | 0.46 s |

Best of five, and the CPU column is the number to be sceptical of: this machine
runs unrelated work (a 21-thread job, load average 31) that has moved a single
measurement by 2-4x. The figures above are the fastest of five interleaved runs,
which is the only statistic that survives that; the profile in the next section
is cycle-counted for the same reason.

The checkpoint is memory-mapped and parsed lazily, so loading it costs nothing
measurable: a 16x16 input - the smallest useful one, one pass over 4 windows -
takes 0.13 s on the CPU and 0.12 s on the CUDA path from process start to PNG on
disk. That is the floor for any run, and it is the network, not the IO.

The CUDA column is this card idle (1809 of 1911 MHz, 75 C, 1 MiB in use). An
earlier revision of this table was measured with another process holding 2.5 GB
and the card at 93 C, and every GPU number in it was roughly twice what it is
here - including the 16x16 floor, which was 0.76 s. The launch-overhead
explanation that used to be attached to that floor was measuring the other
process, not this graph.

**Against PyTorch 2.6 (cu124)**, same checkpoints, forward only, 3-run average,
measured back to back with this engine on the same machine:

| checkpoint | 128x128 CPU | 128x128 CUDA | 256x256 CPU | 256x256 CUDA |
|---|---|---|---|---|
| classical x4 | 1.49 s | 0.32 s | 7.92 s | 1.46 s |
| lightweight x2 | 0.41 s | 0.12 s | 1.90 s | 0.45 s |

So on the CPU this engine is now within 1.1-1.3x of PyTorch on the 180-channel
checkpoints - and ahead of it on both sizes of the 60-channel one, which is the
one whose FLOPs are small enough that PyTorch's interpreter overhead is visible.
On the GPU the classical checkpoint is 2.8-3.6x off while the lightweight one is
within 1.5x; PyTorch runs the same graph in a fraction of the launches, and the
classical model is the one whose kernels are small relative to that. Both
backends are ordinary parallel code - the CPU side is rayon-parallel Rust with no
BLAS, the CUDA side calls the toolkit for every op the toolkit has - and against
this engine's first version, which ran the same graph with its own direct kernels
on one thread, it is tens of times on the CPU and several times on the GPU.

### Where the CPU time goes

Profiled with `rdtsc` - cycles the process is scheduled for, summed over threads,
so these numbers do not move when the machine is busy, unlike the wall clock
above them. `classical-x4` on a 128x128 input, 16.5 Gcycles in total:

| op | Mcycles | share |
|---|---|---|
| attention | 5371 | 32.5% |
| qkv | 2350 | 13.0% |
| fc2 | 1670 | 9.2% |
| stage conv 3x3 | 1605 | 8.9% |
| fc1 | 1418 | 7.8% |
| window scatter | 987 | 5.5% |
| attention proj | 800 | 4.4% |

The matmuls and the convolution are near where this hardware puts them: the
linear layer's inner loop holds an 8x10 tile of accumulators in registers and
sustains about 59 cycles per MFLOP, against 97 for the obvious row-at-a-time form
and 135 for the same form with only SSE2 - all three single-threaded, minimum of
nine interleaved runs. The tile is chosen at RUN time: `is_x86_feature_detected!`
picks between a `#[target_feature]` AVX2 worker and a baseline one, because an
8x10 tile needs twenty YMM registers and spills into something three times slower
without them. Attention is the outlier at 5.7 FLOP per cycle - it is
transcendental-bound on the softmax's `exp`, and it still evaluates one scalar
`exp` per (query, key). Vectorising that is the obvious next step, and so is
transposing Q/K/V per (window, head) so both of its inner loops vectorise over
keys instead of reducing scalars.

## Tests

```sh
cargo test --release                        # both backends
cargo test --release --no-default-features  # CPU only, no nvcc needed
```

* `tests/parity.rs` - the four checkpoints against their golden fixtures, the CPU
  op library's internal checks, and the rejection of a checkpoint whose header
  describes an architecture its tensors do not have.
* `tests/tiling.rs` - the tile loop: that it writes every output pixel from the
  right place (a constant input is the probe, because the whole-image answer is
  known and any unwritten pixel stands out as a zero), that a tile at or above
  the image is bit-identical to one pass, that `--tile-pad` rounds up to a whole
  window, and that more context strictly reduces the error.

Both files need the converted checkpoints and skip without them; see
`convert_all.sh` above.

Two more self-checks are in the binary rather than in `cargo test`, because they
need a specific device:

```
$ swin2sr --self-test
cpu selftest: 8 op families checked against their definitions
$ swin2sr -m swin2sr-classical-x4.safetensors --cuda-selftest
cuda selftest: 31 kernel/shape pairs checked against their CPU twins
```

## Licence and attribution

The Rust and CUDA code in this repository is licensed under the MIT license; see
[LICENSE](LICENSE).

This is an independent reimplementation of the Swin2SR architecture, which is by
[mv-lab](https://github.com/mv-lab/swin2sr) and Apache-2.0 licensed. Swin2SR is
itself heavily based on the
[Swin Transformer](https://github.com/microsoft/Swin-Transformer) by Microsoft,
and refers to KAIR, BasicSR and SwinIR. `tools/network_swin2sr.py` is copied
verbatim from the upstream repository (with `tools/timm_shim.py` standing in for
the `timm` import it uses), and is therefore a derived work, not covered by this
repository's copyright. It is used as the accuracy reference and by
`tools/compare.py`; `tools/convert.py` reshapes the released checkpoints and
transcribes their configurations.

The **checkpoints** are the Swin2SR authors' work as well. The converted
`.safetensors` files are format conversions of the official
`Swin2SR_ClassicalSR_X4_64.pth`, `Swin2SR_ClassicalSR_X2_64.pth`,
`Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth` and `Swin2SR_Lightweight_X2_64.pth`
files from the upstream Apache-2.0 release. Neither the original `.pth` files nor
the converted ones are redistributed in this repository; `convert_all.sh` fetches
the originals and converts them.
