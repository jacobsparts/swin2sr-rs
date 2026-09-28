# swin2sr

Swin2SR image restoration - super-resolution - as a single self-contained
binary. Feed it a PNG, get back a larger PNG. No Python, PyTorch, ONNX Runtime
or CUDA toolkit needed at runtime.

```
swin2sr -m swin2sr-classical-x4.safetensors -i small.png -o large.png
```

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu);
pixeldeck, the family's local web app, drives this and the others.

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. `--device cpu|gpu` picks one by hand; left to itself the
  binary uses the GPU when a driver is there and the CPU when it is not, saying
  which on stderr. Asking for `--device gpu` on a machine with no driver is an
  error rather than a quiet change of plans.
* Five checkpoints, converted from the official `.pth` files: classical x2 and
  x4, real-world x4, lightweight x2, and compressed x4. See
  [Choosing a checkpoint](#choosing-a-checkpoint).
* Agreement with the published network to within 1e-5 of 1.0 on both backends -
  see [Accuracy](#accuracy).
* Tiling for images that do not fit in memory, with the fidelity cost measured
  rather than hidden - see [Large images](#large-images).

## Download

Prebuilt binaries and all five converted checkpoints are attached to the
[releases](https://github.com/jacobsparts/swin2sr-rs/releases) as plain files.

| asset | contents |
|---|---|
| `swin2sr-linux-x86_64` | CPU + CUDA. x86-64 Linux, glibc >= 2.34. The GPU path needs a compute capability 6.1+ card; `--device cpu` runs anywhere |
| `swin2sr-linux-x86_64-cpu-only` | the same build without CUDA: the CPU path only |
| `swin2sr-*.safetensors` | the five checkpoints, 19-82 MiB each |

```sh
chmod +x swin2sr-linux-x86_64        # a download does not carry the executable bit
./swin2sr-linux-x86_64 -m swin2sr-realworld-x4.safetensors -i in.png -o out.png
```

## Build

```sh
cargo build --release                      # needs nvcc (set NVCC= if not on PATH)
cargo build --release --no-default-features  # CPU only, no CUDA toolkit needed
```

The default build produces one binary with both backends. The kernels are
compiled for compute capability 6.1, 7.5 and 8.0 plus 8.0 PTX, so the GPU path
runs on Pascal through Ampere and on anything newer through the PTX. `lightgpu`
is a normal Cargo dependency on [its repository](https://github.com/jacobsparts/lightgpu),
so a clone builds on its own.

The converted checkpoints are not in the repository. `./convert_all.sh`
downloads the upstream `.pth` files and converts them, and regenerates the test
fixtures; the tests skip, naming that script, until it has been run.

## Run

```sh
swin2sr -m swin2sr-classical-x4.safetensors -i in.png -o out.png
swin2sr -m swin2sr-realworld-x4.safetensors -i in.png -o out.png --device cpu
swin2sr -m swin2sr-compressed-x4.safetensors -i in.png -o out.png --aux preview.png
```

| flag | meaning |
| --- | --- |
| `-m, --model <path>` | converted `.safetensors` checkpoint (required) |
| `-i, --input <path>` | input PNG, or `-` (default: stdin) |
| `-o, --output <path>` | output PNG, or `-` (default: stdout) |
| `--device cpu\|gpu` | which backend to use. The default is the GPU when the binary has CUDA *and* a driver answers, otherwise the CPU, and it says on stderr when it falls back; naming `gpu` yourself is an error rather than a fallback |
| `--tile <n\|auto>` | process in tiles of n pixels a side; `0` (default) is one whole-image pass |
| `--tile-pad <n>` | context kept around each tile (default 32, rounded up to a window) |
| `--aux <path>` | `compressed-x4` only: also write the head's second, low-resolution image |
| `-q, --quiet` | no progress output |
| `--info` | what this binary and this machine can do |
| `-h, --help`, `-V, --version` | usage, version |

Input and output are stdin and stdout by default and `-` names either stream, so
`swin2sr -m model.safetensors -i in.png | display -` works. Progress goes to
stderr.

## Choosing a checkpoint

The file decides the task and the scale; there is no flag for either.

| checkpoint | scale | what it is for |
|---|---|---|
| `swin2sr-classical-x4` | x4 | bicubic-degraded input, the usual benchmark setting |
| `swin2sr-classical-x2` | x2 | as above, x2 |
| `swin2sr-realworld-x4` | x4 | **photographs.** Trained on BSRGAN degradations, so it handles JPEG and real sensor damage |
| `swin2sr-lightweight-x2` | x2 | ~10x faster and visibly softer, for a quick look |
| `swin2sr-compressed-x4` | x4 | input that has been through a codec; also writes a second, low-resolution image with `--aux` |

Use `realworld-x4` on a photograph and `classical-x4` on an image that was
cleanly downsampled; the classical models assume their input really is a bicubic
downsample, and a JPEG's blocking is off their training distribution.
`compressed-x4` is the one for an input that has already been through a codec.

## Accuracy

Every checkpoint is checked against the published network - `tools/network_swin2sr.py`,
copied verbatim from upstream - by running both on the same seeded input and
comparing the full output plane. Two backends agreeing with each other prove
only that they share a mistake, so the reference is always the printed network.

| checkpoint | CPU | CUDA |
|---|---|---|
| classical x4 | 4.0e-6 | 8.0e-6 |
| classical x2 | 5.0e-6 | 1.0e-5 |
| real-world x4 | 3.0e-6 | 4.0e-6 |
| lightweight x2 | 2.0e-6 | 4.0e-6 |
| compressed x4 | 2.0e-6 | 5.0e-6 |
| compressed x4 aux | 1.0e-6 | 4.0e-6 |

max absolute difference from the reference, on a 0-1 scale. Reproduce any row:

```sh
swin2sr -m swin2sr-classical-x4.safetensors --verify tests/data/classical_x4.bin --device cpu
```

`--tol` defaults to 2e-3, an order of magnitude either side of the gap between a
correct transcription (~1e-6) and a wrong tap, transposed layout or missing skip
(1e-2 or worse). On an 8-bit PNG the two agree to within one level on every
input.

`cargo test --release` runs the fixtures and the tile loop; `--self-test` and
`--cuda-selftest` check the op library and every CUDA kernel against its CPU twin.

## Large images

A whole-image pass (`--tile 0`, the default) is the only exact mode. `--tile n`
processes a larger image in n-pixel tiles with `--tile-pad` pixels of context
kept around each one, which is a memory tradeoff: a tile boundary cuts the
network's receptive field, so the pad has to be a quarter to a half of the tile
for the seams to stay invisible. One window of context is not enough at any tile
size.

Measured on a 120x160 input with `classical-x4`, against the same run untiled, in
8-bit levels:

| tile | pad | max diff | pixels off by >2 levels |
|---|---|---|---|
| 32 | 32 | 1 | 0.00% |
| 64 | 32 | 1 | 0.00% |
| 128 | 32 | 1 | 0.00% |
| 256 | any | 0 | 0.00% |

`--tile auto` picks one whole-image pass when the image fits the available
memory and the largest aligned tile that does when it does not. The banner prints
what it chose and the figure it sized it against. If an allocation nevertheless
fails, the run is retried at half the tile rather than being fatal, and a
fallback out of one pass is announced even under `-q` - a run that silently
stops being exact has changed what was asked for.

## Performance

Whole image, one process including PNG IO and checkpoint load, on an i7-class
machine (24 threads) with a GTX 1080, best of five:

| checkpoint | input | CPU | CUDA |
|---|---|---|---|
| classical x4 | 128x128 -> 512x512 | 1.96 s | 1.16 s |
| classical x4 | 256x256 -> 1024x1024 | 8.70 s | 4.12 s |
| lightweight x2 | 128x128 -> 256x256 | 0.36 s | 0.17 s |
| lightweight x2 | 256x256 -> 512x512 | 1.52 s | 0.46 s |
| compressed x4 | 128x128 -> 512x512 | 2.15 s | 0.93 s |
| compressed x4 | 256x256 -> 1024x1024 | 9.04 s | 2.93 s |

The body is the whole bill: the compressed head's extra parameters, bicubic
resample and second output cost nothing measurable at these sizes.

## Licence and attribution

The Rust and CUDA code here is MIT licensed; see [LICENSE](LICENSE).

This is an independent reimplementation of the Swin2SR architecture, which is by
[mv-lab](https://github.com/mv-lab/swin2sr) and Apache-2.0 licensed.
`tools/network_swin2sr.py` is copied verbatim from the upstream repository (with
`tools/timm_shim.py` standing in for the `timm` import) and is therefore a
derived work, used as the accuracy reference. `tools/convert.py` reshapes the
released checkpoints.

The checkpoints are the Swin2SR authors' work. The converted `.safetensors`
files are format conversions of the official `Swin2SR_ClassicalSR_X4_64.pth`,
`Swin2SR_ClassicalSR_X2_64.pth`, `Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth`,
`Swin2SR_Lightweight_X2_64.pth` and `Swin2SR_CompressedSR_X4_48.pth` files from
the upstream Apache-2.0 release. Neither the originals nor the converted files
are redistributed in this repository; `convert_all.sh` fetches the originals and
converts them.
