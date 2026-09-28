# swin2sr

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).

Swin2SR super-resolution in one self-contained binary: feed it a PNG, get back a
PNG at two or four times the resolution. No Python, PyTorch, ONNX Runtime or CUDA
toolkit needed at runtime.

```
swin2sr -m swin2sr-classical-x4.safetensors -i small.png -o large.png
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. Left to itself the binary uses the GPU when a driver is
  there and the CPU when it is not, saying which on stderr; `--device cpu|gpu`
  picks one by hand, and asking for `--device gpu` on a machine with no driver
  is an error rather than a quiet change of plans.
* Five checkpoints, converted from the official `.pth` files: classical x2 and
  x4, real-world x4, lightweight x2, and compressed x4.
* Tiling for images that do not fit in memory, with the fidelity cost measured
  rather than hidden - see [Large images](#large-images).

Both backends agree with the published network to within 1e-5 of 1.0.

## Download

Prebuilt binary and all five converted checkpoints are attached to the
[releases](https://github.com/jacobsparts/swin2sr-rs/releases) as plain files.

| asset | contents |
|---|---|
| `swin2sr-linux-x86_64` | the engine: x86-64 Linux, glibc >= 2.34. The GPU path needs a compute capability 6.1+ card; `--device cpu` runs anywhere |
| `swin2sr-*.safetensors` | the five checkpoints, 19-82 MiB each |

```sh
chmod +x swin2sr-linux-x86_64        # a download does not carry the executable bit
./swin2sr-linux-x86_64 -m swin2sr-realworld-x4.safetensors -i in.png -o out.png
```

## Models

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

## Usage

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
| `--device cpu\|gpu` | which backend to use; naming `gpu` yourself is an error rather than a fallback |
| `--tile <n\|auto>` | process in tiles of n pixels a side; `0` (default) is one whole-image pass |
| `--tile-pad <n>` | context kept around each tile (default 32, rounded up to a window) |
| `--aux <path>` | `compressed-x4` only: also write the head's second, low-resolution image |
| `-q, --quiet` | no progress output |
| `--info` | what this binary and this machine can do |
| `-h, --help`, `-V, --version` | usage, version |

Input and output are stdin and stdout by default and `-` names either stream, so
`swin2sr -m model.safetensors -i in.png | display -` works. Progress goes to
stderr.

## Large images

A whole-image pass (`--tile 0`, the default) is the only exact mode. `--tile n`
processes a larger image in n-pixel tiles with `--tile-pad` pixels of context
kept around each one, which is a memory tradeoff: a tile boundary cuts the
network's receptive field, so the pad has to be a quarter to a half of the tile
for the seams to stay invisible.

Measured on a 120x160 input with `classical-x4`, against the same run untiled, in
8-bit levels:

| tile | pad | max diff | pixels off by >2 levels |
|---|---|---|---|
| 32 | 32 | 1 | 0.00% |
| 64 | 32 | 1 | 0.00% |
| 128 | 32 | 1 | 0.00% |
| 256 | any | 0 | 0.00% |

`--tile auto` picks one whole-image pass when the image fits the available
memory and the largest aligned tile that does when it does not. If an allocation
nevertheless fails, the run is retried at half the tile rather than being fatal,
and a fallback out of one pass is announced even under `-q`.

## Licence and attribution

The Rust and CUDA code here is MIT licensed; see [LICENSE](LICENSE).

This is an independent reimplementation of the Swin2SR architecture, which is by
[mv-lab](https://github.com/mv-lab/swin2sr) and Apache-2.0 licensed.
`tools/network_swin2sr.py` is copied verbatim from the upstream repository and is
therefore a derived work. The checkpoints are the Swin2SR authors' work: the
converted `.safetensors` files are format conversions of the official
`Swin2SR_ClassicalSR_X4_64.pth`, `Swin2SR_ClassicalSR_X2_64.pth`,
`Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth`, `Swin2SR_Lightweight_X2_64.pth` and
`Swin2SR_CompressedSR_X4_48.pth` files from the upstream Apache-2.0 release.
Neither the originals nor the converted files are redistributed in this
repository; `convert_all.sh` fetches the originals and converts them.
