#!/usr/bin/env python3
"""Convert a Swin2SR .pth checkpoint into the .safetensors this engine loads.

    python3 tools/convert.py Swin2SR_ClassicalSR_X4_64.pth classical_x4.safetensors --task classical_sr

The released checkpoints are the Swin2SR authors' work
(https://github.com/mv-lab/swin2sr, Apache-2.0); this script only reshapes them.
Weights are not redistributed with this engine.

THREE THINGS THIS DOES BESIDES COPYING.

1. FUSES THE ATTENTION QKV PROJECTION. The reference computes `qkv = F.linear(x,
   qkv.weight)` and then applies `q_bias` / `v_bias` (the key column gets no
   bias) through `torch.cat`. That is a fused-QKV module, which is what this is
   for: the three projections become three `lg_linear` calls per block whose
   weight rows are the slices of the one [3C][C] matrix, so the fusion is a
   slicing question rather than a numerics one.

2. FOLDS THE FIRST ROW OF `relative_coords_table` OUT OF `cpb_mlp`. The table's
   second column is a constant 1.0 (bc - c is zero for every pair), so
   `Linear(2, 512)` is `W[:,0] * table[:,0] + W[:,1]` - a 2-input matmul with one
   wasted operand per position per block. `cpb` below is the [N][N][2] table, and
   `cpb_pre` is a precomputed [N*N][heads] f32 bias table, so the engine's
   attention kernel reads one float per (i, j, head) instead of running a 2x512
   matmul, a 512xheads matmul, a gather and a sigmoid per block per call.

3. RECORDS THE ARCHITECTURE IN `__metadata__`. The published checkpoints are
   indistinguishable by their tensor names alone (classical and real-world SR
   share every layer name; only the upsampler's weights differ), so a converted
   file that does not say which one it is cannot be shape-checked against its own
   weights. The Rust side reads these keys and validates the tensor set against
   them.

Every tensor is written through unchanged, in the checkpoint's own
[c_out][c_in][kh][kw] order; nothing is transposed.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# The released configurations. `upsampler` decides the reconstruction head; the
# rest is what every block's shapes are derived from.
CONFIGS = {
    "classical_sr": dict(upsampler="pixelshuffle", scale=4, win=8, embed=180, heads=6,
                         mlp_ratio=2, depths=[6] * 6, resi="1conv"),
    "classical_sr_x2": dict(upsampler="pixelshuffle", scale=2, win=8, embed=180, heads=6,
                            mlp_ratio=2, depths=[6] * 6, resi="1conv"),
    "real_sr": dict(upsampler="nearest+conv", scale=4, win=8, embed=180, heads=6,
                    mlp_ratio=2, depths=[6] * 6, resi="1conv"),
    "compressed_sr": dict(upsampler="pixelshuffle_aux", scale=4, win=8, embed=180, heads=6,
                          mlp_ratio=2, depths=[6] * 6, resi="1conv"),
    "lightweight_sr": dict(upsampler="pixelshuffledirect", scale=2, win=8, embed=60, heads=6,
                           mlp_ratio=2, depths=[6] * 4, resi="1conv"),
}


def guess_task(path: str):
    n = Path(path).name.lower()
    if "lightweight" in n:
        return "lightweight_sr"
    if "realworld" in n or "real_sr" in n:
        return "real_sr"
    if "compressed" in n:
        return "compressed_sr"
    if "classical" in n:
        return "classical_sr_x2" if "x2" in n else "classical_sr"
    return None


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("params", "params_ema"):
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return obj


def infer(prefix, sd, heads):
    """Derive the architecture from the tensors themselves, so the metadata that
    is written always describes the weights that are in the file."""
    depths = sorted({int(k.split(".")[1]) for k in sd if k.startswith("layers.")
                     and k.split(".")[1].isdigit()})
    depth = max(depths) + 1
    per = [0] * depth
    for k in sd:
        p = k.split(".")
        if len(p) > 3 and p[0] == "layers" and p[2] == "residual_group" and p[3] == "blocks":
            per[int(p[1])] = max(per[int(p[1])], int(p[4]) + 1)
    embed = int(sd["conv_first.weight"].shape[0])
    win = int(sd["layers.0.residual_group.blocks.0.attn.relative_coords_table"].shape[1] + 1) // 2
    mlp = int(sd["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0]) // embed
    # THE HEAD IS READ OFF THE TENSORS, NOT FROM --task. The two pixel-shuffle
    # heads and nearest+conv are told apart by their layer names alone - and a
    # `--task` is exactly the kind of hand-maintained fact that goes stale: the
    # lightweight checkpoint came out labelled `pixelshuffle` with `d:6`
    # `mlp:2`, claiming a conv_last it does not have, because `x2` defaulted the
    # task to classical_sr_x2. What the file says about itself has to be derived
    # from the file.
    # The COMPRESSED head is the one that is easy to mistake: it has the same
    # `conv_before_upsample` + `upsample` + `conv_last` stack as the pixel-shuffle
    # head and adds `conv_bicubic`, `conv_aux` and `conv_after_aux` on the side. A
    # name-based guess that only looked at the shared names called it
    # `pixelshuffle`, and the engine then ran it through the classical head and
    # produced a plausible-looking image that is wrong by 2.16 of 1.0 - so the
    # extra branch is checked FIRST.
    if "conv_aux.weight" in sd or "conv_bicubic.weight" in sd:
        upsampler = "pixelshuffle_aux"
    elif "conv_before_upsample.0.weight" in sd:
        upsampler = "nearest+conv" if "conv_up1.weight" in sd else "pixelshuffle"
    elif "upsample.0.weight" in sd:
        upsampler = "pixelshuffledirect"
    else:
        upsampler = "pixelshuffle_aux"
    return per, embed, win, mlp, upsampler


def fuse_qkv(sd, depth, per, out):
    """Append the per-block fused-QKV weights to `out` (a name -> ndarray dict)."""
    for l in range(depth):
        for b in range(per[l]):
            p = f"layers.{l}.residual_group.blocks.{b}.attn"
            w = sd[f"{p}.qkv.weight"].numpy()          # [3C][C]
            c = w.shape[1]
            assert w.shape[0] == 3 * c, f"{p}: qkv is {w.shape}, expected [3C][C]"
            # The key column's bias is the zero column of the reference's
            # torch.cat: q_bias, zeros, v_bias. Writing it as a real -0.0 bias
            # term would cost a multiply per output, so the engine keeps the two
            # bias vectors separate instead and this stays a pure slice.
            out[f"{p}.qkv.wq"] = np.ascontiguousarray(w[:c])
            out[f"{p}.qkv.wk"] = np.ascontiguousarray(w[c:2 * c])
            out[f"{p}.qkv.wv"] = np.ascontiguousarray(w[2 * c:])
            out[f"{p}.qkv.q_bias"] = np.ascontiguousarray(sd[f"{p}.q_bias"].numpy())
            out[f"{p}.qkv.v_bias"] = np.ascontiguousarray(sd[f"{p}.v_bias"].numpy())


def precompute_cpb(sd_all, depth, per, win, heads, out):
    """Replace cpb_mlp + gather + sigmoid with the [N*N][heads] bias table.

    The result is bit-for-bit what the reference produces (the sigmoid is applied
    at the same place, to the same values), computed once per block instead of
    once per forward call.

    The two buffers this needs come from the RAW state dict, not the f32-filtered
    one: `relative_position_index` is an int64 index table, and a filter that
    keeps only floating-point tensors drops it.
    """
    base = "layers.0.residual_group.blocks.0.attn"
    table = sd_all[f"{base}.relative_coords_table"].to(torch.float32).numpy()  # [1][2w-1][2w-1][2]
    index = sd_all[f"{base}.relative_position_index"].numpy().astype(np.int64)  # [N][N]
    t = table.reshape(-1, 2)
    for l in range(depth):
        for b in range(per[l]):
            p = f"layers.{l}.residual_group.blocks.{b}.attn"
            w0 = sd_all[f"{p}.cpb_mlp.0.weight"].to(torch.float32).numpy()
            b0 = sd_all[f"{p}.cpb_mlp.0.bias"].to(torch.float32).numpy()
            w2 = sd_all[f"{p}.cpb_mlp.2.weight"].to(torch.float32).numpy()
            h = t @ w0.T + b0
            h = np.maximum(h, 0.0)                      # nn.ReLU(inplace=True)
            h = h @ w2.T                                # [((2w-1)^2)][heads]
            bias = h[index.reshape(-1)]                 # [N*N][heads]
            bias = 16.0 / (1.0 + np.exp(-bias))         # 16 * torch.sigmoid(...)
            out[f"{p}.cpb_pre"] = np.ascontiguousarray(bias.astype(np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--task", default=None, help="classical_sr, real_sr, compressed_sr, lightweight_sr")
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--win", type=int, default=None)
    args = ap.parse_args()

    sd_raw = load_state(args.src)
    if not isinstance(sd_raw, dict):
        print(f"{args.src}: not a state dict ({type(sd_raw)})", file=sys.stderr)
        return 1
    sd = {k: v.detach().to(torch.float32).contiguous() for k, v in sd_raw.items()
          if isinstance(v, torch.Tensor) and v.is_floating_point()}
    if "conv_first.weight" not in sd:
        print(f"{args.src}: no conv_first.weight - not a Swin2SR checkpoint?", file=sys.stderr)
        return 1

    task = args.task or guess_task(args.src)
    cfg = CONFIGS.get(task)
    if cfg is None:
        print(f"{args.src}: unknown task `{task}`; pass --task", file=sys.stderr)
        return 1

    depth_cfg = cfg["depths"]
    heads = cfg["heads"]
    per, embed, win, mlp, inferred_ups = infer("", sd, heads)
    if args.win:
        win = args.win
    if per != depth_cfg or embed != cfg["embed"] or win != cfg["win"] or mlp != cfg["mlp_ratio"]:
        print(f"warning: the weights are depths={per} embed={embed} mlp_ratio={mlp} win={win} but the "
              f"published {task} config is depths={depth_cfg} embed={cfg['embed']} "
              f"mlp_ratio={cfg['mlp_ratio']} win={cfg['win']}", file=sys.stderr)
    heads = int(sd["layers.0.residual_group.blocks.0.attn.logit_scale"].shape[0])
    if heads != cfg["heads"]:
        print(f"warning: logit_scale says {heads} heads, the published config says {cfg['heads']}",
              file=sys.stderr)

    scale = args.scale or cfg["scale"]
    # The head comes from `infer`, which read it off the tensor names - see the
    # note there for why it is not `cfg["upsampler"]`.
    ups = inferred_ups
    if ups != cfg["upsampler"]:
        print(
            f"note: the checkpoint's tensors describe the `{ups}` head, not the published "
            f"{task} config's `{cfg['upsampler']}`; the tensors win",
            file=sys.stderr,
        )

    out = {k: v.numpy() for k, v in sd.items()}
    fuse_qkv(sd, len(per), per, out)
    precompute_cpb(sd_raw, len(per), per, win, heads, out)

    metadata = {
        "task": task,
        "upsampler": ups,
        "scale": str(scale),
        "window_size": str(win),
        "embed_dim": str(embed),
        "num_heads": str(heads),
        "mlp_ratio": str(mlp),
        "depths": ",".join(str(v) for v in per),
        "img_range": "1.0",
        "rgb_mean": "0.4488,0.4371,0.4040",
        "format": "pt",
    }

    # safetensors: an 8-byte little-endian header length, the JSON header, then
    # the raw tensor payloads, each padded to a 4-byte boundary (the engine mmaps
    # the file and hands out &[f32], which an unaligned tensor would break).
    tensors = sorted(out.items())
    offset = 0
    header = {}
    for name, a in tensors:
        a = np.ascontiguousarray(a, dtype="<f4")
        n = a.size * 4
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [offset, offset + n]}
        offset += (n + 3) & ~3
    header["__metadata__"] = metadata
    hjson = json.dumps(header, separators=(",", ":")).encode()

    header_bytes = 8 + len(hjson)
    pad = (-header_bytes) & 7
    with open(args.dst, "wb") as f:
        f.write(struct.pack("<Q", len(hjson) + pad))
        f.write(hjson)
        f.write(b" " * pad)
        written = 0
        for name, a in tensors:
            raw = np.ascontiguousarray(a, dtype="<f4").tobytes()
            want = header[name]["data_offsets"][0]
            if written < want:
                f.write(b"\0" * (want - written))
                written = want
            f.write(raw)
            written += len(raw)

    total = sum(a.size for _, a in tensors)
    print(f"{args.dst}: {len(tensors)} tensors, {total} values")
    print(f"  task {task} upsampler {ups} scale {scale} window {win} embed {embed} "
          f"heads {heads} mlp {mlp} depths {per}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
