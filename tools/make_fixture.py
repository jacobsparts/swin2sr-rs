#!/usr/bin/env python3
"""Write the golden fixtures swin2sr-rs's tests check against.

The engine is checked against the PUBLISHED network (tools/network_swin2sr.py,
copied verbatim from mv-lab/swin2sr), not against its own reading of it: a CPU
and a GPU backend that agree with each other prove only that they share a
mistake. This script runs that network on a deterministic input, records the
input and the output the network produces, and writes both as raw little-endian
f32 so the Rust side needs no numpy at test time.

    python3 tools/make_fixture.py --model models/Swin2SR_ClassicalSR_X4_64.pth \
        --task classical_sr --scale 4 --h 37 --w 29 --seed 1 \
        --out tests/data/classical_x4_37x29.bin

The input is written as C*H*W f32 (NCHW, no batch), the output as C*(H*scale)*(W*scale),
both in the order the engine reads them. The geometry is a 12-float header so one
file is self-describing:

    magic "SW2F" (4 bytes) | u32 version | u32 h | u32 w | u32 c | u32 scale
    u32 win | u32 flags | u32 reserved | f32 input[h*w*c] | f32 expected[oh*ow*c]

`--dump` also writes every intermediate activation as a `.pt` file, which is how
a divergence is located by STAGE on the Rust side rather than by bisecting the
output image (see tools/compare.py).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network_swin2sr import Swin2SR  # noqa: E402

# The released configurations, from the `--task` handling in
# main_test_swin2sr.py: (depths, embed_dim, num_heads, mlp_ratio, upsampler).
CONFIGS = {
    "classical_sr": dict(depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
                         window_size=8, upsampler="pixelshuffle", resi_connection="1conv"),
    "compressed_sr": dict(depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
                          window_size=8, upsampler="pixelshuffle_aux", resi_connection="1conv"),
    "real_sr": dict(depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
                    window_size=8, upsampler="nearest+conv", resi_connection="1conv"),
    "lightweight_sr": dict(depths=[6] * 4, embed_dim=60, num_heads=[6] * 4, mlp_ratio=2,
                           window_size=8, upsampler="pixelshuffledirect", resi_connection="1conv"),
}
MAGIC = b"SW2F"
VERSION = 1


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("params", "params_ema"):
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return obj


def build(args):
    cfg = CONFIGS[args.task]
    net = Swin2SR(upscale=args.scale, in_chans=3, img_size=args.patch_size, window_size=cfg["window_size"],
                  img_range=1., depths=cfg["depths"], embed_dim=cfg["embed_dim"],
                  num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
                  upsampler=cfg["upsampler"], resi_connection=cfg["resi_connection"])
    state = load_state(args.model)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected"
                         f"\n  {missing[:4]}\n  {unexpected[:4]}")
    net.eval()
    return net, cfg


def pad_like_reference(x, window_size):
    """main_test_swin2sr.py's padding: enough to reach the NEXT window multiple,
    by reflecting the whole plane (which mirrors the first h_pad columns of the
    image, not its last ones)."""
    _, _, h_old, w_old = x.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h_old + h_pad, :]
    x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w_old + w_pad]
    return x, h_old, w_old


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="classical_sr", choices=sorted(CONFIGS))
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--patch_size", type=int, default=64, help="training_patch_size (only changes img_size)")
    ap.add_argument("--win", type=int, default=8)
    ap.add_argument("--h", type=int, default=37)
    ap.add_argument("--w", type=int, default=29)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    net, cfg = build(args)
    if args.win != cfg["window_size"]:
        print(f"warning: --win {args.win} but the config uses {cfg['window_size']}", file=sys.stderr)

    g = torch.Generator().manual_seed(args.seed)
    x = torch.rand(1, 3, args.h, args.w, generator=g)
    xp, h_old, w_old = pad_like_reference(x, cfg["window_size"])

    store = {}
    dump = (lambda name, t: store.__setitem__(name, t.detach().float().cpu())) if args.dump else None
    if dump is not None:
        # Patch through the network by hand so each stage can be named. This is
        # the one place the fixture generator duplicates the network's forward.
        hooks = []

        def hook(name):
            def fn(_m, _i, out):
                dump(name, out[0] if isinstance(out, tuple) else out)
            return fn

        for name, mod in net.named_modules():
            if name and name.split(".")[-1] in ("conv_first", "conv_after_body", "norm",
                                                "patch_embed", "patch_unembed", "conv_before_upsample",
                                                "upsample", "conv_last"):
                hooks.append(mod.register_forward_hook(hook(name)))
        for i, layer in enumerate(net.layers):
            hooks.append(layer.register_forward_hook(hook(f"layers.{i}")))
            for j, blk in enumerate(layer.residual_group.blocks):
                hooks.append(blk.register_forward_hook(hook(f"layers.{i}.blocks.{j}")))
                hooks.append(blk.attn.register_forward_hook(hook(f"layers.{i}.blocks.{j}.attn")))
                hooks.append(blk.attn.qkv.register_forward_hook(hook(f"layers.{i}.blocks.{j}.qkv")))

    with torch.no_grad():
        y = net(xp)
    if isinstance(y, tuple):
        y = y[0]
    oh, ow = h_old * args.scale, w_old * args.scale
    y = y[0, :, :oh, :ow].contiguous()

    if args.dump:
        torch.save({"in_padded": xp, "in": x, **store}, args.dump)
        print(f"{args.dump}: {len(store) + 2} tensors")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    xi = x[0].contiguous().numpy().astype("<f4")
    yo = y.numpy().astype("<f4")
    with out.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<IIIIIIII", VERSION, args.h, args.w, 3, args.scale,
                             cfg["window_size"], 0, 0))
        fh.write(xi.tobytes())
        fh.write(yo.tobytes())
    print(f"{out}: {args.h}x{args.w} -> {oh}x{ow}, {out.stat().st_size} bytes, "
          f"expected range [{yo.min():.4f}, {yo.max():.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
