#!/usr/bin/env python3
"""Compare a `--dump` from the engine against the PyTorch reference, stage by stage.

Why this exists: a fixture says the engine is wrong, not WHERE. The engine and the
reference are two independent transcriptions of the same network, so the first
stage at which they disagree is the answer - and the stages are named after
torch's own module paths so the two can be matched mechanically.

    python3 tools/compare.py --model models/Swin2SR_ClassicalSR_X4_64.pth \
        --dump /tmp/eng.dump --h 37 --w 29

The dump is the flat f32 blob `swin2sr --dump` writes, with a JSON index next to
it. The input is regenerated here from (h, w) - the engine's `--raw` fills the
plane with the golden-ratio sequence, so the two sides can agree on an input
neither had to store.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network_swin2sr import Swin2SR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fixture import CONFIGS, load_state, pad_like_reference  # noqa: E402


def raw_input(h, w):
    """The engine's `--raw` input, value for value."""
    n = 3 * h * w
    i = np.arange(n, dtype=np.float32)
    t = i * np.float32(0.6180339887)
    return (t - np.floor(t)).astype(np.float32)


def engine_input(dump):
    """The input the dump was produced from, as [3][h][w]."""
    return raw_input(dump["geometry"]["h"], dump["geometry"]["w"])


class Compare:
    def __init__(self):
        self.rows = []

    def add(self, name, want, got):
        """`want` and `got` are both numpy arrays flattened in the same order."""
        want = want.reshape(-1).astype(np.float64)
        got = got.reshape(-1).astype(np.float64)
        if want.size != got.size:
            self.rows.append((name, None, None, f"shape {want.size} vs {got.size}"))
            return
        d = np.abs(want - got)
        self.rows.append((name, float(d.max()), float(d.mean()), ""))
        return d

    def report(self, tol):
        w = max(len(r[0]) for r in self.rows)
        worst = 0.0
        first_bad = None
        for name, mx, mean, note in self.rows:
            if mx is None:
                print(f"  {name:<{w}}  {note}")
                if first_bad is None:
                    first_bad = name
                continue
            flag = "" if mx <= tol else "   <-- DIVERGES"
            print(f"  {name:<{w}}  max {mx:.3e}  mean {mean:.3e}{flag}")
            worst = max(worst, mx)
            if mx > tol and first_bad is None:
                first_bad = name
        print()
        if first_bad is None:
            print(f"every stage agrees within {tol:.0e} (worst {worst:.3e})")
            return 0
        print(f"first stage over tolerance: {first_bad}")
        return 1


def load_dump(path):
    idx = json.loads(Path(str(path) + ".json").read_text())
    blob = np.fromfile(path, dtype="<f4")
    out = {}
    for e in idx["order"]:
        out[e["name"]] = blob[e["offset"]:e["offset"] + e["count"]]
    return idx["geometry"], out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--task", default="classical_sr")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--plot", action="store_true", help="write a PNG of the worst stage's difference")
    args = ap.parse_args()

    geo, eng = load_dump(args.dump)
    h, w, hp, wp, c, win, scale = (geo["h"], geo["w"], geo["hp"], geo["wp"],
                                   geo["c"], geo["win"], geo["scale"])
    print(f"engine dump: {w}x{h} padded to {wp}x{hp}, {c} channels, window {win}, x{scale}")

    cfg = CONFIGS[args.task]
    net = Swin2SR(upscale=args.scale, in_chans=3, img_size=64, window_size=cfg["window_size"],
                  img_range=1., depths=cfg["depths"], embed_dim=cfg["embed_dim"],
                  num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
                  upsampler=cfg["upsampler"], resi_connection=cfg["resi_connection"])
    missing, unexpected = net.load_state_dict(load_state(args.model), strict=False)
    if missing or unexpected:
        print(f"state_dict mismatch: {missing[:3]} / {unexpected[:3]}", file=sys.stderr)
        return 1
    net.eval()

    x = torch.from_numpy(raw_input(h, w)).reshape(1, 3, h, w)
    # The model subtracts the mean itself, so it is fed the RAW padded plane -
    # subtracting it here as well would compare every stage against a network that
    # had seen (x - mean) twice, which looks exactly like a broken conv_first.
    xp, _, _ = pad_like_reference(x, cfg["window_size"])
    adjusted = (xp - net.mean.detach()) * net.img_range

    stages = {}

    def grab(module, name, kind="tokens"):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            with torch.no_grad():
                t = t.detach().float()
                if kind == "tokens":
                    # [tokens][C] - the ENGINE's order. A transpose here would
                    # report every token stage as divergent while both sides were
                    # right, which is how an earlier round of this harness wasted
                    # an hour.
                    stages[name] = t.reshape(-1, t.shape[-1]).numpy().copy()
                elif kind == "plane":
                    stages[name] = t.reshape(t.shape[1], -1).numpy()
                elif kind == "tokens_transposed":
                    # [C][tokens] - for `PatchEmbed`, whose output is
                    # [B, L, C] where L is the FLATTENED PLANE, not the channel
                    # axis: an NCHW dump compares against this, element for
                    # element, and against nothing else. Getting this wrong
                    # reported the (correct) patch_embed as a 7.2 divergence.
                    stages[name] = t.reshape(-1, t.shape[-1]).numpy().T.copy()
                else:  # qkv: [3][tokens][C], the order the attention kernel reads
                    c3 = t.shape[-1] // 3
                    stages[name] = t.reshape(-1, 3, c3).transpose(1, 0, 2).reshape(-1).copy()
        module.register_forward_hook(fn)

    grab(net.conv_first, "conv_first", "plane")
    grab(net.patch_embed, "patch_embed", "tokens_transposed")
    for i, layer in enumerate(net.layers):
        for j, blk in enumerate(layer.residual_group.blocks):
            grab(blk, f"layers.{i}.blocks.{j}", "tokens_transposed")
            grab(blk.norm1, f"layers.{i}.blocks.{j}.norm1", "tokens_transposed")
            grab(blk.attn, f"layers.{i}.blocks.{j}.attn")
            # `WindowAttention.forward` calls `F.linear(x, self.qkv.weight, ...)`
            # DIRECTLY, so a hook on the `qkv` MODULE never fires - a trap that
            # silently dropped all 36 qkv rows from an earlier report. The input
            # is captured with a pre-hook and the projection redone here.
            def qkv_hook(_m, inputs, name=f"layers.{i}.blocks.{j}.qkv", mod=blk.attn):
                x_in = inputs[0].detach().float()
                bias = torch.cat([mod.q_bias, torch.zeros_like(mod.v_bias), mod.v_bias])
                q = torch.nn.functional.linear(x_in, mod.qkv.weight, bias)
                c3 = q.shape[-1] // 3
                stages[name] = (q.reshape(-1, 3, c3).numpy().transpose(1, 0, 2)
                                .reshape(-1).copy())
            blk.attn.register_forward_pre_hook(qkv_hook)
        grab(layer, f"layers.{i}", "tokens_transposed")
    grab(net.norm, "norm", "tokens_transposed")
    grab(net.conv_after_body, "conv_after_body", "plane")
    grab(net.conv_before_upsample, "conv_before_upsample", "plane")
    for o, mod in enumerate(net.upsample):
        if isinstance(mod, torch.nn.PixelShuffle):
            # The PIXEL SHUFFLE, not the conv before it: the conv's output is
            # 4C wide at the pre-shuffle resolution, the shuffle's is C wide at twice it.
            grab(net.upsample[o], f"upsample.{o // 2}", "plane")

    with torch.no_grad():
        y = net(xp)
    if isinstance(y, tuple):
        y = y[0]
    y = y[0, :, :h * scale, :w * scale].numpy()

    cmp = Compare()
    cmp.add("input_padded", adjusted.numpy().reshape(-1), eng["input_padded"])
    for name, arr in stages.items():
        if name not in eng:
            cmp.rows.append((name, None, None, "engine did not dump this stage"))
            continue
        cmp.add(name, arr, eng[name])
    cmp.add("output", y.reshape(-1), eng["output"])
    # The engine's `conv_after_body` is the POST-residual activation (that is what
    # the stage means in the graph), while the reference's hook fires on the Conv2d
    # alone. Add the body's skip in torch so the two are the same tensor.
    if "conv_after_body" in eng and "conv_first" in eng:
        cmp.add("conv_after_body+residual",
                stages["conv_after_body"].reshape(-1) + eng["conv_first"], eng["conv_after_body"])
    return cmp.report(args.tol)


if __name__ == "__main__":
    sys.exit(main())
