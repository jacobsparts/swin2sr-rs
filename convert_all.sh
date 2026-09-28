#!/bin/sh
# Convert every released Swin2SR checkpoint this engine's tests and README use,
# and regenerate the golden fixtures from them.
#
#     ./convert_all.sh
#
# The converted checkpoints are ~85 MiB each and are deliberately NOT in the
# repository, so the integration tests SKIP until this has been run (a missing
# checkpoint is not a failing test - see tests/common/mod.rs). One `torch.load`
# per released checkpoint is needed to convert; the engine itself needs neither
# Python nor torch.
#
# The `.pth` files are the Swin2SR authors' work (https://github.com/mv-lab/swin2sr,
# Apache-2.0). They are downloaded from the upstream release when absent and are
# not redistributed here. The converted files are written next to them, in the
# family's shared models/ directory, and used by every engine's tests.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
cd "$here"

# The directory holding the family's checkpoints. Override with SWIN2SR_MODELS,
# the same variable the tests read.
models=${SWIN2SR_MODELS:-$here/../models}
# A python with torch. The engine needs none, this script does.
python=${SWIN2SR_PYTHON:-/usr/local/bin/python3.11}

if ! "$python" -c 'import torch, numpy' 2>/dev/null; then
    python=python3
fi
if ! "$python" -c 'import torch, numpy' 2>/dev/null; then
    echo "convert_all.sh needs a python with torch and numpy (set SWIN2SR_PYTHON)" >&2
    exit 1
fi

mkdir -p "$models" tests/data

base=https://github.com/mv-lab/swin2sr/releases/download/v0.0.1

# task:checkpoint:converted name - the task name is only used for the fixture
# header, the converter derives the architecture from the tensors themselves.
convert() {
    task=$1
    pth=$2
    name=$3
    if [ ! -f "$models/$pth" ]; then
        echo "downloading $pth"
        curl -fsSL "$base/$pth" -o "$models/$pth"
    fi
    "$python" tools/convert.py "$models/$pth" "$models/swin2sr-$name.safetensors"
}

convert classical_sr  Swin2SR_ClassicalSR_X4_64.pth                      classical-x4
convert classical_sr  Swin2SR_ClassicalSR_X2_64.pth                      classical-x2
convert real_sr       Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth          realworld-x4
convert lightweight_sr Swin2SR_Lightweight_X2_64.pth                     lightweight-x2
# The compressed model is x4 and 48-px-registered; its head produces a SECOND
# output image, which its fixture carries (see `src/fixture.rs`'s FLAG_AUX).
convert compressed_sr Swin2SR_CompressedSR_X4_48.pth                      compressed-x4

# The fixtures are the accuracy record: a seeded input through the published
# network, written as the raw f32 planes the engine compares against with
# --verify. Regenerating them from the converted checkpoint is how a fresh clone
# reproduces the numbers in the README rather than trusting them.
fixture() {
    pth=$1
    task=$2
    scale=$3
    out=$4
    # The 5th argument is the checkpoint's REGISTERED img_size, and it only has to
    # be given when it is not 64: it sets the shape of the stored `attn_mask`
    # buffers, so a mismatch is a hard `load_state_dict` failure. The compressed
    # checkpoint was registered at 48.
    patch=${5:-64}
    "$python" tools/make_fixture.py --model "$models/$pth" --task "$task" --scale "$scale" \
        --patch_size "$patch" --h 37 --w 29 --seed 1 --out "tests/data/$out.bin"
}

fixture Swin2SR_ClassicalSR_X4_64.pth             classical_sr  4 classical_x4
fixture Swin2SR_ClassicalSR_X2_64.pth             classical_sr  2 classical_x2
fixture Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth real_sr       4 realworld_x4
fixture Swin2SR_Lightweight_X2_64.pth             lightweight_sr 2 lightweight_x2
fixture Swin2SR_CompressedSR_X4_48.pth            compressed_sr 4 compressed_x4 48

echo
echo "converted into $models, fixtures in tests/data:"
echo "    cargo test --release                       # everything, CUDA feature built in"
