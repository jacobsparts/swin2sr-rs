//! The compressed_sr head's bicubic pre-upsample, checked against torch.
//!
//! `F.interpolate(..., mode="bicubic", align_corners=False)` is applied to the
//! PADDED input plane and resized to the UNPADDED output grid, so the ratio is
//! not an integer (40x32 -> 148x116 for a 37x29 image) and the kernel has to be
//! torch's to the last bit: its output is added to the body's before the final
//! convolution, so a different resample is a different image.
//!
//! The reference arrays are produced by torch and committed under tests/data in
//! the same raw-f32 form the fixtures use. `.npy` is a 128-byte header (v1.0) or
//! a longer one (v2.0+); this reads the header length rather than assuming it.
use swin2sr::cpu::bicubic_resize;

fn read_npy(path: &str) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let major = bytes[6];
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize)
    } else {
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12usize)
    };
    let header = std::str::from_utf8(&bytes[hstart..hstart + hlen]).unwrap();
    let shape_at = header.find("'shape':").expect("shape in .npy header");
    let open = header[shape_at..].find('(').unwrap() + shape_at + 1;
    let close = header[open..].find(')').unwrap() + open;
    let shape: Vec<usize> = header[open..close]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data: Vec<f32> = bytes[hstart + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, data)
}

#[test]
fn bicubic_matches_torch_on_the_padded_plane() {
    let (si, x) = read_npy("tests/data/bicubic_in_40x32.npy");
    let (so, want) = read_npy("tests/data/bicubic_out_148x116.npy");
    assert_eq!(si, vec![1, 1, 40, 32]);
    assert_eq!(so, vec![1, 1, 148, 116]);
    let mut got = vec![0.0f32; 148 * 116];
    bicubic_resize(&x, 1, 40, 32, 148, 116, &mut got);
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    println!("bicubic 40x32 -> 148x116: max |diff| {worst:e} at {at} (got {}, want {})",
             got[at], want[at]);
    assert!(worst < 2e-5, "max |diff| {worst:e}");
}
