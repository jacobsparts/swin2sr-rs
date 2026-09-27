// swin2sr's own kernels: the four that are this engine's and not the toolkit's.
//
// THE SPLIT IS lightgpu's PROMOTION TEST (docs/MAINTAINING.md). A toolkit kernel
// is one another model family could call unchanged. What is here is what the
// window-versus-image duality forces onto the consumer: the two window index maps
// that depend on this model's shift schedule, the attention kernel whose per-window
// tables and shift mask the converter writes, and the pixel-shuffle of the head -
// the toolkit has the INVERSE of that one (`lg_pixel_unshuffle2`) and no forward
// form.
//
// EVERYTHING ELSE IS THE TOOLKIT'S, AND THAT IS A CHANGE. This file used to carry
// `ss_conv3x3`, `ss_linear`, `ss_conv1x1` and `ss_upsample2x_nearest` as well, and
// all four were the same ops as `lg_conv3x3s1p1` / `lg_conv3x3_winograd`,
// `lg_linear`, `lg_conv1x1` and `lg_upsample2x_nearest` - which is exactly the
// duplication the promotion test exists to prevent. Two of them were also
// measurably worse: the toolkit's tiled matmul is 10x the per-thread one it
// replaced, and its F(4,3) is 36x the direct 3x3 on this graph. What is left is a
// kernel set with one job each, and a graph that calls the toolkit for everything
// the toolkit does.
//
// NCHW THROUGHOUT. An activation is `[c][h][w]`; the token layout `[tokens][c]`
// appears only inside a block, and `[3][tokens][c]` only for the fused qkv. This
// is the layout the CPU backend uses, so `--cuda-selftest` can compare the two
// backends op for op rather than through a transpose.
//
// THE INDEX MAP IS THE DANGEROUS PART. `ss_window_gather` and `ss_window_scatter`
// are declared inverses of each other, and a pair that is wrong in the SAME way
// is still mutually inverse - the CPU backend shipped exactly that bug (`(y*wp+x)*c`
// where the channel stride is `hp*wp`) and every self-consistency check passed
// while the projections were fed a permutation of the right numbers. What catches
// it is not the pair, it is the golden fixture. Both kernels below therefore take
// the same `ss_pos` helper, so a mistake in it moves BOTH sides and cannot hide
// behind disagreement between them.

// The largest attention window this kernel's per-thread logit array can hold.
// Swin2SR's window is 8, so n = 64; the ceiling is checked against the plan at
// launch rather than silently overflowing.
#define SS_MAX_N 64

// The largest head width the query row and the output accumulator can hold. Every
// released checkpoint has head_dim = 30; the ceiling is checked against the
// checkpoint at launch, like SS_MAX_N.
#define SS_MAX_HD 64

// The largest head count the key-norm table can be sized for. `heads` arrives as a
// kernel argument, so the shared array's extent has to be a compile-time constant
// and the count has to be checked against it at launch - the same bargain as
// SS_MAX_N. Every released checkpoint has 6 heads.
#define SS_MAX_HEADS 12

// The plane coordinate of window token `t` of window `wi`, and the token order
// inside a window.
//
// THE SIGN OF THE SHIFT IS THE WHOLE TRICK. The reference rolls the plane by
// `-shift` and then partitions it; the same result is obtained by partitioning
// the UNROLLED plane at `+shift`, which is what this does. Writing `-shift` here
// is the natural mistake and it is not symmetric with the mask, which is defined
// on the rolled plane - the engine's first device version had it, and the gather
// selftest caught it at shift=4 while shift=0 agreed.
//
// The modulo is a WRAP, as `torch.roll` is: the shifted window attention reads
// the plane cyclically, and the `-100` mask is what stops the wrapped part from
// contributing.
__device__ __forceinline__ void ss_pos(int wi, int t, int nww, int win, int hp, int wp,
                                       int shift, int *py, int *px)
{
    const int wh = wi / nww;
    const int ww = wi % nww;
    const int i = t / win;
    const int j = t % win;
    *py = (wh * win + i + shift) % hp;
    *px = (ww * win + j + shift) % wp;
}

// The shift mask's region id along one axis, on the ROLLED plane: `_compute_mask`
// builds `h_slices = (slice(0, -win), slice(-win, -shift), slice(-shift, None))`
// and tags each band 0, 1, 2, then adds `w_slices` (0, 3, 6, with 9 as the "both
// sides wrapped" marker). Two tokens that end up with different tags are masked.
__device__ __forceinline__ int ss_region(int a, int len, int win, int shift)
{
    if (a + win < len) return 0;
    if (a + shift < len) return 1;
    return 2;
}

// ---------------------------------------------------------------------------
// The window pair.
// ---------------------------------------------------------------------------

// Read the plane into the token layout, optionally normalising each token. `x` is
// NCHW: the channel stride is `hp*wp`. If `norm_w` is null the token is a copy.
extern "C" __global__ void ss_window_gather(
    const float *__restrict__ x, const float *__restrict__ norm_w,
    const float *__restrict__ norm_b, float *__restrict__ tok,
    int nw, int n, int nww, int win, int hp, int wp, int c, int shift, float eps)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)nw * n * c;
    if (idx >= total) return;
    const int ch = (int)(idx % c);
    const long t2 = idx / c;
    const int t = (int)(t2 % n);
    const int wi = (int)(t2 / n);

    int y, xx;
    ss_pos(wi, t, nww, win, hp, wp, shift, &y, &xx);
    const int p = y * wp + xx;
    const size_t hw = (size_t)hp * wp;
    float *dst = tok + (size_t)(wi * n + t) * c;

    if (norm_w == 0) {
        dst[ch] = x[(size_t)ch * hw + p];
        return;
    }
    // nn.LayerNorm over the channel axis: the two-pass variance the CPU twin and
    // the toolkit's `lg_channel_layer_norm` also use, so the three agree to
    // rounding rather than to a different summation order.
    float s1 = 0.f, s2 = 0.f;
    for (int i = 0; i < c; ++i) {
        const float v = x[(size_t)i * hw + p];
        s1 += v;
        s2 += v * v;
    }
    const float mean = s1 / (float)c;
    const float var = s2 / (float)c - mean * mean;
    const float rstd = rsqrtf(fmaxf(var, 0.f) + eps);
    dst[ch] = (x[(size_t)ch * hw + p] - mean) * rstd * norm_w[ch] + norm_b[ch];
}

// The inverse of the gather: `window_reverse` plus the undo of the roll.
extern "C" __global__ void ss_window_scatter(
    const float *__restrict__ tok, float *__restrict__ x,
    int nw, int n, int nww, int win, int hp, int wp, int c, int shift)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)nw * n * c;
    if (idx >= total) return;
    const int ch = (int)(idx % c);
    const long t2 = idx / c;
    const int t = (int)(t2 % n);
    const int wi = (int)(t2 / n);

    int y, xx;
    ss_pos(wi, t, nww, win, hp, wp, shift, &y, &xx);
    const size_t hw = (size_t)hp * wp;
    x[(size_t)ch * hw + (size_t)y * wp + xx] = tok[(size_t)(wi * n + t) * c + ch];
}

// ---------------------------------------------------------------------------
// Window attention: cosine, a relative-position bias, and the shift mask.
// ---------------------------------------------------------------------------

// ONE BLOCK PER WINDOW, one thread per (query, head) pair: blockDim is
// (n, heads), so this model's block is 64x6 = 384 threads and 12 warps.
// THAT SHAPE IS THE WHOLE POINT. The first version launched one block per
// (window, head) - 64 threads, two warps - and measured 11.6 ms per launch against
// an arithmetic ideal of 0.16 ms: two warps cannot hide the latency of a chain of
// dependent FMAs and `expf`, and the SM sits idle waiting on them. Nothing about
// the work changed; the same 110,976 threads now arrive in 289 blocks instead of
// 1734, six times as many warps at a time.
//
// THE ROW STAYS IN ONE THREAD ON PURPOSE. A block-wide softmax needs the logits in
// shared memory, two tree reductions and three `__syncthreads()`, and it makes the
// summation order depend on the block size - so the device result would differ from
// the CPU twin's by more than rounding, and `--cuda-selftest` could only compare
// them with a looser tolerance than it compares everything else with. A row is
// `n * (2*head_dim + 3)` operations: for this model (n = 64, head_dim = 30) about
// 4k, which a thread does in the time a sync costs. Revisit only with a profile in
// hand - and if it is revisited, keep THIS kernel for the selftest, the way the
// toolkit keeps `lg_conv1x1` for nafnet's.
//
// The `n` logits per query live in a local array: 64 floats, indexed by a runtime
// `k`, so nvcc puts them in local memory (L1-cached). Nothing else in this kernel
// touches memory per key beyond the qkv rows.
extern "C" __global__ void ss_attention(
    const float *__restrict__ qkv, const float *__restrict__ logit_scale,
    const float *__restrict__ cpb, float *__restrict__ out,
    int nw, int n, int nww, int win, int hp, int wp, int heads, int head_dim, int shift)
{
    const int wi = blockIdx.x;
    const int c = heads * head_dim;
    const size_t tok_n = (size_t)n * nw;
    const float *Q = qkv;
    const float *K = qkv + tok_n * c;
    const float *V = qkv + 2 * tok_n * c;
    const int masked = shift > 0;

    // One row of `kns` per head: the block spans every head, so the norms cannot
    // be shared across it.
    __shared__ float kns[SS_MAX_HEADS][SS_MAX_N];
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        for (int h = 0; h < heads; ++h) {
            const float *krow = K + ((size_t)wi * n + k) * c + h * head_dim;
            float kn = 0.f;
            for (int d = 0; d < head_dim; ++d) kn += krow[d] * krow[d];
            kn = sqrtf(kn);
            kns[h][k] = kn < 1e-12f ? 1e-12f : kn;
        }
    }
    __syncthreads();

    {
        const int q = threadIdx.x;
        const int h = threadIdx.y;
        const int off = h * head_dim;
        // The reference clamps the learned temperature at log(100) and
        // exponentiates it, and there is NO 1/sqrt(head_dim) - cosine attention is
        // already normalised.
        const float ls = expf(fminf(logit_scale[h], 4.605170185988092f));
        const float *qrow = Q + ((size_t)wi * n + q) * c + off;
        // THE QUERY ROW GOES INTO REGISTERS FIRST. `Q` is [tokens][c], so
        // consecutive queries are `c` floats apart and one warp's 32 reads of
        // `qrow[d]` land on 32 different cache lines - and the loop below read it
        // again for every key, 64 times over. Copied once here, the whole row is
        // 30 registers and the only global reads left in the key loop are of `K`
        // and `V`, which every thread in the warp reads at the SAME address (the
        // row depends on the key and the dimension, not on the query) and which
        // the hardware therefore broadcasts.
        float qr[SS_MAX_HD];
        float qn = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            qr[d] = qrow[d];
            qn += qr[d] * qr[d];
        }
        qn = sqrtf(qn);
        if (qn < 1e-12f) qn = 1e-12f;
        const int rqy = masked ? ss_region((wi / nww) * win + q / win, hp, win, shift) : 0;
        const int rqx = masked ? ss_region((wi % nww) * win + q % win, wp, win, shift) : 0;

        float logits[SS_MAX_N];
        float mx = -INFINITY;
        for (int k = 0; k < n; ++k) {
            const float *krow = K + ((size_t)wi * n + k) * c + off;
            // Four accumulators: one FMA chain is 4 cycles of latency each and the
            // row is only 30 long, so a single chain leaves the pipe empty.
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            int d = 0;
            for (; d + 4 <= head_dim; d += 4) {
                a0 += qr[d] * krow[d];
                a1 += qr[d + 1] * krow[d + 1];
                a2 += qr[d + 2] * krow[d + 2];
                a3 += qr[d + 3] * krow[d + 3];
            }
            for (; d < head_dim; ++d) a0 += qr[d] * krow[d];
            const float dot = ((a0 + a1) + (a2 + a3)) / kns[h][k];
            float s = dot * (ls / qn) + cpb[(q * n + k) * heads + h];
            if (masked) {
                const int rky = ss_region((wi / nww) * win + k / win, hp, win, shift);
                const int rkx = ss_region((wi % nww) * win + k % win, wp, win, shift);
                if (rqy != rky || rqx != rkx) s -= 100.f;
            }
            logits[k] = s;
            mx = fmaxf(mx, s);
        }
        // Max-subtracted, as torch's softmax is, so a row of large logits cannot
        // overflow where the reference would not. The normalisation is folded into
        // the weights once (`logits[k] * (1/sum)`) rather than divided out again
        // inside the weighted sum, which turned 30 divisions per key into one
        // multiplication per key - and it is what the CPU twin does too.
        float sum = 0.f;
        for (int k = 0; k < n; ++k) {
            logits[k] = expf(logits[k] - mx);
            sum += logits[k];
        }
        const float inv_sum = 1.f / sum;
        // THE WEIGHTED SUM IS KEY-OUTER, DIM-INNER. The other order - one output
        // dimension at a time, sweeping the keys - reads `V[k][d]` with a stride of
        // `c` floats, so a thread touches 64 cache lines for every one of its 30
        // dimensions and the whole 46 KB head block has to stay resident to avoid
        // re-fetching it. This way each key's 30 floats are ONE contiguous run, read
        // once, and the accumulator is 30 registers' worth of local memory that
        // stays in L1.
        float oacc[SS_MAX_HD];
        for (int d = 0; d < head_dim; ++d) oacc[d] = 0.f;
        for (int k = 0; k < n; ++k) {
            const float w = logits[k] * inv_sum;
            const float *vrow = V + ((size_t)wi * n + k) * c + off;
            for (int d = 0; d < head_dim; ++d) oacc[d] += w * vrow[d];
        }
        float *orow = out + ((size_t)wi * n + q) * c + off;
        for (int d = 0; d < head_dim; ++d) orow[d] = oacc[d];
    }
}

// ---------------------------------------------------------------------------
// The reconstruction head's resampling.
// ---------------------------------------------------------------------------

// `F.pixel_shuffle(x, 2)`: [4C][H][W] -> [C][2H][2W]. The toolkit's
// `lg_pixel_unshuffle2` is the inverse of this and takes the same channel
// permutation, so a forward form could arguably be promoted next to it; until it
// is, this is the head's own (nafnet-rs keeps its own `nf_pixel_shuffle2` too).
//
// The upsampling the other heads need is NOT here: `nearest` is the toolkit's
// `lg_upsample2x_nearest`, and the direct and aux heads have no resampling at all.
extern "C" __global__ void ss_pixel_shuffle2(
    const float *__restrict__ in, float *__restrict__ out, int c4, int h, int wd)
{
    const int c = c4 / 4;
    const int oh = h * 2, ow = wd * 2;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)c * oh * ow;
    if (idx >= total) return;
    const int x = (int)(idx % ow);
    const long t = idx / ow;
    const int y = (int)(t % oh);
    const int ch = (int)(t / oh);
    const int dy = y & 1, dx = x & 1;
    const int sub = ch * 4 + dy * 2 + dx;
    out[idx] = in[((size_t)sub * h + (y >> 1)) * wd + (x >> 1)];
}


// ---------------------------------------------------------------------------
// The reconstruction head's last octave: the 3x3 conv AND the 2x shuffle.
// ---------------------------------------------------------------------------

// `ss_conv3x3_shuffle2`: the pixel-shuffle head's last octave, which is a 3x3
// convolution to `4F` channels followed by `F.pixel_shuffle(x, 2)`, fused into one
// pass.
//
// WHY IT IS FUSED. The intermediate `big[4F][h][w]` every element of which is read
// exactly once, by the shuffle, and at the head's last octave it is the single
// largest allocation in the engine - `4 * feat` channels over the whole padded
// plane, where `feat` is 64 of a 180-wide model. The CPU side of this engine
// already fuses the pair (`cpu::conv3x3_shuffle2`), where it removed 152 MB of a
// 380 MB peak; this is the same fusion on the device, and it also removes the
// shuffle's transposed global-memory read (the fusion reads the activation row it
// needs and writes it interleaved).
//
// THE OUTPUT IS THE SHUFFLE'S, NOT THE CONV'S: `out[f][2y + dy][2x + dx]` is
// conv channel `4f + 2dy + dx` at (y, x). The caller therefore writes both the even
// and the odd column of an output row from one thread, which is why `blockDim.x`
// is `wd` (one thread per INPUT column) and each thread produces two outputs.
//
// THE THREE SOURCE ROWS ARE STAGED IN TWO SHARED SLOTS. Output row `y` of the conv
// reads input rows `y - 1`, `y` and `y + 1`, and row `h` does not exist - the plan's
// padded height is already a window multiple, so the conv's own border rule (skip
// the taps that fall outside) is the only rule that applies; there is no extra
// padding row to read. So a thread cannot gather all three rows into shared memory
// at once. Instead the block stages rows `y - 1` and `y`, takes the ky = 0 and
// ky = 1 taps from them, then OVERWRITES the `y - 1` slot with row `y + 1` behind a
// `__syncthreads` and takes the ky = 2 taps - reading the row through a clamped
// index so that a tap at the top or bottom edge multiplies a row that is in fact
// present (that is the conv's border rule; the multiplication is legal because the
// tap's weight contribution is what is being dropped, not the value).
//
// The row `y - 1` loaded at the top of a block is the REFLECTED padding of the
// plane, not the conv's zero padding: the caller passes `x` as the padded input
// plane (`lg_reflect_pad`'s output), so `x[-1] == x[1]` is already true and the
// staging here needs no edge special case. That is the same convention
// `lg_conv3x3s1p1` has, and the reason a mismatched pad would show up as a border
// ring in the output rather than as a uniform shift.
//
// Limits, checked by the caller: `2 * wd <= 1024` (one output row per block) and
// `c_in <= SS_SHS_CI`. The plan's padded width for the largest image this engine
// can run is 1080, so the first is 2160 > 1024 - the head's LAST octave is run
// row-blocked by the caller instead. See `gpu.rs`.
// Columns per thread: four keeps the block at `ceil(wd / 4)` threads, under the
// 1024 a block has, for every padded width this engine can plan (1080 at the
// largest image, i.e. 270 threads).
#define SS_SHS_COLS 4

extern "C" __global__ void ss_conv3x3_shuffle2(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ bias, float *__restrict__ out,
    int c_in, int feat, int h, int wd, int chunk)
{
    // One block per (f, y2) output row; one thread per input column.
    const int rows = 2 * h;
    const int f = blockIdx.x / rows;
    const int y2 = blockIdx.x % rows;
    const int dy = y2 & 1, y = y2 >> 1;
    const int tx = threadIdx.x;
    const int ow = 2 * wd;
    // The two conv channels this output row interleaves, in the shuffle's order.
    const int co0 = 4 * f + 2 * dy;
    const int co1 = co0 + 1;
    // A STAGED ROW IS WIDER THAN THE DATA: `SS_SHS_COLS * blockDim.x + 2` columns,
    // not `wd + 2`. The j-th accumulator's taps reach column
    // `(SS_SHS_COLS-1)*blockDim.x + 2`, which is past `wd` whenever the block does
    // not divide `wd` - and the whole stage is then zero-padded, so those columns
    // of a READ are the conv's zero border and the guarded STORE simply never
    // writes them. Reading them unstaged (which is what sizing the row by `wd`
    // does) walks off the row into the next channel's stage and off the end of the
    // allocation on the last slot - an illegal-address fault, and the shape that
    // hit it was every width the fixtures do not use.
    const size_t row = (size_t)(SS_SHS_COLS * blockDim.x + 2);

    extern __shared__ float smem[];
    float *a_row = smem;                            // centre row y
    float *b_row = a_row + (size_t)chunk * row;     // row y + 1
    float *c_row = b_row + (size_t)chunk * row;     // row y - 1
    float *bs = c_row + (size_t)chunk * row;        // the two biases

    if (tx == 0) {
        bs[0] = bias[co0];
        bs[1] = bias[co1];
    }

    // THE CHANNEL REDUCTION IS SPLIT INTO CHUNKS, because the three stage rows are
    // `3 * c_in * (wd + 2)` floats and the head's last octave is 64 wide over a
    // 66-wide row - 50 KB, past the 48 KB a launch may take without the driver
    // opt-in that this engine's launch layer does not expose. Each pass stages
    // `chunk` channels and accumulates into the same two registers, in the same
    // (ci, ky, kx) order, so the sum is unchanged: a chunk boundary adds no
    // rounding, since the accumulation is over the same sequence of terms.
    //
    // THE ROWS ARE STAGED THREE-AT-A-TIME, one slot each, because the conv's border
    // rule is per tap: `ky = 0` reads row `y - 1` and is SKIPPED at `y == 0`, `ky =
    // 1` reads row `y`, `ky = 2` reads row `y + 1` and is skipped at `y + 1 == h`.
    // The missing row is written as ZEROS so its taps contribute nothing.
    // Clamping it into a neighbour's slot instead - which is what a two-slot scheme
    // does - is NOT the same computation: it counts the centre row twice at the top
    // edge, and that was the first version of this kernel.
    //
    // The caller has already reflect-padded the plane, so `x`'s own edges are what
    // the conv's zero padding would be at the image border, and nothing here needs
    // an edge special case beyond the two skips.
    // EACH THREAD CARRIES `SS_SHS_COLS` COLUMNS' ACCUMULATORS, strided by the
    // block, so `blockDim.x` need not equal `wd`: the head's last octave reaches
    // 656 columns of padded plane at a 648-pixel image, and one thread per column
    // would need 1312 threads to write the interleaved row. Four columns a thread
    // with `blockDim.x = ceil(wd / 4)` keeps the block under 1024 threads at any
    // width this engine can plan, and the store below is a strided loop for the
    // same reason.
    float acc0[SS_SHS_COLS], acc1[SS_SHS_COLS];
    #pragma unroll
    for (int j = 0; j < SS_SHS_COLS; ++j) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
    for (int c0 = 0; c0 < c_in; c0 += chunk) {
        const int n = (c0 + chunk < c_in) ? chunk : (c_in - c0);
        __syncthreads();
        for (int i = 0; i < n; ++i) {
            const int ci = c0 + i;
            float *ar = a_row + (size_t)i * row;
            float *br = b_row + (size_t)i * row;
            float *cr = c_row + (size_t)i * row;
            const float *src_a = in + ((size_t)ci * h + y) * wd;
            const float *src_b = in + ((size_t)ci * h + (y + 1 < h ? y + 1 : y)) * wd;
            const float *src_c = in + ((size_t)ci * h + (y > 0 ? y - 1 : 0)) * wd;
            for (int x = tx; x < (int)row; x += blockDim.x) {
                ar[x] = 0.0f;
                br[x] = 0.0f;
                cr[x] = 0.0f;
            }
            for (int x = tx; x < wd; x += blockDim.x) {
                ar[x + 1] = src_a[x];
                br[x + 1] = (y + 1 < h) ? src_b[x] : 0.0f;
                cr[x + 1] = (y > 0) ? src_c[x] : 0.0f;
            }
        }
        __syncthreads();

        const float *arow = a_row + tx;
        const float *brow = b_row + tx;
        const float *crow = c_row + tx;
        for (int i = 0; i < n; ++i) {
            const int ci = c0 + i;
            const float *w0 = w + ((size_t)co0 * c_in + ci) * 9;
            const float *w1 = w + ((size_t)co1 * c_in + ci) * 9;
            #pragma unroll
            for (int j = 0; j < SS_SHS_COLS; ++j) {
                const float sa0 = arow[j * blockDim.x], sa1 = arow[j * blockDim.x + 1],
                            sa2 = arow[j * blockDim.x + 2];
                const float sb0 = brow[j * blockDim.x], sb1 = brow[j * blockDim.x + 1],
                            sb2 = brow[j * blockDim.x + 2];
                const float sc0 = crow[j * blockDim.x], sc1 = crow[j * blockDim.x + 1],
                            sc2 = crow[j * blockDim.x + 2];
                acc0[j] += w0[0] * sc0 + w0[1] * sc1 + w0[2] * sc2
                         + w0[3] * sa0 + w0[4] * sa1 + w0[5] * sa2
                         + w0[6] * sb0 + w0[7] * sb1 + w0[8] * sb2;
                acc1[j] += w1[0] * sc0 + w1[1] * sc1 + w1[2] * sc2
                         + w1[3] * sa0 + w1[4] * sa1 + w1[5] * sa2
                         + w1[6] * sb0 + w1[7] * sb1 + w1[8] * sb2;
            }
            arow += row;
            brow += row;
            crow += row;
        }
    }

    // Interleave: even output column from co0, odd from co1. The store is
    // STRIDED over the same column set the accumulators were, so `blockDim.x`
    // divides the work rather than defining it.
    const size_t orow = ((size_t)f * rows + y2) * ow;
    #pragma unroll
    for (int j = 0; j < SS_SHS_COLS; ++j) {
        const int x = tx + j * blockDim.x;
        if (x < wd) {
            out[orow + 2 * x] = acc0[j] + bs[0];
            out[orow + 2 * x + 1] = acc1[j] + bs[1];
        }
    }
}
