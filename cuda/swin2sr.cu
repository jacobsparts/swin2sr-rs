// swin2sr's own kernels: the four that are this engine's and not the toolkit's.
//
// THE SPLIT IS lightgpu's PROMOTION TEST (docs/MAINTAINING.md). A toolkit kernel
// is one another model family could call unchanged. What is here is what the
// window-versus-image duality forces onto the consumer: the two window index maps
// that depend on this model's shift schedule, the attention kernel whose per-window
// tables and shift mask the converter writes, and the FUSED 3x3-plus-shuffle of the
// head's last octave, which is a fusion rather than an op.
//
// THE HEAD'S PLAIN PIXEL SHUFFLE IS NO LONGER ONE OF THEM. It was
// `ss_pixel_shuffle2`, and it duplicated what two other engines had written
// privately - and the toolkit's CONVENTIONS.md already recorded the inversion
// (`lg_pixel_unshuffle2` with no forward form) as a gap rather than a rule. It is
// now the toolkit's `lg_pixel_shuffle`, whose `r` is a runtime argument; that the
// head used to REFUSE any scale but 2 was a consequence of the duplicate, not of
// the model. Verified bit-identical against this engine's copy before removal.
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
    // THE MASK BAND, computed once per token instead of once per (query, key, head).
    // `ss_region` is two comparisons but the divides that feed it (`i / win`,
    // `i % win`, `wi / nww`, `wi % nww`) are not free, and the old loop called it
    // four times per (query, key) pair - 2.4M divisions per launch on a kernel whose
    // arithmetic is a rounding error next to its own latency. A token's band is a
    // property of the token, so ONE table serves both roles: `kry[i]`/`krx[i]` are
    // read as the key's band and as the query's band.
    __shared__ signed char kry[SS_MAX_N], krx[SS_MAX_N];
    // THE BIAS TABLE IS PRE-TRANSPOSED BY THE HOST, to [head][query][key]. It cannot
    // be staged here: `cpb` is [query][key][head] and the transposed form is
    // heads*n*n floats, 196 KB at this model's shape against a 48 KB static limit.
    // The host does it once per upload instead (`gpu::transpose_cpb`), which is the
    // same re-indexing `cpu::attention` documents: the old read of
    // `cpb[(q * n + k) * heads + h]` walks with a stride of `heads` floats, so a
    // warp - 32 consecutive queries of one head - touches 32 cache lines per key and
    // covers the whole 98 KB table once per key row; transposed, one head's 16 KB
    // plane is contiguous and stays in L1 for the whole query loop. Same numbers,
    // same order, same accumulation: bit-identical.
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < n * heads;
         i += blockDim.x * blockDim.y) {
        const int k = i % n, h = i / n;
        const float *krow = K + ((size_t)wi * n + k) * c + h * head_dim;
        float kn = 0.f;
        for (int d = 0; d < head_dim; ++d) kn += krow[d] * krow[d];
        kn = sqrtf(kn);
        kns[h][k] = kn < 1e-12f ? 1e-12f : kn;
    }
    if (masked) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            kry[i] = (signed char)ss_region((wi / nww) * win + i / win, hp, win, shift);
            krx[i] = (signed char)ss_region((wi % nww) * win + i % win, wp, win, shift);
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
        const int rqy = masked ? kry[q] : 0;
        const int rqx = masked ? krx[q] : 0;

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
            float s = dot * (ls / qn) + cpb[((size_t)h * n + q) * n + k];
            if (masked && (rqy != kry[k] || rqx != krx[k])) s -= 100.f;
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
        // THE WEIGHTED SUM IS DIM-OUTER, KEY-INNER, AND THAT ORDER IS MEASURED, NOT
        // PREFERRED. This used to be key-outer with one serial accumulator per output
        // dimension: a 64-long dependent FMA chain per `d`, whose operand loads are
        // 720 bytes apart, for every one of 30 dimensions. An experiment that skipped
        // the pass entirely put ss_attention at 2469 ms against 5822 ms - so 3353 ms,
        // 58% of the kernel, was this loop and not the dot products or the softmax
        // (replacing `expf` with a subtract changed nothing).
        //
        // The CPU twin has always done it this way: `cpu::attention` sweeps `d` outer
        // and `k` inner in groups of FOUR, with a tail, and reduces
        // `((a0+a1)+(a2+a3)) + tail`. Doing the same here is therefore not a
        // tolerance-eating change - it makes the device's summation order IDENTICAL
        // to the reference the selftest compares it against, where before it was
        // merely close. `w` is the same weight, the products are the same, and the
        // single multiply by `1/sum` still happens once per element at the end.
        float *orow = out + ((size_t)wi * n + q) * c + off;
        for (int d = 0; d < head_dim; ++d) {
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            int k = 0;
            for (; k + 4 <= n; k += 4) {
                a0 += logits[k] * V[((size_t)wi * n + k) * c + off + d];
                a1 += logits[k + 1] * V[((size_t)wi * n + k + 1) * c + off + d];
                a2 += logits[k + 2] * V[((size_t)wi * n + k + 2) * c + off + d];
                a3 += logits[k + 3] * V[((size_t)wi * n + k + 3) * c + off + d];
            }
            float tail = 0.f;
            for (; k < n; ++k) tail += logits[k] * V[((size_t)wi * n + k) * c + off + d];
            orow[d] = ((a0 + a1) + (a2 + a3) + tail) * inv_sum;
        }
    }
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
// THE THREE SOURCE ROWS ARE STAGED IN THREE SHARED SLOTS, one row each, and the
// missing one at either plane edge is written as ZEROS so its taps contribute
// nothing. Output row `y` of the conv reads input rows `y - 1`, `y` and `y + 1`;
// row `-1` and row `h` do not exist, and the conv's border rule is to skip those
// taps. Clamping a missing row into a neighbour's slot instead - which is what an
// earlier two-slot scheme did - is NOT the same computation: it counts the centre
// row twice at the top edge. The caller has already reflect-padded the plane, so
// the plane's own edges are what the conv's zero padding would be at the image
// border, and the staging here needs no edge special case beyond the two skips.
//
// Limits, checked by the caller: `blockDim.x` is `ceil(wd / 4)` (SS_SHS_COLS
// columns per thread), so the block stays under the 1024 a block has for every
// padded width this engine can plan (1080 at the largest image, i.e. 270 threads),
// and the channel reduction is chunked by the caller so the three stage rows fit
// the 48 KB a launch may take without the driver's opt-in. See `gpu.rs`.
//
// COLUMNS PER THREAD: four. One thread per column needs `2 * wd <= 1024` and
// refuses every pixel-shuffle checkpoint above a 480-pixel input. Four.
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
            // ONE THREAD PER STAGED COLUMN, and that column's ONLY writer. Zeroing
            // the whole row in one loop and then filling `x + 1` in a second loop
            // puts two DIFFERENT threads' writes on the same shared address in an
            // order nothing orders: the thread that owns column `x` filled it
            // before the barrier, and the thread that owns `x + 1` may zero it
            // before or after that fill. The `__syncthreads()` below orders the
            // staging against the ACCUMULATION, not the staging against itself.
            //
            // THE BUG WAS INVISIBLE BELOW A WARP: with `blockDim.x <= 32` the block
            // is one warp, whose lanes issue together, so the two loops interleave
            // the same way every run and every width agreed to one level. The
            // measured boundary is exactly there and nowhere else - `blockDim.x` is
            // `ceil(wd / 4)`, so a plane up to 128 is one warp and agrees, and every
            // plane above 128 (block >= 33) diverges. That is why the fixtures, which
            // are 33 wide, never saw it, and why it showed up first as run-to-run
            // non-determinism at image sizes.
            for (int x = tx; x < (int)row; x += blockDim.x) {
                const int sx = x - 1;
                if (sx >= 0 && sx < wd) {
                    ar[x] = src_a[sx];
                    br[x] = (y + 1 < h) ? src_b[sx] : 0.0f;
                    cr[x] = (y > 0) ? src_c[sx] : 0.0f;
                } else {
                    ar[x] = 0.0f;
                    br[x] = 0.0f;
                    cr[x] = 0.0f;
                }
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
