// Vendored flash-attention prefill kernel, derived from candle's vendored MLX
// steel attention (candle-metal-kernels scaled_dot_product_attention.metal,
// MIT license, "Updated from MLX commit f70764a"; MLX is Copyright © 2023-2024
// Apple Inc.). The block structure, tile sizes (BQ=32 / BK=16 / BD=128,
// WM=4 / WN=1), online-softmax exp2 trick and accumulation order are steel's
// EXACTLY — the numerics goal is value-identity with candle's float32 steel
// kernel (`steel_attention_float32_bq32_bk16_bd128_wm4_wn1_maskfloat32`), which
// the flash.rs tests measure against the composed f32 sdpa reference.
//
// Deviations from the steel template, all decided up front:
//
//  - ONE dtype configuration instead of the template T: Q is read from device
//    as float into FLOAT shared memory (candle's f16 kernel would round the
//    rope output); K and V are read from device as HALF into half shared
//    memory — the MMAFrag loads static_cast smem to AccumType=float, and since
//    the cached values are f16 the upcast is exact, so the mma consumes the
//    same float values candle's f32 kernel reads from host-widened tensors.
//    All MMA tiles, softmax stats and the O accumulator are float (as steel
//    already had); O is stored as float (no half store).
//
//  - The mask tensor input and MaskType machinery are deleted. Masking is
//    computed in-kernel from three args: `q_off` (absolute position of query
//    row 0), `k_off` (absolute position of key column 0) and `window` (the
//    sliding window; 0x7fffffff = full attention). Key column j is visible to
//    query row i iff
//        (j + k_off) <= (i + q_off)  AND  (i + q_off) - (j + k_off) < window.
//    Invisible pairs have their score set to -INFINITY before the
//    online-softmax max/exp — bit-equivalent to adding kv_cache.rs's additive
//    0/-inf mask (score + 0.0 and score + -inf differ from the assignment only
//    at -0.0 scores, where the downstream exp2/sum values are identical).
//    Out-of-range key columns of an unaligned last K block (c >= kL) are
//    folded into the same rule (steel's separate kL_rem branch).
//
//  - Block-level skip in the KV loop: a KV block is skipped entirely when it
//    is fully future (its oldest key is newer than the newest valid query) or
//    fully expired (its newest key is older than window-reach of the oldest
//    query). A processed all--inf block contributes exp terms of exactly 0, a
//    rescale factor of exactly 1 and no max update, so skipping is
//    bit-identical to processing (the flash.rs block-skip test proves it via
//    the `disable_skip` arg, which is test-only plumbing — production always
//    passes 0).
//
//  - Unaligned seq / K are handled by the ALIGN_Q / ALIGN_K template bools,
//    instantiated as four distinct [[host_name]] functions (steel used
//    MTLFunctionConstantValues; our compiled_pipeline does not do constants).
//    Out-of-bounds query rows are loaded zero-filled and never stored;
//    out-of-bounds key columns are masked by the c >= kL rule.
//
// Threadgroup memory (statically declared, no host set_threadgroup_memory):
//   Q tile  32 x (128+4) floats = 16896 B   (padQ = 16/sizeof(float) = 4)
//   K/V tile max((16+8)*128, 16*(128+8)) halves = 3072 halves = 6144 B
//   total 23040 B < 32768 B — the float Q tile fits without shrinking BK.
//
// Compiled math_mode(fast) like candle's kernels (candle builds its metallib
// with MTLMathMode::Fast); NO contract/reassociate pragmas — we want the same
// contraction decisions candle's compiler makes for the steel source.

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

#pragma METAL fp math_mode(fast)

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// ============ steel/utils/integral_constant.h (trimmed to what is used)

template <int val>
using Int = integral_constant<int, val>;

// ============ steel/utils/type_traits.h (trimmed)

template <typename T>
struct pointer_element {};

template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};

template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

// ============ steel/attn/loader.h — BlockLoaderT, verbatim

template <
    typename T,
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoaderT {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  /* Constructor */
  METAL_FUNC BlockLoaderT(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(src_ + bi * src_ld + bj) {}

  /* Apply operation to threadgroup without bound checking */
  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& op) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] =
            op.apply(dst[i * kDstStrRow + j * kDstStrCol]);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
      }
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    // Skip loading if thread has no valid reads
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
        }
      }
      return;
    }

    // Use fast thread memory for bound checks
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      // Make sure tmp_idx only contains valid indices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }

      // Read valid indices into tmp_val
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }

      // Zero out uneeded values
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }

      // Copy values to threadgroup memory
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    src += tile_stride;
  }
};

// ============ steel/attn/mma.h — BaseMMAFrag / MMATile / tile_matmad
// (trimmed to the members the attention kernel uses)

template <typename T, int kFragRows_, int kFragCols_>
struct BaseMMAFrag {
  static_assert(
      kFragRows_ == 8,
      "Only 8 x 8 fragment matrices are currently supported");
  static_assert(
      kFragCols_ == 8,
      "Only 8 x 8 fragment matrices are currently supported");
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;

  STEEL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST int kElemRows = 1;
  STEEL_CONST int kElemCols = 2;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;

  template <typename U>
  using dtype_mat_t = typename metal::simdgroup_matrix<U, kFragRows, kFragCols>;

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static constexpr short2 get_coord(ushort simd_lane_id
                                               [[thread_index_in_simdgroup]]) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <typename SrcPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  load(thread frag_type& dst, SrcPtrType src, StrX str_x, StrY str_y) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] = static_cast<T>(src[i * str_x.value + j * str_y.value]);
      }
    }
  }

  template <typename DstPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  store(const thread frag_type& src, DstPtrType dst, StrX str_x, StrY str_y) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * str_x + j * str_y.value] = static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void store_safe(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y.value] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename Atype, typename Btype, typename Ctype>
  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread dtype_frag_t<Atype>& A,
      thread dtype_frag_t<Btype>& B,
      thread dtype_frag_t<Ctype>& C) {
    mat_type D_mat;
    dtype_mat_t<Atype> A_mat;
    dtype_mat_t<Btype> B_mat;
    dtype_mat_t<Ctype> C_mat;

    reinterpret_cast<thread dtype_frag_t<Atype>&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread dtype_frag_t<Btype>&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread dtype_frag_t<Ctype>&>(C_mat.thread_elements()) = C;

    mma(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  template <typename Atype, typename Btype, typename Ctype>
  METAL_FUNC static constexpr void mma(
      thread mat_type& D,
      thread dtype_mat_t<Atype>& A,
      thread dtype_mat_t<Btype>& B,
      thread dtype_mat_t<Ctype>& C) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_reduce(
      thread const frag_type& inp_vals,
      thread T* reduced_vals) {
    T thr_reduce = Op::apply(inp_vals.x, inp_vals.y);

    T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
    qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

    T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
    sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

    reduced_vals[0] = Op::apply(reduced_vals[0], sgr_reduce);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_bin_op(
      thread frag_type& inp_vals,
      thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }
};

template <
    typename T,
    int kTileRows_,
    int kTileCols_,
    class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  STEEL_CONST int kFragRows = MMAFrag_t::kFragRows;
  STEEL_CONST int kFragCols = MMAFrag_t::kFragCols;
  STEEL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;

  STEEL_CONST int kTileRows = kTileRows_;
  STEEL_CONST int kTileCols = kTileCols_;

  STEEL_CONST int kRows = kTileRows * kFragRows;
  STEEL_CONST int kCols = kTileCols * kFragCols;

  STEEL_CONST int kNumFrags = kTileRows * kTileCols;
  STEEL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST int kRowsPerThread = kTileRows * MMAFrag_t::kElemRows;
  STEEL_CONST int kColsPerThread = kTileCols * MMAFrag_t::kElemCols;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags];

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kTileCols + j];
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread T vals[kRowsPerThread]) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread T vals[kRowsPerThread]) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(
                src[(i * kFragRows) * w_x * str_x +
                    (j * kFragCols) * w_y * str_y]),
            Int<str_x>{},
            Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void
  store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }
};

template <
    typename Dtype,
    typename Atype,
    typename Btype,
    typename Ctype,
    int M,
    int N,
    int K,
    class MMAFragD,
    class MMAFragA,
    class MMAFragB,
    class MMAFragC>
METAL_FUNC void tile_matmad(
    thread MMATile<Dtype, M, N, MMAFragD>& D,
    thread MMATile<Atype, M, K, MMAFragA>& A,
    thread MMATile<Btype, K, N, MMAFragB>& B,
    thread MMATile<Ctype, M, N, MMAFragC>& C) {
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short m_serp = m; //(n % 2) ? (M - 1 - m) : m;
      short n_serp = (m % 2) ? (N - 1 - n) : n;

      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMAFragD::mma(
            D.frag_at(m_serp, n_serp),
            A.frag_at(m_serp, k),
            B.frag_at(k, n_serp),
            C.frag_at(m_serp, n_serp));
      }
    }
  }
}

// ============ steel_attention.h — softmax ops + scale transform, verbatim

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}

  METAL_FUNC T apply(T x) const {
    return scale * x;
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct ExpSubOp {
  // Guard: when y (row max) is -inf, all scores in the row are -inf (entirely
  // masked). Return 0 instead of exp2(-inf - (-inf)) = exp2(NaN).
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return (y == -metal::numeric_limits<T>::infinity())
        ? T(0)
        : fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// ============ the attention kernel

// Matches dispatch.rs FlashAttnArgs (#[repr(C)]). Strides are ELEMENT strides
// (`_h` between heads, `_r` between sequence rows; the head_dim stride is 1 —
// the host enforces it). `window` is the sliding window (0x7fffffff = full
// attention); `disable_skip` forces the block-skip bounds open (test-only —
// production always passes 0; a "skipped" block processed under it is all -inf
// and contributes exactly nothing, which is what the block-skip test proves).
typedef struct {
    int gqa_factor;
    float scale;
    int NK;         // number of key blocks: ceil(kL / BK)
    int NQ_aligned; // qL / BQ (index of the unaligned last query block)
    int NK_aligned; // kL / BK (index of the unaligned last key block)
    int qL_rem;     // qL % BQ
    int kL_rem;     // kL % BK
    int kL;         // key count
    int q_off;      // absolute position of query row 0
    int k_off;      // absolute position of key column 0
    int window;     // sliding window (0x7fffffff = full attention)
    int disable_skip;
    int64_t q_stride_h;
    int64_t q_stride_r;
    int64_t k_stride_h;
    int64_t k_stride_r;
    int64_t v_stride_h;
    int64_t v_stride_r;
    int64_t o_stride_h;
    int64_t o_stride_r;
} flash_attn_params;

STEEL_CONST int FLASH_WINDOW_UNBOUNDED = 0x7fffffff;

template <bool ALIGN_Q, bool ALIGN_K>
[[kernel, max_total_threads_per_threadgroup(128)]] void flash_attn(
    constant flash_attn_params* params [[buffer(0)]],
    const device float* Q [[buffer(1)]],
    const device half* K [[buffer(2)]],
    const device half* V [[buffer(3)]],
    device float* O [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  constexpr int BQ = 32;
  constexpr int BK = 16;
  constexpr int BD = 128;
  constexpr int WM = 4;
  constexpr int WN = 1;

  // Move to correct block (batch is fixed at 1; grid is {NQ, H, 1}).
  Q += int64_t(tid.y) * params->q_stride_h +
      int64_t(tid.x) * BQ * params->q_stride_r;

  const int64_t kv_head_idx = int64_t(tid.y) / params->gqa_factor;

  O += int64_t(tid.y) * params->o_stride_h +
      int64_t(tid.x) * BQ * params->o_stride_r;

  // Absolute position range of this block's VALID query rows, and the
  // block-skip bounds derived from it. kb_lim drops fully-future key blocks
  // (steel's do_causal kb limit, generalized to q_off/k_off); kb_start drops
  // fully-expired ones under a sliding window. Both are exact: a skipped
  // block's scores would all be -inf, contributing 0 to every accumulator.
  const int q_rows =
      (!ALIGN_Q && int(tid.x) == params->NQ_aligned) ? params->qL_rem : BQ;
  const int q_lo = int(tid.x) * BQ + params->q_off;
  const int q_hi = int(tid.x) * BQ + (q_rows - 1) + params->q_off;

  int kb_lim = params->NK;
  {
    const int max_col = q_hi - params->k_off; // newest visible key column
    kb_lim = metal::min(params->NK, max_col / BK + 1);
  }
  int kb_start = 0;
  if (params->window != FLASH_WINDOW_UNBOUNDED) {
    const int min_col =
        q_lo - params->window + 1 - params->k_off; // oldest visible key column
    const int a = min_col - (BK - 1);
    if (a > 0) {
      kb_start = (a + BK - 1) / BK;
    }
  }
  if (params->disable_skip) {
    kb_start = 0;
    kb_lim = params->NK;
  }

  K += kv_head_idx * params->k_stride_h +
      int64_t(kb_start) * BK * params->k_stride_r;
  V += kv_head_idx * params->v_stride_h +
      int64_t(kb_start) * BK * params->v_stride_r;

  // Prepare threadgroup memory. Q is a float tile (rope output stays f32);
  // K/V are half tiles (the cache dtype) — the MMAFrag loads upcast exactly.
  constexpr short padQ = 16 / sizeof(float);
  constexpr short padK = 16 / sizeof(half);
  constexpr short padV = 16 / sizeof(half);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup float Q_smem[BQ * (BD + padQ)];
  threadgroup half KV_smem[tgp_mem_s];

  threadgroup float* Qs = Q_smem;
  // Ks and Vs deliberately alias the same threadgroup allocation (as in the
  // upstream steel kernel): within each KV block the K tile is fully consumed
  // by the QK^T matmul before the barrier that precedes the V load, so K and
  // V never coexist. tgp_mem_s is sized for the larger of the two tiles.
  threadgroup half* Ks = KV_smem;
  threadgroup half* Vs = KV_smem;

  // Prepare block loaders
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ float,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded in transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ half,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ half,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(
      Q, int(params->q_stride_r), Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(
      K, int(params->k_stride_r), Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, int(params->v_stride_r), Vs, simd_group_id, simd_lane_id);

  TransformScale<float> ts(static_cast<float>(params->scale * 1.44269504089));

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<float, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<float, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<float, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<float, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<float, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<float, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q blocks apply scale
  if (!ALIGN_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }
  loader_q.apply_inplace_op(ts);

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  float max_score[kRowsPT];
  float sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = -metal::numeric_limits<float>::infinity();
  }

  // Loop over KV seq length
  for (int kb = kb_start; kb < kb_lim; kb++) {
    // Load K block and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!ALIGN_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<float, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<half, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // In-kernel masking, replacing steel's mask tensor / do_causal / kL_rem
    // branches with ONE rule: key column c (absolute position c + k_off) is
    // invisible to query row r (absolute position r + q_off) when it is out of
    // range (c >= kL, the unaligned-K tail), strictly future, or expired
    // (>= window positions behind the query). Setting the score to -inf is
    // bit-equivalent to adding the additive 0/-inf mask. The edge tests keep
    // fully-interior blocks (every pair visible) out of the masking loop.
    {
      const bool k_tail = !ALIGN_K && kb == (params->NK_aligned);
      const bool future_edge =
          (kb * BK + (BK - 1) + params->k_off) > q_lo;
      const bool expired_edge = params->window != FLASH_WINDOW_UNBOUNDED &&
          (q_hi - (kb * BK + params->k_off)) >= params->window;

      if (k_tail || future_edge || expired_edge) {
        using stile_t = decltype(Stile);
        using selem_t = typename stile_t::elem_type;
        constexpr auto neg_inf = -metal::numeric_limits<selem_t>::infinity();

        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < stile_t::kTileRows; i++) {
          const int row_abs =
              int(tid.x) * BQ + params->q_off + tm + sm + (i * stile_t::kFragRows);
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < stile_t::kTileCols; j++) {
            const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
              const int c = col_pos + jj;
              const int col_abs = c + params->k_off;
              if (c >= params->kL || col_abs > row_abs ||
                  (row_abs - col_abs) >= params->window) {
                Stile.frag_at(i, j)[jj] = neg_inf;
              }
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V blocks
    if (!ALIGN_K && kb == (params->NK_aligned)) {
      loader_v.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    // Do softmax

    // Temp variables
    float new_max[kRowsPT];
    float factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    // Guard: when max_score == -inf (no valid K seen yet), the previous
    // accumulation is all zeros so the correct rescaling factor is 0.
    // Without this, -inf - (-inf) = NaN which poisons the output.
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = (max_score[i] == -metal::numeric_limits<float>::infinity())
          ? float(0)
          : fast::exp2(max_score[i] - new_max[i]);
    }

    // Save max for next iteration
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    // Row Sum
    float sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    // Load V into registers
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<half, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    // Prepare for next iteration
    loader_k.next();
    loader_v.next();
  }

  // Normalize output
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  // Store results
  O += (tm + sm) * params->o_stride_r + sn;

  if (!ALIGN_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<float, 1, 1>(
        O, int(params->o_stride_r), dst_tile_dims);
  } else {
    Otile.template store<float, 1, 1>(O, int(params->o_stride_r));
  }
}

// One [[host_name]] per (ALIGN_Q, ALIGN_K) combination — steel selected these
// via function constants 200/201; our compiled_pipeline resolves plain names.
#define instantiate_flash_attn(name, aq, ak)     \
  template [[host_name(name)]] [[kernel]] decltype(flash_attn<aq, ak>) \
      flash_attn<aq, ak>;

instantiate_flash_attn("kernel_flash_attn_q1_k1", true, true)
instantiate_flash_attn("kernel_flash_attn_q1_k0", true, false)
instantiate_flash_attn("kernel_flash_attn_q0_k1", false, true)
instantiate_flash_attn("kernel_flash_attn_q0_k0", false, false)
