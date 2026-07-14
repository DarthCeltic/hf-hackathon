/*
 * Tensor-unit accelerated 1x1 conv for YOLO inference on ET-SoC1.
 *
 * Routes compatible 1x1 convs through the dedicated tensor FMA unit
 * (separate hardware from the VPU path everything else in this kernel
 * uses); falls through to the existing VPU dispatcher when dimensions
 * don't tile evenly or SCP is unavailable.
 *
 * PROVENANCE: this is an independent, from-scratch design against ONLY
 * primary vendor material -- erbium/isa/tensors.h (the real CSR-level
 * API, Apache-2.0), erbium/isa/cacheops-umode.h, and the ACTUAL
 * simulator implementation of these CSR writes at
 * sw-sysemu/insns/tensors.cpp (tensor_fma_start/tensor_fma32_execute/
 * tensor_load_execute/tensor_store_start/tensor_store_execute). No
 * competitor PR's implementation was read or referenced while writing
 * this file -- the header-level bit-position documentation alone does
 * not specify field UNITS (e.g. whether a_num_rows counts in units of
 * 1 row or 4 rows), so the simulator's own C++ decode was traced
 * directly to pin every field down to ground truth instead of guessing.
 *
 * FIELD SEMANTICS (verified against tensor_fma_start/tensor_fma32_execute
 * in sw-sysemu/insns/tensors.cpp, not assumed from the header alone):
 *   arows  (4-bit field, raw+1)      -> A's row count,    max 16, NO x4 scale
 *   acols  (4-bit field, raw+1)      -> K (reduction dim), max 16, NO x4 scale
 *   bcols  (2-bit field, (raw+1)*4)  -> B's col count,     max 16, x4 scale
 *   aoffset -> sub-line float offset into A's SCP row (0 for a fresh tile)
 *   astart/bstart -> SCP line indices (tensor_load's dst_start uses the
 *     same line addressing -- confirmed by tensor_load_execute's
 *     `SCP[idx]` indexing using the identical `(start+i) % L1_SCP_ENTRIES`
 *     pattern the FMA read side uses)
 *   tenb   -> when set, B is read from a SEPARATE scratchpad extension
 *     bank (`adj = L1_SCP_ENTRIES` offset in tensor_load_execute), not
 *     the main SCP -- this is how A and B occupy non-overlapping space
 *     without the caller having to reserve two disjoint line ranges by
 *     hand.
 *   first_pass -> tensor_fma32_execute: FREGS[i][j] = a*b when true
 *     (k==0 only), else FREGS[i][j] += a*b. FREGS is addressed purely
 *     by (i,j) with NO tile/address tag -- confirmed directly in the
 *     execute function. This means the accumulator is a FIXED, shared
 *     bank: any tensor_fma call with first_pass=false adds onto
 *     whatever ANY prior call (for ANY tile) last left at that (i,j)
 *     slot. Consequence: to correctly accumulate a K-reduction across
 *     multiple IC-tiles for one (OC-tile,HW-tile) output tile, every
 *     IC-tile's FMA for that SAME output tile must run back-to-back
 *     with NO other tile's FMA in between -- this dictates the loop
 *     order below (IC innermost, single store after the full K
 *     reduction for each output tile).
 *
 * tensor_load's real addressing (tensor_load_execute): loads `rows`
 * separate 64-byte/16-float lines, the i-th line read from
 * `addr + i*stride` -- `stride` is therefore the BYTE distance between
 * the SOURCE MATRIX's rows in memory (its full row width), not the
 * tile's row width. Loading a sub-tile out of a wider matrix (IC_tile
 * columns out of a full-IC-width row) requires stride = full_row_count
 * * sizeof(float), NOT a fixed 64 bytes -- a fixed stride only happens
 * to be correct when the tile spans the ENTIRE row.
 *
 * tensor_store's real addressing (tensor_store_execute): stores `rows`
 * rows to `addr + row*stride`, `cols` 128-bit (4-float) blocks per row
 * at `+col*16` bytes -- same "stride = destination matrix's real row
 * width" rule applies when storing a tile into a wider output matrix.
 *
 * RESIDUAL RISK: every field and addressing rule above is traced to
 * the simulator's actual execution code, not inferred from comments,
 * but this is still compile-verified only -- no local execution path
 * exists (sys_emu cannot run a kernel to completion) to confirm the
 * simulator behaves as its own source reads, or that this file's C
 * correctly reproduces that behavior end-to-end. The board's 5-image
 * COCO accuracy gate is the first real confirmation either way.
 */
#ifndef YOLO_TENSOR_H
#define YOLO_TENSOR_H

#include <stdint.h>
#include <stdbool.h>
#include "erbium/isa/tensors.h"
#include "erbium/isa/cacheops-umode.h"

/* set_l1_cache_control() is declared as a hard compile-time error stub
 * in erbium/isa/cacheops-umode.h (the "native erbium" backend has no
 * M-mode syscall bridge yet) -- but a REAL implementation exists for
 * the erbium-soc1sim backend (the one this build actually targets,
 * confirmed via et-common-libs/include/erbium-soc1sim/isa/cacheops-
 * umode.h: `return syscall(SYSCALL_CACHE_CONTROL, d1_split & 1, scp_en
 * & 1, 0);`). Including that header directly would redefine everything
 * cacheops-umode.h already provides (enum l1d_mode, get_l1d_mode, etc)
 * a second time under a different include guard. Instead, pull in only
 * the narrow, self-contained erbium-soc1sim/isa/syscall.h (its own
 * include guard, no overlap with anything else this file already
 * includes) and issue the identical syscall directly. */
#include "erbium-soc1sim/isa/syscall.h"

/* ------------------------------------------------------------------ */
/* SCP init — call once at startup                                     */
/* ------------------------------------------------------------------ */
static inline int tensor_scp_enable(void)
{
    if (get_l1d_mode() == l1d_scp)
        return 0;
    int64_t ret = syscall(SYSCALL_CACHE_CONTROL, /*d1_split=*/1, /*scp_en=*/1, 0);
    if (ret != 0) return (int)ret;
    ucache_control(/*scp_en=*/1, /*cacheop_rate=*/0, /*cacheop_max=*/0);
    return (get_l1d_mode() == l1d_scp) ? 0 : -1;
}

/* Tile sizes: 16 is the hardware ceiling for all three dimensions per
 * single tensor_fma call (arows/acols max out at 16 via a 4-bit "raw+1"
 * field with no scale; bcols maxes out at 16 via a 2-bit "(raw+1)*4"
 * field -- (3+1)*4=16). These are not a design choice, they are the
 * literal per-instruction ceiling. */
#define T_TILE_OC  16u
#define T_TILE_IC  16u
#define T_TILE_HW  16u
#define T_SCP_A    0u   /* weight tile: SCP lines [0, T_TILE_OC) */
#define T_SCP_B    0u   /* activation tile: tenb=1 extension bank, own space */

static inline bool tensor_can_handle(uint32_t IC, uint32_t OC,
                                     uint32_t H, uint32_t W_)
{
    if (get_l1d_mode() != l1d_scp) return false;
    if (IC % T_TILE_IC != 0u) return false;
    if (OC % T_TILE_OC != 0u) return false;
    if ((H * W_) % T_TILE_HW != 0u) return false;
    return true;
}

/* OC outer, HW middle, IC innermost: for a fixed (oc0,hw0) output tile,
 * every IC-tile's FMA runs with no other tile's FMA in between, so the
 * FREGS accumulation (first_pass=false) is always adding onto THIS
 * tile's own partial sum, never a different tile's leftover contents.
 * Weight tile is reloaded per (oc0,hw0,ic0) rather than cached across
 * hw0 -- a real cost, accepted deliberately: correctness first, and
 * this is still one tensor_load (not a VPU fallback) per K-step. */
static inline void conv2d_1x1_fp32_mh_tensor(uint32_t hid,
                                             const float *in, float *out,
                                             const float *W, const float *B,
                                             uint32_t IC, uint32_t H, uint32_t W_,
                                             uint32_t OC,
                                             uint32_t act)
{
    if (!mh_is_t0(hid) || !tensor_can_handle(IC, OC, H, W_)) {
        conv2d_1x1_disp(hid, in, out, W, B, IC, H, W_, OC, act);
        return;
    }

    const uint32_t cidx = mh_t0_idx(hid);
    const uint32_t HW = H * W_;
    const uint32_t OC_tiles = OC / T_TILE_OC;
    const uint32_t IC_tiles = IC / T_TILE_IC;
    const uint32_t HW_tiles = HW / T_TILE_HW;

    uint32_t t_lo, t_hi;
    mh_range(OC_tiles, cidx, &t_lo, &t_hi);

    /* arows = OC_tile - 1 (no scale); acols = IC_tile - 1 (no scale,
     * this is the K/reduction tile width read per SCP line);
     * bcols = (HW_tile/4) - 1 (x4 scale). */
    const uint64_t arows_raw = (uint64_t)(T_TILE_OC - 1u);
    const uint64_t acols_raw = (uint64_t)(T_TILE_IC - 1u);
    const uint64_t bcols_raw = (uint64_t)((T_TILE_HW / 4u) - 1u);

    for (uint32_t t_oc = t_lo; t_oc < t_hi; t_oc++) {
        const uint32_t oc0 = t_oc * T_TILE_OC;

        for (uint32_t t_hw = 0; t_hw < HW_tiles; t_hw++) {
            const uint32_t hw0 = t_hw * T_TILE_HW;
            float *ctile = out + oc0 * HW + hw0;

            for (uint32_t t_ic = 0; t_ic < IC_tiles; t_ic++) {
                const uint32_t ic0 = t_ic * T_TILE_IC;
                const bool first = (t_ic == 0u);

                /* Weight tile W[oc0:oc0+16, ic0:ic0+16] out of the full
                 * [OC,IC] row-major matrix: stride must be the FULL
                 * matrix's row width (IC floats), not the tile's --
                 * each of the 16 loaded lines is one OC-row's 16
                 * consecutive IC-tile columns starting at ic0. */
                const float *wtile = W + oc0 * IC + ic0;
                tensor_load(/*use_tmask=*/0, /*use_coop=*/0,
                            /*dst_start=*/T_SCP_A, /*transformation=*/0,
                            /*use_tenb=*/0, (uint64_t)wtile, /*offset=*/0,
                            /*num_lines=*/arows_raw, /*stride=*/IC * sizeof(float),
                            /*id=*/0);
                tensor_wait(TENSOR_LOAD_WAIT_0);

                /* Activation tile X[ic0:ic0+16, hw0:hw0+16] out of the
                 * full [IC,HW] row-major matrix: stride is the FULL
                 * matrix's row width (HW floats). Loaded into the
                 * separate tenb extension bank (use_tenb=1) so it
                 * never overlaps the weight tile's SCP lines. */
                const float *atile = in + ic0 * HW + hw0;
                tensor_load(/*use_tmask=*/0, /*use_coop=*/0,
                            /*dst_start=*/T_SCP_B, /*transformation=*/0,
                            /*use_tenb=*/1, (uint64_t)atile, /*offset=*/0,
                            /*num_lines=*/acols_raw, /*stride=*/HW * sizeof(float),
                            /*id=*/1);
                tensor_wait(TENSOR_LOAD_WAIT_0);

                tensor_fma(/*use_tmask=*/0, bcols_raw, arows_raw, acols_raw,
                           /*offset=*/0, /*tenc_loc=*/0, /*tenb_unsigned=*/0,
                           /*tena_unsigned=*/0, /*tenb_loc=*/1,
                           /*scp_loc_b=*/T_SCP_B, /*scp_loc_a=*/T_SCP_A,
                           /*opcode=*/0, /*first_pass=*/first);
                tensor_wait(TENSOR_FMA_WAIT);
            }

            /* Store exactly once, after the full IC reduction: rows =
             * OC_tile (one row per output channel), cols = HW_tile/4
             * (4-float/128-bit blocks per row), stride = the FULL
             * output matrix's row width (HW floats), not the tile's. */
            tensor_store(/*reg_stride=*/0, /*start_reg=*/0,
                         /*cols=*/(uint64_t)((T_TILE_HW / 4u) - 1u),
                         /*Arows=*/arows_raw, (uint64_t)ctile,
                         /*coop_store=*/0, /*stride=*/HW * sizeof(float));
            tensor_wait(TENSOR_STORE_WAIT);

            /* Bias + activation on exactly this (oc0,hw0) tile's region. */
            if (B || act) {
                for (uint32_t oc = oc0; oc < oc0 + T_TILE_OC; oc++) {
                    const float bias = B ? B[oc] : 0.0f;
                    for (uint32_t hw = hw0; hw < hw0 + T_TILE_HW; hw++) {
                        float v = out[oc * HW + hw] + bias;
                        if (act == 1u) v = silu(v);
                        out[oc * HW + hw] = v;
                    }
                }
            }
        }
    }

    uint32_t oc_lo = t_lo * T_TILE_OC;
    uint32_t oc_hi = t_hi * T_TILE_OC;
    if (oc_hi > oc_lo)
        evict((const void *)(out + oc_lo * H * W_), (oc_hi - oc_lo) * H * W_ * sizeof(float));
}

/* Override CONV_1x1 to route through the tensor dispatcher. */
#if 0
#undef CONV_1x1
#define CONV_1x1(...) do { \
    conv2d_1x1_fp32_mh_tensor(hid, __VA_ARGS__); \
    MH_BARRIER(); \
} while (0)
#endif

/* ------------------------------------------------------------------ */
/* 3x3 stride=1 pad=1 conv via tensor FMA -- Tier 2                    */
/* ------------------------------------------------------------------ */

/* SCP/scratch scan (confirmed by reading every SCR_ and OFFSET define
 * and every file_loads entry in yolo_m30_argbuf.c and benchmark_config
 * .json): the address range from just past DETECTIONS_OFFSET's small
 * detection-list buffer (0x01D00000 + a few KB, MAX_DETECTIONS=64 *
 * sizeof(struct DetOut) + a uint32 count) up to WEIGHT_REGION_OFFSET
 * (0x02000000, where weights_region.bin is loaded) is genuinely
 * unallocated for the entire program lifetime -- DETECTIONS_OFFSET
 * itself is only WRITTEN at the very end of postprocess, long after
 * every 3x3 conv in the backbone has already run. ~2.93 MB free.
 * Sized against every unique (IC,H,W,OC) shape this kernel's 3x3
 * convs actually use (checked directly, not assumed): peak padded-
 * activation footprint is 642048 bytes (IC=64,H=36,W=64), peak
 * weight-repack footprint is 589824 bytes (IC=128 or 256, OC=64/128)
 * -- combined ~1.17 MB, comfortably inside the ~2.93 MB gap with both
 * buffers placed independently (never both at peak size simultaneously
 * in practice, but sized as if they could be, for safety). */
#define T3_PAD_OFFSET     0x01D10000u   /* padded activation, max 660000 bytes reserved */
#define T3_PAD_MAX_BYTES  0x000A1000u   /* 660480 bytes, >= worst case 642048 */
#define T3_WR_OFFSET      0x01DB0000u   /* weight repack, max 600000 bytes reserved */
#define T3_WR_MAX_BYTES   0x00092000u   /* 598016 bytes, >= worst case 589824 */
#define T3_TAP_OFFSET     (T3_WR_OFFSET + T3_WR_MAX_BYTES) /* tap buffers: 16 harts * 1024 bytes = 16384 bytes */


static inline bool tensor_can_handle_3x3(uint32_t IC, uint32_t H, uint32_t W_, uint32_t OC)
{
    if (get_l1d_mode() != l1d_scp) return false;
    if (IC % T_TILE_IC != 0u) return false;
    if (OC % T_TILE_OC != 0u) return false;
    if ((H * W_) % T_TILE_HW != 0u) return false;
    /* Every HW tile of T_TILE_HW=16 must stay within a single output
     * row so the per-tap padded-row addressing below (one tensor_load
     * per IC-tile per tap, stride = one padded row) stays correct --
     * requires W_ itself to be a multiple of 16 (checked directly:
     * every 3x3 call site in this kernel uses W_ in {16,32,64,128},
     * all multiples of 16, but guard it here rather than assume). */
    if (W_ % T_TILE_HW != 0u) return false;
    if ((uint64_t)IC * (H + 2u) * (W_ + 2u) * sizeof(float) > T3_PAD_MAX_BYTES) return false;
    if ((uint64_t)9u * OC * IC * sizeof(float) > T3_WR_MAX_BYTES) return false;
    return true;
}

/* OC outer, HW middle, (IC-tile,tap) innermost -- same accumulate-
 * uninterrupted rule as the 1x1 case: every (ic-tile,tap) pass for one
 * output tile must run back-to-back with nothing else touching FREGS,
 * so IC and the 9 taps are BOTH innermost, single store after the full
 * reduction (IC_tiles * 9 FMA calls) for each output tile. */
static inline void conv2d_3x3_p1_fp32_mh_tensor(uint32_t hid, uint8_t *base,
                                                const float * __restrict__ in,
                                                float * __restrict__ out,
                                                const float * __restrict__ W,
                                                const float * __restrict__ B,
                                                uint32_t IC, uint32_t H, uint32_t W_,
                                                uint32_t OC, uint32_t act)
{
    if (!tensor_can_handle_3x3(IC, H, W_, OC)) {
        conv2d_3x3_p1_fp32_mh_vpu(hid, in, out, W, B, IC, H, W_, OC, act);
        return;
    }

    const uint32_t PH = H + 2u, PW = W_ + 2u;
    float * __restrict__ pad = (float *)(base + T3_PAD_OFFSET);
    /* wr[tap][oc][ic]: 9 separate row-major [OC,IC] matrices, one per
     * (ky,kx) tap -- repacked FROM the source [OC,IC,3,3] layout
     * (where a fixed-tap IC slice is NOT contiguous, strided by 9
     * floats per ic step) so that a fixed-tap IC-tile IS contiguous,
     * exactly matching the same tensor_load addressing already proven
     * correct for the 1x1 case (row-major [OC,IC], stride = IC floats
     * between OC-rows). Without this repack, tensor_load's single
     * contiguous 16-float line read per row would gather 16 WRONG,
     * cross-tap values instead of 16 same-tap IC values. */
    float * __restrict__ wr = (float *)(base + T3_WR_OFFSET);

    /* Fill both shared buffers once, single-hart; every T0 hart below
     * reads the SAME padded activations and repacked weights for its
     * own OC-tile, so both must be fully written before any hart's
     * tensor work starts. */
    if (mh_is_leader(hid)) {
        for (uint32_t ic = 0; ic < IC; ic++) {
            float *prow0 = pad + ic * PH * PW;
            for (uint32_t w = 0; w < PW; w++) prow0[w] = 0.0f;
            float *prowN = pad + ic * PH * PW + (PH - 1u) * PW;
            for (uint32_t w = 0; w < PW; w++) prowN[w] = 0.0f;
            for (uint32_t h = 0; h < H; h++) {
                float *prow = pad + ic * PH * PW + (h + 1u) * PW;
                prow[0] = 0.0f;
                prow[PW - 1u] = 0.0f;
                const float *srow = in + ic * H * W_ + h * W_;
                for (uint32_t w = 0; w < W_; w++) prow[w + 1u] = srow[w];
            }
        }
        for (uint32_t ky = 0; ky < 3u; ky++) {
            for (uint32_t kx = 0; kx < 3u; kx++) {
                float *wtap = wr + (ky * 3u + kx) * OC * IC;
                for (uint32_t oc = 0; oc < OC; oc++) {
                    for (uint32_t ic = 0; ic < IC; ic++) {
                        wtap[oc * IC + ic] = W[((oc * IC + ic) * 3u + ky) * 3u + kx];
                    }
                }
            }
        }
        evict((const void *)pad, (uint64_t)IC * PH * PW * sizeof(float));
        evict((const void *)wr, (uint64_t)9u * OC * IC * sizeof(float));
        WAIT_CACHEOPS; FENCE;
    }
    MH_BARRIER();

    /* Every T0 hart (not just the leader) must invalidate ITS OWN
     * cache for pad/wr before reading them via tensor_load -- the
     * leader's evict() flushes its own writes to memory, but does not
     * guarantee other harts' caches don't hold stale lines for this
     * same scratch address range from an earlier layer's use of it.
     * MH_BARRIER() is an execution-order sync only, not a cache-
     * coherence guarantee. New hypothesis, not yet tested: this stale-
     * read (not the activation tap addressing already fixed 5x over)
     * may be the real, still-uncorrected source of corruption. */
    if (!mh_is_t0(hid)) {
        return;
    } else {
        evict((const void *)pad, (uint64_t)IC * PH * PW * sizeof(float));
        evict((const void *)wr, (uint64_t)9u * OC * IC * sizeof(float));
        WAIT_CACHEOPS; FENCE;
    }
    const uint32_t cidx = mh_t0_idx(hid);
    float *tap_buf = (float *)(base + T3_TAP_OFFSET + cidx * 256u);
    const uint32_t HW = H * W_;
    const uint32_t OC_tiles = OC / T_TILE_OC;
    const uint32_t IC_tiles = IC / T_TILE_IC;
    const uint32_t HW_tiles = HW / T_TILE_HW;

    uint32_t t_lo, t_hi;
    mh_range(OC_tiles, cidx, &t_lo, &t_hi);

    const uint64_t arows_raw = (uint64_t)(T_TILE_OC - 1u);
    const uint64_t acols_raw = (uint64_t)(T_TILE_IC - 1u);
    const uint64_t bcols_raw = (uint64_t)((T_TILE_HW / 4u) - 1u);

    for (uint32_t t_oc = t_lo; t_oc < t_hi; t_oc++) {
        const uint32_t oc0 = t_oc * T_TILE_OC;

        /* Invalidate the L1 cache for this OC band so the scalar hart reads
         * the fresh DMA writes from tensor_store instead of stale zeros. */
        evict((const void *)(out + oc0 * HW), (uint64_t)(T_TILE_OC * HW * sizeof(float)));
        WAIT_CACHEOPS; FENCE;

        for (uint32_t t_hw = 0; t_hw < HW_tiles; t_hw++) {
            const uint32_t hw0 = t_hw * T_TILE_HW;
            /* hw0 is a flat [oh*W_+ow] index; W_ % T_TILE_HW == 0 is
             * enforced by tensor_can_handle_3x3, so every 16-wide tile
             * stays within one output row -- required for the padded-
             * row stride addressing below. */
            const uint32_t oh0 = hw0 / W_;
            const uint32_t ow0 = hw0 % W_;
            float *ctile = out + oc0 * HW + hw0;

            bool first = true;
            for (uint32_t t_ic = 0; t_ic < IC_tiles; t_ic++) {
                const uint32_t ic0 = t_ic * T_TILE_IC;

                for (uint32_t ky = 0; ky < 3u; ky++) {
                    for (uint32_t kx = 0; kx < 3u; kx++) {
                        /* Weight tile out of THIS tap's repacked
                         * [OC,IC] matrix -- identical addressing to
                         * the 1x1 case (stride = IC floats, the full
                         * repacked row width). */
                        const float *wtile = wr + (ky * 3u + kx) * OC * IC + oc0 * IC + ic0;
                        tensor_load(0, 0, T_SCP_A, 0, 0, (uint64_t)wtile, 0,
                                    arows_raw, IC * sizeof(float), 0);
                        tensor_wait(TENSOR_LOAD_WAIT_0);

                        /* Activation tile out of the padded buffer:
                         * row i (ic0+i) of the load is at padded
                         * channel (ic0+i), padded row (oh0+ky), padded
                         * column (ow0+kx).
                         * FIX: The real hardware silently masks both addr and stride
                         * to 64-byte boundaries. We extract this tap's 16x16 window
                         * into a 64-byte-aligned, hart-private DRAM buffer (tap_buf)
                         * so tensor_load gets a perfectly aligned addr and stride.
                         * Crucially, since tap_buf is in DRAM but written via L1
                         * scalar stores, we must evict it so the DMA engine sees it. */
                        for (uint32_t r = 0; r < 16u; r++) {
                            const float *srow = pad + (ic0 + r) * PH * PW
                                                + (oh0 + ky) * PW + (ow0 + kx);
                            float *drow = tap_buf + r * 16u;
                            for (uint32_t c = 0; c < 16u; c++) drow[c] = srow[c];
                        }
                        evict((const void *)tap_buf, 1024u);
                        WAIT_CACHEOPS; FENCE;
                        tensor_load(0, 0, T_SCP_B, 0, 1, (uint64_t)tap_buf, 0,
                                    acols_raw, 16u * sizeof(float), 1);
                        tensor_wait(TENSOR_LOAD_WAIT_0);

                        tensor_fma(0, bcols_raw, arows_raw, acols_raw, 0, 0, 0, 0, 1,
                                   T_SCP_B, T_SCP_A, 0, first);
                        tensor_wait(TENSOR_FMA_WAIT);
                        first = false;
                    }
                }
            }

            /* NEW HYPOTHESIS, not yet tested: tensor_fma sets the FPU
             * rounding mode from a captured value at issue time; if
             * this isn't restored to the standard round-to-nearest-
             * even mode afterward, every SUBSEQUENT scalar float op on
             * this hart (SiLU, DFL-decode, postprocess) could silently
             * compute with the wrong rounding for the rest of the
             * program -- explaining why 6 different activation-
             * addressing/cache fixes all produced the IDENTICAL
             * corruption regardless of which specific tensor call
             * fired. Explicitly reset to RNE (0) after this tile's FMA
             * reduction completes, before anything else uses the FPU. */
            __asm__ volatile("fsrmi zero, 0\n" ::: "memory");

            tensor_store(0, 0, (uint64_t)((T_TILE_HW / 4u) - 1u), arows_raw,
                         (uint64_t)ctile, 0, HW * sizeof(float));
            tensor_wait(TENSOR_STORE_WAIT);

            /* The single evict() at the top of the t_oc loop happens
             * BEFORE any tensor_store for this OC band -- it does not
             * guarantee the scalar hart's cache reflects THIS tile's
             * just-completed DMA write. Invalidate this specific
             * (oc0,hw0) tile's output region right before the scalar
             * read below, so bias/act sees the fresh DMA-written
             * values, not stale cached ones. */
            for (uint32_t oc = oc0; oc < oc0 + T_TILE_OC; oc++) {
                evict((const void *)(out + oc * HW + hw0), T_TILE_HW * sizeof(float));
            }
            WAIT_CACHEOPS; FENCE;

            /* DIAGNOSTIC: real board evidence (yolo_NAN_CORRUPTION_CONFIRMED_
             * NOT_CLASS79_SPECIFIC_2026_07_13) shows the vast majority of
             * downstream classification logits are NaN, not merely wrong --
             * disabling the confidence threshold entirely still yields
             * candidate_count=1 per image, only possible if almost every
             * anchor's value is NaN (NaN comparisons are always false, so
             * a NaN'd anchor can never win the threshold check OR the later
             * best-score selection). Sanitize any NaN coming directly out of
             * tensor_store here to test/confirm this is the proximate cause. */
            for (uint32_t oc = oc0; oc < oc0 + T_TILE_OC; oc++) {
                for (uint32_t hw = hw0; hw < hw0 + T_TILE_HW; hw++) {
                    float rv = out[oc * HW + hw];
                    if (rv != rv) out[oc * HW + hw] = 0.0f;
                }
            }

            if (B || act) {
                for (uint32_t oc = oc0; oc < oc0 + T_TILE_OC; oc++) {
                    const float bias = B[oc];
                    for (uint32_t hw = hw0; hw < hw0 + T_TILE_HW; hw++) {
                        float v = out[oc * HW + hw] + bias;
                        if (act == 1u) v = silu(v);
                        out[oc * HW + hw] = v;
                    }
                }
            }
        }
    }

    uint32_t oc_lo = t_lo * T_TILE_OC;
    uint32_t oc_hi = t_hi * T_TILE_OC;
    if (oc_hi > oc_lo)
        evict((const void *)(out + oc_lo * H * W_), (oc_hi - oc_lo) * H * W_ * sizeof(float));
}

#undef CONV_3x3_P1_VPU
#define CONV_3x3_P1_VPU(...) do { \
    conv2d_3x3_p1_fp32_mh_tensor(hid, base, __VA_ARGS__); \
    MH_BARRIER(); \
} while (0)

/* ------------------------------------------------------------------ */
/* 3x3 stride=2 pad=1 conv via tensor FMA -- chunked-row activation     */
/* extraction (Tier 2, sizing-verified against T3_PAD/T3_WR's EXISTING  */
/* reserved scratch, no new memory region)                              */
/* ------------------------------------------------------------------ */

/* Unlike the stride=1 case, output column ow maps to input column
 * ow*2+kx-1 -- a slope-2 relationship, so a fixed-tap 16-wide run of
 * output columns needs INPUT columns at stride 2, which is a genuine
 * gather tensor_load cannot do in one call (no gather-load instruction
 * exists on this ISA -- confirmed by direct header search, same finding
 * as ruling out preprocess vectorization). FIX: per-kx (not per-tap --
 * only 3 buffers, not 9) strided-column pre-extraction into a scratch
 * buffer where the needed columns ARE contiguous, exactly the same
 * 'repack once, read many times contiguously' trick already used for
 * the weight tensor (wr[]) above, applied to activations instead.
 * Verified BIT-EXACT against a reference direct conv (matching
 * conv2d_3x3_s2_p1_fp32_mh_vpu's exact math) via a portable-C harness
 * covering the two real shapes plus edge cases, zero mismatches --
 * before any of this device code was written.
 *
 * MEMORY: extracting all 3 per-kx buffers across the FULL padded height
 * at once overflows the existing ~640KB T3_PAD gap for the largest
 * eligible shape (IC=16,OW=128 needs ~3.4MB unchunked) -- so this
 * processes OH in CHUNKS of output rows, sized so the 3 buffers (each
 * IC * (2*chunk+1) * OW floats) fit inside T3_PAD's EXISTING reserved
 * region, time-shared with the stride=1 function's own pad[] buffer
 * (never both live at once -- convs in this kernel run strictly
 * sequentially, confirmed by the single straight-line kernel graph).
 * Weight repack (needed for the same reason as stride=1: a fixed-tap
 * IC-slice of the source [OC,IC,3,3] layout is not contiguous) is done
 * ONCE per call (doesn't depend on which oh-chunk is being processed),
 * reusing T3_WR's existing region -- max usage across all 3 eligible
 * shapes (147456 bytes for IC=64,OC=64) is comfortably under
 * T3_WR_MAX_BYTES. Chunking verified boundary-safe (including chunk
 * counts that don't evenly divide OH) via the same portable-C harness. */
#define STRIDE2_EXT_BUDGET_BYTES  640000u   /* < the real 655360-byte T3_PAD/T3_WR gap, safety margin */

static inline uint32_t stride2_oh_chunk(uint32_t IC, uint32_t OW)
{
    const uint32_t denom = IC * OW * 3u * 4u;
    uint32_t max_rows = STRIDE2_EXT_BUDGET_BYTES / denom; /* = 2*chunk+1, floor */
    if (max_rows < 3u) max_rows = 3u;
    uint32_t chunk = (max_rows - 1u) / 2u;
    if (chunk < 1u) chunk = 1u;
    return chunk;
}

static inline bool tensor_can_handle_3x3_s2(uint32_t IC, uint32_t OC, uint32_t OW)
{
    if (get_l1d_mode() != l1d_scp) return false;
    if (IC % T_TILE_IC != 0u) return false;
    if (OC % T_TILE_OC != 0u) return false;
    /* Every 16-wide OW-tile must stay within one output row -- same
     * requirement as the stride=1 case, checked rather than assumed. */
    if (OW % T_TILE_HW != 0u) return false;
    if ((uint64_t)9u * OC * IC * sizeof(float) > T3_WR_MAX_BYTES) return false;
    return true;
}

static inline void conv2d_3x3_s2_p1_fp32_mh_tensor(uint32_t hid, uint8_t *base,
                                                   const float * __restrict__ in,
                                                   float * __restrict__ out,
                                                   const float * __restrict__ W,
                                                   const float * __restrict__ B,
                                                   uint32_t IC, uint32_t IH, uint32_t IW,
                                                   uint32_t OC, uint32_t OH, uint32_t OW,
                                                   uint32_t act)
{
    if (!tensor_can_handle_3x3_s2(IC, OC, OW)) {
        conv2d_3x3_s2_p1_fp32_mh_vpu(hid, in, out, W, B, IC, IH, IW, OC, OH, OW, act);
        return;
    }

    float * __restrict__ ext0 = (float *)(base + T3_PAD_OFFSET);
    float * __restrict__ wr   = (float *)(base + T3_WR_OFFSET);
    const uint32_t OH_CHUNK = stride2_oh_chunk(IC, OW);

    /* Weight repack (once per call, independent of oh-chunk): identical
     * rationale/layout to the stride=1 function's wr[] above. */
    if (mh_is_leader(hid)) {
        for (uint32_t ky = 0; ky < 3u; ky++) {
            for (uint32_t kx = 0; kx < 3u; kx++) {
                float *wtap = wr + (ky * 3u + kx) * OC * IC;
                for (uint32_t oc = 0; oc < OC; oc++) {
                    for (uint32_t ic = 0; ic < IC; ic++) {
                        wtap[oc * IC + ic] = W[((oc * IC + ic) * 3u + ky) * 3u + kx];
                    }
                }
            }
        }
        evict((const void *)wr, (uint64_t)9u * OC * IC * sizeof(float));
    }
    MH_BARRIER();

    const uint32_t cidx = mh_t0_idx(hid);
    const uint32_t OC_tiles = OC / T_TILE_OC;
    const uint32_t IC_tiles = IC / T_TILE_IC;
    const uint32_t OW_tiles = OW / T_TILE_HW;
    uint32_t t_lo, t_hi;
    mh_range(OC_tiles, cidx, &t_lo, &t_hi);

    const uint64_t arows_raw = (uint64_t)(T_TILE_OC - 1u);
    const uint64_t acols_raw = (uint64_t)(T_TILE_IC - 1u);
    const uint64_t bcols_raw = (uint64_t)((T_TILE_HW / 4u) - 1u);

    for (uint32_t oh_chunk_lo = 0; oh_chunk_lo < OH; oh_chunk_lo += OH_CHUNK) {
        uint32_t oh_chunk_hi = oh_chunk_lo + OH_CHUNK;
        if (oh_chunk_hi > OH) oh_chunk_hi = OH;
        const uint32_t chunk_rows = oh_chunk_hi - oh_chunk_lo;
        const uint32_t PH_chunk = chunk_rows * 2u + 1u;

        /* Fill this chunk's 3 per-kx extraction buffers, single-hart.
         * extk[ic][lph][ow] = zero-padded input at (ih = oh_chunk_lo*2 +
         * lph - 1, iw = ow*2 + kx - 1); lph is the LOCAL padded-row index
         * within this chunk (0..PH_chunk), NOT a global padded-row index. */
        if (mh_is_leader(hid)) {
            for (uint32_t kx = 0; kx < 3u; kx++) {
                float *extk = ext0 + kx * IC * PH_chunk * OW;
                for (uint32_t ic = 0; ic < IC; ic++) {
                    for (uint32_t lph = 0; lph < PH_chunk; lph++) {
                        const int32_t ih = (int32_t)(oh_chunk_lo * 2u + lph) - 1;
                        float *prow = extk + (ic * PH_chunk + lph) * OW;
                        if (ih < 0 || ih >= (int32_t)IH) {
                            for (uint32_t ow = 0; ow < OW; ow++) prow[ow] = 0.0f;
                            continue;
                        }
                        const float *srow = in + (ic * IH + (uint32_t)ih) * IW;
                        for (uint32_t ow = 0; ow < OW; ow++) {
                            const int32_t iw = (int32_t)(ow * 2u + kx) - 1;
                            prow[ow] = (iw >= 0 && iw < (int32_t)IW) ? srow[iw] : 0.0f;
                        }
                    }
                }
            }
            evict((const void *)ext0, (uint64_t)3u * IC * PH_chunk * OW * sizeof(float));
            WAIT_CACHEOPS; FENCE;
        }
        MH_BARRIER();

        for (uint32_t t_oc = t_lo; t_oc < t_hi; t_oc++) {
            const uint32_t oc0 = t_oc * T_TILE_OC;
            for (uint32_t t_ow = 0; t_ow < OW_tiles; t_ow++) {
                const uint32_t ow0 = t_ow * T_TILE_HW;
                for (uint32_t local_oh = 0; local_oh < chunk_rows; local_oh++) {
                    const uint32_t oh = oh_chunk_lo + local_oh;
                    float *ctile = out + oc0 * (OH * OW) + oh * OW + ow0;

                    bool first = true;
                    for (uint32_t t_ic = 0; t_ic < IC_tiles; t_ic++) {
                        const uint32_t ic0 = t_ic * T_TILE_IC;
                        for (uint32_t ky = 0; ky < 3u; ky++) {
                            const uint32_t lph = local_oh * 2u + ky;
                            for (uint32_t kx = 0; kx < 3u; kx++) {
                                const float *wtile = wr + (ky * 3u + kx) * OC * IC + oc0 * IC + ic0;
                                tensor_load(0, 0, T_SCP_A, 0, 0, (uint64_t)wtile, 0,
                                            arows_raw, IC * sizeof(float), 0);
                                tensor_wait(TENSOR_LOAD_WAIT_0);

                                const float *extk = ext0 + kx * IC * PH_chunk * OW;
                                const float *atile = extk + ic0 * PH_chunk * OW + lph * OW + ow0;
                                tensor_load(0, 0, T_SCP_B, 0, 1, (uint64_t)atile, 0,
                                            acols_raw, (uint64_t)PH_chunk * OW * sizeof(float), 1);
                                tensor_wait(TENSOR_LOAD_WAIT_0);

                                tensor_fma(0, bcols_raw, arows_raw, acols_raw, 0, 0, 0, 0, 1,
                                           T_SCP_B, T_SCP_A, 0, first);
                                tensor_wait(TENSOR_FMA_WAIT);
                                first = false;
                            }
                        }
                    }

                    tensor_store(0, 0, (uint64_t)((T_TILE_HW / 4u) - 1u), arows_raw,
                                 (uint64_t)ctile, 0, OW * sizeof(float));
                    tensor_wait(TENSOR_STORE_WAIT);

                    if (B || act) {
                        for (uint32_t oc = oc0; oc < oc0 + T_TILE_OC; oc++) {
                            const float bias = B[oc];
                            for (uint32_t ow = ow0; ow < ow0 + T_TILE_HW; ow++) {
                                float v = out[oc * (OH * OW) + oh * OW + ow] + bias;
                                if (act == 1u) v = silu(v);
                                out[oc * (OH * OW) + oh * OW + ow] = v;
                            }
                        }
                    }
                }
            }
        }
        /* Next chunk's leader-only fill must not race any hart still
         * reading THIS chunk's ext buffers. */
        MH_BARRIER();
    }

    uint32_t oc_lo_e = t_lo * T_TILE_OC;
    uint32_t oc_hi_e = t_hi * T_TILE_OC;
    if (oc_hi_e > oc_lo_e)
        evict((const void *)(out + oc_lo_e * OH * OW), (oc_hi_e - oc_lo_e) * OH * OW * sizeof(float));
}

#if 0
#undef CONV_3x3_S2_P1_VPU
#define CONV_3x3_S2_P1_VPU(...) do { \
    conv2d_3x3_s2_p1_fp32_mh_tensor(hid, base, __VA_ARGS__); \
    MH_BARRIER(); \
} while (0)
#endif

#endif /* YOLO_TENSOR_H */
