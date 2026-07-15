#ifndef YOLO_TENSOR_H
#define YOLO_TENSOR_H

#include <stdbool.h>
#include <stdint.h>
#include "erbium-soc1sim/isa/syscall.h"
#include "erbium/isa/tensors.h"

#define YOLO_TENSOR_OC 16u
#define YOLO_TENSOR_IC 16u
#define YOLO_TENSOR_HW 16u

static inline int yolo_tensor_enable(void) {
    if (get_l1d_mode() == l1d_scp)
        return 0;
    const int64_t ret = syscall(SYSCALL_CACHE_CONTROL, 1u, 1u, 0u);
    if (ret != 0)
        return (int)ret;
    ucache_control(1u, 0u, 0u);
    return get_l1d_mode() == l1d_scp ? 0 : -1;
}

static inline bool yolo_tensor_can_1x1(uint32_t IC, uint32_t OC, uint32_t H,
                                       uint32_t W_) {
    const uint32_t HW = H * W_;
    return get_l1d_mode() == l1d_scp && (IC % YOLO_TENSOR_IC) == 0u &&
           (OC % YOLO_TENSOR_OC) == 0u && (HW % YOLO_TENSOR_HW) == 0u;
}

static inline void yolo_tensor_clobber_fregs(void) {
    __asm__ volatile("" ::
                         : "memory", "f0", "f1", "f2", "f3", "f4", "f5", "f6",
                           "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14",
                           "f15", "f16", "f17", "f18", "f19", "f20", "f21",
                           "f22", "f23", "f24", "f25", "f26", "f27", "f28",
                           "f29", "f30", "f31");
}

static inline void yolo_tensor_epilogue_16(float *out, const float *B,
                                           uint32_t oc0, uint32_t HW,
                                           uint32_t hw0, uint32_t act) {
    float vz, vo, vl2e;
    union {
        float f;
        uint32_t u;
    } z = {0.0f};
    union {
        float f;
        uint32_t u;
    } o = {1.0f};
    union {
        float f;
        uint32_t u;
    } l = {1.4426950408889634f};
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(vz) : "r"((uint64_t)z.u));
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(vo) : "r"((uint64_t)o.u));
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(vl2e) : "r"((uint64_t)l.u));

    for (uint32_t r = 0; r < YOLO_TENSOR_OC; r++) {
        union {
            float f;
            uint32_t u;
        } bb = {B[r + oc0]};
        float vb;
        __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(vb) : "r"((uint64_t)bb.u));
        float *dst = out + (oc0 + r) * HW + hw0;
        for (uint32_t j = 0; j < YOLO_TENSOR_HW; j += 8u) {
            float x;
            __asm__ volatile("flq2 %0, 0(%1)\n" : "=f"(x) : "r"(dst + j));
            __asm__ volatile("fadd.ps %0, %0, %1\n" : "+f"(x) : "f"(vb));
            if (act == 1u) {
                float t;
                __asm__ volatile(
                    "fsub.ps %[t], %[z], %[x]\n"
                    "fmul.ps %[t], %[t], %[l2e]\n"
                    "fexp.ps %[t], %[t]\n"
                    "fadd.ps %[t], %[t], %[o]\n"
                    "frcp.ps %[t], %[t]\n"
                    "fmul.ps %[t], %[t], %[x]\n"
                    : [t] "=&f"(t)
                    : [x] "f"(x), [z] "f"(vz), [o] "f"(vo), [l2e] "f"(vl2e));
                x = t;
            }
            __asm__ volatile("fsq2 %1, 0(%0)\n" ::"r"(dst + j), "f"(x)
                             : "memory");
        }
    }
}

static inline void conv2d_1x1_fp32_mh_tensor(uint32_t hid, uint8_t *base,
                                             const float *in, float *out,
                                             const float *W, const float *B,
                                             uint32_t IC, uint32_t H,
                                             uint32_t W_, uint32_t OC,
                                             uint32_t act) {
    if (!mh_is_t0(hid) || !yolo_tensor_can_1x1(IC, OC, H, W_)) {
        conv2d_1x1_disp(hid, (float *)(base + WR1X1_SCRATCH_OFFSET), in, out, W,
                        B, IC, H, W_, OC, act);
        return;
    }

    const uint32_t HW = H * W_;
    const uint32_t oc_tiles = OC / YOLO_TENSOR_OC;
    const uint32_t ic_tiles = IC / YOLO_TENSOR_IC;
    const uint32_t hw_tiles = HW / YOLO_TENSOR_HW;
    const uint32_t cidx = mh_t0_idx(hid);
    uint32_t t_lo, t_hi;
    mh_range(oc_tiles, cidx, &t_lo, &t_hi);

    for (uint32_t t_oc = t_lo; t_oc < t_hi; t_oc++) {
        const uint32_t oc0 = t_oc * YOLO_TENSOR_OC;
        const uint32_t oc_bytes = YOLO_TENSOR_OC * HW * sizeof(float);
        evict((const void *)(out + oc0 * HW), oc_bytes);
        WAIT_CACHEOPS;
        FENCE;

        for (uint32_t t_hw = 0; t_hw < hw_tiles; t_hw++) {
            const uint32_t hw0 = t_hw * YOLO_TENSOR_HW;
            for (uint32_t t_ic = 0; t_ic < ic_tiles; t_ic++) {
                const uint32_t ic0 = t_ic * YOLO_TENSOR_IC;
                const float *wtile = W + oc0 * IC + ic0;
                const float *atile = in + ic0 * HW + hw0;

                tensor_load(0u, 0u, 0u, 0u, 0u, (uint64_t)wtile, 0u, 15u,
                            IC * sizeof(float), 0u);
                tensor_wait(TENSOR_LOAD_WAIT_0);
                tensor_load(0u, 0u, 0u, 0u, 1u, (uint64_t)atile, 0u, 15u,
                            HW * sizeof(float), 1u);
                tensor_wait(TENSOR_LOAD_WAIT_0);
                tensor_fma(0u, 3u, 15u, 15u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u,
                           t_ic == 0u);
                tensor_wait(TENSOR_FMA_WAIT);
            }

            float *ctile = out + oc0 * HW + hw0;
            tensor_store(0u, 0u, 3u, 15u, (uint64_t)ctile, 0u,
                         HW * sizeof(float));
            tensor_wait(TENSOR_STORE_WAIT);
            yolo_tensor_clobber_fregs();
            yolo_tensor_epilogue_16(out, B, oc0, HW, hw0, act);
        }

        evict((const void *)(out + oc0 * HW), oc_bytes);
        WAIT_CACHEOPS;
        FENCE;
    }
}

/* Repack 3x3 weights into OC16 tiles.  Within a tile, the layout is
 * [ky,kx][ic][oc_lane].  A transpose32 tensor load over 16 consecutive IC
 * cache lines then produces the required [oc_lane][ic_lane] A matrix without
 * an im2col buffer. */
static inline void yolo_tensor_repack_3x3(uint32_t hid, const float *W,
                                          float *WR, uint32_t IC, uint32_t OC) {
    if (!mh_is_t0(hid))
        return;

    const uint32_t oc_tiles = OC / YOLO_TENSOR_OC;
    const uint32_t rows = oc_tiles * 9u * IC;
    const uint32_t cidx = mh_t0_idx(hid);
    uint32_t row_lo, row_hi;
    mh_range(rows, cidx, &row_lo, &row_hi);

    for (uint32_t row = row_lo; row < row_hi; row++) {
        const uint32_t t_oc = row / (9u * IC);
        const uint32_t rem = row - t_oc * 9u * IC;
        const uint32_t k = rem / IC;
        const uint32_t ic = rem - k * IC;
        const uint32_t oc0 = t_oc * YOLO_TENSOR_OC;
        float *dst_tile = WR + (uint64_t)t_oc * 9u * IC * YOLO_TENSOR_OC;
        float *dst = dst_tile + (k * IC + ic) * YOLO_TENSOR_OC;
        for (uint32_t o = 0; o < YOLO_TENSOR_OC; o++)
            dst[o] = W[((oc0 + o) * IC + ic) * 9u + k];
    }

    if (row_hi > row_lo) {
        evict((const void *)(WR + (uint64_t)row_lo * YOLO_TENSOR_OC),
              (uint64_t)(row_hi - row_lo) * YOLO_TENSOR_OC * sizeof(float));
        WAIT_CACHEOPS;
        FENCE;
    }
}

static inline bool yolo_tensor_can_3x3_p1(uint32_t IC, uint32_t H, uint32_t W_,
                                          uint32_t OC) {
    return get_l1d_mode() == l1d_scp && (IC % YOLO_TENSOR_IC) == 0u &&
           (OC % YOLO_TENSOR_OC) == 0u && (W_ % YOLO_TENSOR_HW) == 0u &&
           H != 0u;
}

/* Add the kx={0,2} columns to the tensor-computed kx=1 partial result, then
 * apply bias and SiLU.  Four output channels share each activation load and
 * keep independent VPU accumulators, matching the board-proven OC4 pipeline
 * while doing only six of the original nine taps. */
static inline void yolo_tensor_finish_3x3_sides(float *out, const float *in,
                                                const float *WR, const float *B,
                                                uint32_t t_oc, uint32_t IC,
                                                uint32_t H, uint32_t W_,
                                                uint32_t oh, uint32_t ow16,
                                                uint32_t act) {
    const uint32_t HW = H * W_;
    const uint32_t oc0 = t_oc * YOLO_TENSOR_OC;
    const float *wr_tile = WR + (uint64_t)t_oc * 9u * IC * YOLO_TENSOR_OC;
    float edge[8] __attribute__((aligned(32)));

    float vz, vo, vl2e;
    union {
        float f;
        uint32_t u;
    } z = {0.0f};
    union {
        float f;
        uint32_t u;
    } o = {1.0f};
    union {
        float f;
        uint32_t u;
    } l = {1.4426950408889634f};
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(vz) : "r"((uint64_t)z.u));
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(vo) : "r"((uint64_t)o.u));
    __asm__ volatile("fbcx.ps %0, %1\n" : "=f"(vl2e) : "r"((uint64_t)l.u));

    for (uint32_t oo = 0; oo < YOLO_TENSOR_OC; oo += 4u) {
        float vb0, vb1, vb2, vb3;
#define YOLO_TENSOR_BCAST_BIAS(REG, INDEX)                                     \
    do {                                                                       \
        union {                                                                \
            float f;                                                           \
            uint32_t u;                                                        \
        } _b = {B[oc0 + oo + (INDEX)]};                                        \
        __asm__ volatile("fbcx.ps %0, %1\n"                                    \
                         : "=f"(REG)                                           \
                         : "r"((uint64_t)_b.u));                               \
    } while (0)
        YOLO_TENSOR_BCAST_BIAS(vb0, 0u);
        YOLO_TENSOR_BCAST_BIAS(vb1, 1u);
        YOLO_TENSOR_BCAST_BIAS(vb2, 2u);
        YOLO_TENSOR_BCAST_BIAS(vb3, 3u);
#undef YOLO_TENSOR_BCAST_BIAS

        for (uint32_t half = 0; half < 2u; half++) {
            const uint32_t ow8 = ow16 + half * 8u;
            float a0, a1, a2, a3;
            __asm__ volatile("flq2 %0, 0(%1)\n"
                             : "=f"(a0)
                             : "r"(out + (oc0 + oo + 0u) * HW + oh * W_ + ow8));
            __asm__ volatile("flq2 %0, 0(%1)\n"
                             : "=f"(a1)
                             : "r"(out + (oc0 + oo + 1u) * HW + oh * W_ + ow8));
            __asm__ volatile("flq2 %0, 0(%1)\n"
                             : "=f"(a2)
                             : "r"(out + (oc0 + oo + 2u) * HW + oh * W_ + ow8));
            __asm__ volatile("flq2 %0, 0(%1)\n"
                             : "=f"(a3)
                             : "r"(out + (oc0 + oo + 3u) * HW + oh * W_ + ow8));

            for (uint32_t ic = 0; ic < IC; ic++) {
                for (uint32_t ky = 0; ky < 3u; ky++) {
                    const int32_t ih = (int32_t)oh + (int32_t)ky - 1;
                    if (ih < 0 || ih >= (int32_t)H)
                        continue;
                    for (uint32_t side = 0; side < 2u; side++) {
                        const uint32_t kx = side * 2u;
                        const int32_t iw = (int32_t)ow8 + (int32_t)kx - 1;
                        float v;
                        if (iw >= 0 && iw + 7 < (int32_t)W_) {
                            const float *src =
                                in + ic * HW + (uint32_t)ih * W_ + (uint32_t)iw;
                            __asm__ volatile("flq2 %0, 0(%1)\n"
                                             : "=f"(v)
                                             : "r"(src));
                        } else {
                            for (uint32_t lane = 0; lane < 8u; lane++) {
                                const int32_t x = iw + (int32_t)lane;
                                edge[lane] =
                                    (x >= 0 && x < (int32_t)W_)
                                        ? in[ic * HW + (uint32_t)ih * W_ +
                                             (uint32_t)x]
                                        : 0.0f;
                            }
                            __asm__ volatile("flq2 %0, 0(%1)\n"
                                             : "=f"(v)
                                             : "r"(edge));
                        }

                        const uint32_t k = ky * 3u + kx;
                        const float *wp =
                            wr_tile + (k * IC + ic) * YOLO_TENSOR_OC + oo;
                        float w0, w1, w2, w3;
#define YOLO_TENSOR_BCAST_W(REG, INDEX)                                        \
    do {                                                                       \
        union {                                                                \
            float f;                                                           \
            uint32_t u;                                                        \
        } _w = {wp[(INDEX)]};                                                  \
        __asm__ volatile("fbcx.ps %0, %1\n"                                    \
                         : "=f"(REG)                                           \
                         : "r"((uint64_t)_w.u));                               \
    } while (0)
                        YOLO_TENSOR_BCAST_W(w0, 0u);
                        YOLO_TENSOR_BCAST_W(w1, 1u);
                        YOLO_TENSOR_BCAST_W(w2, 2u);
                        YOLO_TENSOR_BCAST_W(w3, 3u);
#undef YOLO_TENSOR_BCAST_W
                        __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                         : "+f"(a0)
                                         : "f"(v), "f"(w0));
                        __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                         : "+f"(a1)
                                         : "f"(v), "f"(w1));
                        __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                         : "+f"(a2)
                                         : "f"(v), "f"(w2));
                        __asm__ volatile("fmadd.ps %0, %1, %2, %0\n"
                                         : "+f"(a3)
                                         : "f"(v), "f"(w3));
                    }
                }
            }

#define YOLO_TENSOR_FINISH_ACC(REG, BIAS, INDEX)                               \
    do {                                                                       \
        __asm__ volatile("fadd.ps %0, %0, %1\n" : "+f"(REG) : "f"(BIAS));      \
        if (act == 1u) {                                                       \
            float _t;                                                          \
            __asm__ volatile(                                                  \
                "fsub.ps %[t], %[z], %[x]\n"                                   \
                "fmul.ps %[t], %[t], %[l2e]\n"                                 \
                "fexp.ps %[t], %[t]\n"                                         \
                "fadd.ps %[t], %[t], %[one]\n"                                 \
                "frcp.ps %[t], %[t]\n"                                         \
                "fmul.ps %[t], %[t], %[x]\n"                                   \
                : [t] "=&f"(_t)                                                \
                : [x] "f"(REG), [z] "f"(vz), [one] "f"(vo), [l2e] "f"(vl2e));  \
            (REG) = _t;                                                        \
        }                                                                      \
        __asm__ volatile("fsq2 %1, 0(%0)\n" ::"r"(                             \
                             out + (oc0 + oo + (INDEX)) * HW + oh * W_ + ow8), \
                         "f"(REG)                                              \
                         : "memory");                                          \
    } while (0)
            YOLO_TENSOR_FINISH_ACC(a0, vb0, 0u);
            YOLO_TENSOR_FINISH_ACC(a1, vb1, 1u);
            YOLO_TENSOR_FINISH_ACC(a2, vb2, 2u);
            YOLO_TENSOR_FINISH_ACC(a3, vb3, 3u);
#undef YOLO_TENSOR_FINISH_ACC
        }
    }
}

static inline void
conv2d_3x3_p1_fp32_mh_tensor_center(uint32_t hid, uint8_t *base,
                                    const float *in, float *out, const float *W,
                                    const float *B, uint32_t IC, uint32_t H,
                                    uint32_t W_, uint32_t OC, uint32_t act) {
    float *const WR = (float *)(base + WR3X3_SCRATCH_OFFSET);
    if (!yolo_tensor_can_3x3_p1(IC, H, W_, OC)) {
        conv2d_3x3_p1_disp(hid, WR, in, out, W, B, IC, H, W_, OC, act);
        return;
    }

    yolo_tensor_repack_3x3(hid, W, WR, IC, OC);
    MH_BARRIER();
    if (!mh_is_t0(hid))
        return;

    const uint32_t HW = H * W_;
    const uint32_t oc_tiles = OC / YOLO_TENSOR_OC;
    const uint32_t ic_tiles = IC / YOLO_TENSOR_IC;
    const uint32_t cidx = mh_t0_idx(hid);
    const uint32_t w_tiles = W_ / YOLO_TENSOR_HW;
    const uint32_t spatial_tiles = H * w_tiles;
    const uint32_t tasks = oc_tiles * spatial_tiles;
    uint32_t task_lo, task_hi;
    mh_range(tasks, cidx, &task_lo, &task_hi);

    for (uint32_t task = task_lo; task < task_hi; task++) {
        const uint32_t t_oc = task / spatial_tiles;
        const uint32_t pos = task - t_oc * spatial_tiles;
        const uint32_t oh = pos / w_tiles;
        const uint32_t ow = (pos - oh * w_tiles) * YOLO_TENSOR_HW;
        const uint32_t oc0 = t_oc * YOLO_TENSOR_OC;
        const float *wr_tile = WR + (uint64_t)t_oc * 9u * IC * YOLO_TENSOR_OC;
        for (uint32_t o = 0; o < YOLO_TENSOR_OC; o++)
            evict((const void *)(out + (oc0 + o) * HW + oh * W_ + ow),
                  YOLO_TENSOR_HW * sizeof(float));
        WAIT_CACHEOPS;
        FENCE;

        bool first = true;
        for (uint32_t ky = 0; ky < 3u; ky++) {
            const int32_t ih = (int32_t)oh + (int32_t)ky - 1;
            if (ih < 0 || ih >= (int32_t)H)
                continue;
            const uint32_t k = ky * 3u + 1u;
            for (uint32_t t_ic = 0; t_ic < ic_tiles; t_ic++) {
                const uint32_t ic0 = t_ic * YOLO_TENSOR_IC;
                const float *wtile = wr_tile + (k * IC + ic0) * YOLO_TENSOR_OC;
                const float *atile = in + ic0 * HW + (uint32_t)ih * W_ + ow;

                /* transpose32 turns [ic][oc] cache lines into the
                 * [oc][ic] A matrix consumed by FP32 TFMA. */
                tensor_load(0u, 0u, 0u, 7u, 0u, (uint64_t)wtile, 0u, 15u,
                            YOLO_TENSOR_OC * sizeof(float), 0u);
                tensor_wait(TENSOR_LOAD_WAIT_0);
                tensor_load(0u, 0u, 0u, 0u, 1u, (uint64_t)atile, 0u, 15u,
                            HW * sizeof(float), 1u);
                tensor_wait(TENSOR_LOAD_WAIT_0);
                tensor_fma(0u, 3u, 15u, 15u, 0u, 0u, 0u, 0u, 1u, 0u, 0u, 0u,
                           first);
                tensor_wait(TENSOR_FMA_WAIT);
                first = false;
            }
        }

        float *ctile = out + oc0 * HW + oh * W_ + ow;
        tensor_store(0u, 0u, 3u, 15u, (uint64_t)ctile, 0u, HW * sizeof(float));
        tensor_wait(TENSOR_STORE_WAIT);
        yolo_tensor_clobber_fregs();

        yolo_tensor_finish_3x3_sides(out, in, WR, B, t_oc, IC, H, W_, oh, ow,
                                     act);
        for (uint32_t o = 0; o < YOLO_TENSOR_OC; o++)
            evict((const void *)(out + (oc0 + o) * HW + oh * W_ + ow),
                  YOLO_TENSOR_HW * sizeof(float));
        WAIT_CACHEOPS;
        FENCE;
    }
}

#undef CONV_1x1
#define CONV_1x1(...)                                                          \
    do {                                                                       \
        conv2d_1x1_fp32_mh_tensor(hid, base, __VA_ARGS__);                     \
        MH_BARRIER();                                                          \
    } while (0)

#undef CONV_3x3_P1
#define CONV_3x3_P1(...)                                                       \
    do {                                                                       \
        conv2d_3x3_p1_fp32_mh_tensor_center(hid, base, __VA_ARGS__);           \
        MH_BARRIER();                                                          \
    } while (0)

#endif
