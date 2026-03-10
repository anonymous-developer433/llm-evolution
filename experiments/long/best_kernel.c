/*  Optimized Needleman‑Wunsch wave‑front kernel with pointer‑arithmetic
 *  simplifications (index hoisting, pointer bumping) and RVV vectorisation.
 *
 *  Compile with: clang -march=rv64gcv -O2 -DLMUL=2 -DTILE=64
 */

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "blosum62.h"

/* ----------------------------------------------------------------------
 * LMUL tuning: compile with -DLMUL=1,2,4 or 8 (default 2)
 * ---------------------------------------------------------------------- */
#ifndef LMUL
#define LMUL 2
#endif

/* ----------------------------------------------------------------------
 * Resolve intrinsics and vector types for the selected LMUL.
 * ---------------------------------------------------------------------- */
#if LMUL == 1
#define VTYPE          vint32m1_t
#define VSETVL(len)    __riscv_vsetvl_e32m1(len)
#define VLE(ptr,vl)    __riscv_vle32_v_i32m1(ptr,vl)
#define VSE(ptr,vec,vl) __riscv_vse32_v_i32m1(ptr,vec,vl)
#define VLE_STRIDE(ptr,stride,vl) __riscv_vlse32_v_i32m1(ptr,stride,vl)
#define VSE_STRIDE(ptr,stride,vec,vl) __riscv_vsse32_v_i32m1(ptr,stride,vec,vl)
#elif LMUL == 2
#define VTYPE          vint32m2_t
#define VSETVL(len)    __riscv_vsetvl_e32m2(len)
#define VLE(ptr,vl)    __riscv_vle32_v_i32m2(ptr,vl)
#define VSE(ptr,vec,vl) __riscv_vse32_v_i32m2(ptr,vec,vl)
#define VLE_STRIDE(ptr,stride,vl) __riscv_vlse32_v_i32m2(ptr,stride,vl)
#define VSE_STRIDE(ptr,stride,vec,vl) __riscv_vsse32_v_i32m2(ptr,stride,vec,vl)
#elif LMUL == 4
#define VTYPE          vint32m4_t
#define VSETVL(len)    __riscv_vsetvl_e32m4(len)
#define VLE(ptr,vl)    __riscv_vle32_v_i32m4(ptr,vl)
#define VSE(ptr,vec,vl) __riscv_vse32_v_i32m4(ptr,vec,vl)
#define VLE_STRIDE(ptr,stride,vl) __riscv_vlse32_v_i32m4(ptr,stride,vl)
#define VSE_STRIDE(ptr,stride,vec,vl) __riscv_vsse32_v_i32m4(ptr,stride,vec,vl)
#elif LMUL == 8
#define VTYPE          vint32m8_t
#define VSETVL(len)    __riscv_vsetvl_e32m8(len)
#define VLE(ptr,vl)    __riscv_vle32_v_i32m8(ptr,vl)
#define VSE(ptr,vec,vl) __riscv_vse32_v_i32m8(ptr,vec,vl)
#define VLE_STRIDE(ptr,stride,vl) __riscv_vlse32_v_i32m8(ptr,stride,vl)
#define VSE_STRIDE(ptr,stride,vec,vl) __riscv_vsse32_v_i32m8(ptr,stride,vec,vl)
#else
#error "Unsupported LMUL value. Use 1, 2, 4 or 8."
#endif

/* ----------------------------------------------------------------------
 * Helper: max of three int32 values
 * ---------------------------------------------------------------------- */
static inline int32_t imax3(int32_t a, int32_t b, int32_t c)
{
    int32_t ab = a > b ? a : b;
    return ab > c ? ab : c;
}

/* ----------------------------------------------------------------------
 * Tile size – must be >= 1.  Larger tiles improve locality but increase
 * temporary storage.  Adjust according to the target cache size.
 * ---------------------------------------------------------------------- */
#ifndef TILE
#define TILE 64
#endif

/* ----------------------------------------------------------------------
 * Unroll factor for the inner DP loop.
 * ---------------------------------------------------------------------- */
#ifndef UNROLL
#define UNROLL 4
#endif

void sequence_alignment_wavefront(int32_t *H,
                                 int32_t *E,
                                 int32_t *F,
                                 const char *seq1,
                                 size_t n,
                                 const char *seq2,
                                 size_t m,
                                 const int32_t *blosum_flat,
                                 int32_t gap_open,
                                 int32_t gap_extend)
{
    const int32_t NEG_INF = -1000000;
    const size_t cols = m + 1;               /* stride of the DP matrices */

    /* --------------------------------------------------------------
     * 1) Convert sequences to BLOSUM62 indices once.
     * -------------------------------------------------------------- */
    int32_t *idx1 = (int32_t *)malloc(n * sizeof(int32_t));
    int32_t *idx2 = (int32_t *)malloc(m * sizeof(int32_t));
    for (size_t i = 0; i < n; ++i) idx1[i] = blosum62_aa_to_idx(seq1[i]);
    for (size_t j = 0; j < m; ++j) idx2[j] = blosum62_aa_to_idx(seq2[j]);

    /* --------------------------------------------------------------
     * 2) Initialise first row / column (reference behaviour).
     * -------------------------------------------------------------- */
    H[0] = 0;
    E[0] = NEG_INF;
    F[0] = NEG_INF;

    for (size_t j = 1; j <= m; ++j) {
        H[j] = -(gap_open + (int32_t)(j - 1) * gap_extend);
        E[j] = H[j];
        F[j] = NEG_INF;
    }
    for (size_t i = 1; i <= n; ++i) {
        H[i * cols] = -(gap_open + (int32_t)(i - 1) * gap_extend);
        E[i * cols] = NEG_INF;
        F[i * cols] = H[i * cols];
    }

    /* --------------------------------------------------------------
     * 3) Allocate tile buffers and a temporary substitution‑row buffer.
     * -------------------------------------------------------------- */
    const size_t max_tile_rows = TILE + 1;
    const size_t max_tile_cols = TILE + 1;
    int32_t *tile_H = (int32_t *)malloc(max_tile_rows * max_tile_cols * sizeof(int32_t));
    int32_t *tile_E = (int32_t *)malloc(max_tile_rows * max_tile_cols * sizeof(int32_t));
    int32_t *tile_F = (int32_t *)malloc(max_tile_rows * max_tile_cols * sizeof(int32_t));
    int32_t *sub_row = (int32_t *)malloc(TILE * sizeof(int32_t));

    /* --------------------------------------------------------------
     * 4) Process the DP matrix in anti‑diagonal tiles (wave‑front order).
     * -------------------------------------------------------------- */
    for (size_t ti = 0; ti < n; ti += TILE) {
        size_t ti_len = (ti + TILE <= n) ? TILE : n - ti;   /* rows inside this tile */

        for (size_t tj = 0; tj < m; tj += TILE) {
            size_t tj_len = (tj + TILE <= m) ? TILE : m - tj;   /* cols inside this tile */

            /* ------------------------------------------------------
             * 4.1 Copy the top border of the tile (contiguous row)
             * ------------------------------------------------------ */
            {
                size_t avl = tj_len + 1;
                const int32_t *srcH = &H[ti * cols + tj];
                const int32_t *srcE = &E[ti * cols + tj];
                const int32_t *srcF = &F[ti * cols + tj];
                int32_t *dstH = &tile_H[0];
                int32_t *dstE = &tile_E[0];
                int32_t *dstF = &tile_F[0];

                while (avl) {
                    size_t vl = VSETVL(avl);
                    VTYPE vH = VLE(srcH, vl);
                    VTYPE vE = VLE(srcE, vl);
                    VTYPE vF = VLE(srcF, vl);
                    VSE(dstH, vH, vl);
                    VSE(dstE, vE, vl);
                    VSE(dstF, vF, vl);
                    srcH += vl; srcE += vl; srcF += vl;
                    dstH += vl; dstE += vl; dstF += vl;
                    avl -= vl;
                }
            }

            /* ------------------------------------------------------
             * 4.2 Copy the left border of the tile (scalar loads, pointer bumping)
             * ------------------------------------------------------ */
            {
                const int32_t *srcH = &H[ti * cols + tj];
                const int32_t *srcE = &E[ti * cols + tj];
                const int32_t *srcF = &F[ti * cols + tj];
                int32_t *dstH = &tile_H[0];
                int32_t *dstE = &tile_E[0];
                int32_t *dstF = &tile_F[0];
                size_t row_stride = cols;               /* distance between rows in global DP */
                size_t tile_stride = tj_len + 1;         /* distance between rows in tile buffer */

                for (size_t i = 0; i <= ti_len; ++i) {
                    *dstH = *srcH;
                    *dstE = *srcE;
                    *dstF = *srcF;
                    srcH += row_stride;
                    srcE += row_stride;
                    srcF += row_stride;
                    dstH += tile_stride;
                    dstE += tile_stride;
                    dstF += tile_stride;
                }
            }

            /* ------------------------------------------------------
             * 4.3 Scalar wave‑front inside the tile with row‑wise
             *     substitution‑score pre‑computation and unrolled DP.
             * ------------------------------------------------------ */
            for (size_t i = 1; i <= ti_len; ++i) {
                /* pre‑compute substitution scores for this tile row */
                int32_t a_idx = idx1[ti + i - 1];
                size_t base = (size_t)a_idx * BLOSUM62_SIZE;
                const int32_t *idx2_ptr = &idx2[tj];
                for (size_t j = 0; j < tj_len; ++j) {
                    sub_row[j] = blosum_flat[base + (size_t)idx2_ptr[j]];
                }

                /* row base pointers (pointer‑arithmetic version) */
                int32_t *curH  = &tile_H[i * (tj_len + 1)];      /* column 0 of current row */
                int32_t *curE  = &tile_E[i * (tj_len + 1)];
                int32_t *curF  = &tile_F[i * (tj_len + 1)];
                int32_t *prevH = &tile_H[(i - 1) * (tj_len + 1)];
                int32_t *prevF = &tile_F[(i - 1) * (tj_len + 1)];

                size_t j = 1;
                size_t limit = (tj_len / UNROLL) * UNROLL;   /* largest multiple of UNROLL <= tj_len */

                /* ----- unrolled main loop ----- */
                for (; j <= limit; j += UNROLL) {
                    /* iteration 0 (column j) */
                    {
                        int32_t sub = sub_row[j - 1];
                        int32_t e = imax3(curH[j - 1] - gap_open,
                                          curE[j - 1] - gap_extend,
                                          NEG_INF);
                        curE[j] = e;

                        int32_t f = imax3(prevH[j] - gap_open,
                                          prevF[j] - gap_extend,
                                          NEG_INF);
                        curF[j] = f;

                        int32_t h = imax3(prevH[j - 1] + sub, e, f);
                        curH[j] = h;
                    }
                    /* iteration 1 (column j+1) */
                    {
                        int32_t sub = sub_row[j];
                        int32_t e = imax3(curH[j] - gap_open,
                                          curE[j] - gap_extend,
                                          NEG_INF);
                        curE[j + 1] = e;

                        int32_t f = imax3(prevH[j + 1] - gap_open,
                                          prevF[j + 1] - gap_extend,
                                          NEG_INF);
                        curF[j + 1] = f;

                        int32_t h = imax3(prevH[j] + sub, e, f);
                        curH[j + 1] = h;
                    }
                    /* iteration 2 (column j+2) */
                    {
                        int32_t sub = sub_row[j + 1];
                        int32_t e = imax3(curH[j + 1] - gap_open,
                                          curE[j + 1] - gap_extend,
                                          NEG_INF);
                        curE[j + 2] = e;

                        int32_t f = imax3(prevH[j + 2] - gap_open,
                                          prevF[j + 2] - gap_extend,
                                          NEG_INF);
                        curF[j + 2] = f;

                        int32_t h = imax3(prevH[j + 1] + sub, e, f);
                        curH[j + 2] = h;
                    }
                    /* iteration 3 (column j+3) */
                    {
                        int32_t sub = sub_row[j + 2];
                        int32_t e = imax3(curH[j + 2] - gap_open,
                                          curE[j + 2] - gap_extend,
                                          NEG_INF);
                        curE[j + 3] = e;

                        int32_t f = imax3(prevH[j + 3] - gap_open,
                                          prevF[j + 3] - gap_extend,
                                          NEG_INF);
                        curF[j + 3] = f;

                        int32_t h = imax3(prevH[j + 2] + sub, e, f);
                        curH[j + 3] = h;
                    }
                }

                /* ----- tail loop for remaining columns ----- */
                for (; j <= tj_len; ++j) {
                    int32_t sub = sub_row[j - 1];
                    int32_t e = imax3(curH[j - 1] - gap_open,
                                      curE[j - 1] - gap_extend,
                                      NEG_INF);
                    curE[j] = e;

                    int32_t f = imax3(prevH[j] - gap_open,
                                      prevF[j] - gap_extend,
                                      NEG_INF);
                    curF[j] = f;

                    int32_t h = imax3(prevH[j - 1] + sub, e, f);
                    curH[j] = h;
                }
            }

            /* ------------------------------------------------------
             * 4.4 Write back the interior of the tile (vectorised rows)
             * ------------------------------------------------------ */
            for (size_t i = 1; i <= ti_len; ++i) {
                size_t avl = tj_len;
                const int32_t *srcH = &tile_H[i * (tj_len + 1) + 1];
                const int32_t *srcE = &tile_E[i * (tj_len + 1) + 1];
                const int32_t *srcF = &tile_F[i * (tj_len + 1) + 1];
                int32_t *dstH = &H[(ti + i) * cols + (tj + 1)];
                int32_t *dstE = &E[(ti + i) * cols + (tj + 1)];
                int32_t *dstF = &F[(ti + i) * cols + (tj + 1)];

                while (avl) {
                    size_t vl = VSETVL(avl);
                    VTYPE vH = VLE(srcH, vl);
                    VTYPE vE = VLE(srcE, vl);
                    VTYPE vF = VLE(srcF, vl);
                    VSE(dstH, vH, vl);
                    VSE(dstE, vE, vl);
                    VSE(dstF, vF, vl);
                    srcH += vl; srcE += vl; srcF += vl;
                    dstH += vl; dstE += vl; dstF += vl;
                    avl -= vl;
                }
            }
        }
    }

    /* --------------------------------------------------------------
     * 5) Clean‑up
     * -------------------------------------------------------------- */
    free(tile_H);
    free(tile_E);
    free(tile_F);
    free(sub_row);
    free(idx1);
    free(idx2);
    (void)imax3;   /* silence unused‑function warning if optimised away */
}