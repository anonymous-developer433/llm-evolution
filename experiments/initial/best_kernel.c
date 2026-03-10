#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "blosum62.h"

/*  Needleman‑Wunsch with affine gaps, BLOSUM62.
 *  Combines:
 *   • Pre‑computed amino‑acid index profiles (Parent 1)
 *   • Rolling anti‑diagonal buffers for unit‑stride loads (Parent 2)
 *   • Vector length‑agnostic loops with LMUL=4 for higher throughput.
 *
 *  The DP matrices H, E, F are written to the caller‑provided buffers,
 *  guaranteeing bit‑identical results with the reference implementation.
 */
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
    const size_t cols = m + 1;                     /* matrix stride            */
    const ptrdiff_t stride_dp = (ptrdiff_t)(cols - 1) *
                                (ptrdiff_t)sizeof(int32_t); /* stride between
                                                              * consecutive
                                                              * anti‑diagonal
                                                              * elements */

    /* --------------------------------------------------------------
     *  Initialise first row and first column (identical to reference)
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
     *  Pre‑compute amino‑acid index profiles (Parent 1)
     * -------------------------------------------------------------- */
    int8_t *idx1 = (int8_t *)malloc(n * sizeof(int8_t));
    int8_t *idx2 = (int8_t *)malloc(m * sizeof(int8_t));
    for (size_t i = 0; i < n; ++i) idx1[i] = (int8_t)blosum62_aa_to_idx(seq1[i]);
    for (size_t j = 0; j < m; ++j) idx2[j] = (int8_t)blosum62_aa_to_idx(seq2[j]);

    /* --------------------------------------------------------------
     *  Rolling buffers for three anti‑diagonals (Parent 2)
     * -------------------------------------------------------------- */
    const size_t max_len = (n < m ? n : m) + 1;
    int32_t *buf_H[3];
    int32_t *buf_E[3];
    int32_t *buf_F[3];
    size_t   buf_i_start[3];          /* smallest i stored in each buffer */

    for (int k = 0; k < 3; ++k) {
        buf_H[k] = (int32_t *)malloc(max_len * sizeof(int32_t));
        buf_E[k] = (int32_t *)malloc(max_len * sizeof(int32_t));
        buf_F[k] = (int32_t *)malloc(max_len * sizeof(int32_t));
    }

    /* --------------------------------------------------------------
     *  Diagonal d = 0  (cell (0,0))
     * -------------------------------------------------------------- */
    buf_i_start[0] = 0;
    buf_H[0][0] = 0;
    buf_E[0][0] = NEG_INF;
    buf_F[0][0] = NEG_INF;

    /* --------------------------------------------------------------
     *  Diagonal d = 1  (cells (0,1) and (1,0) if they exist)
     * -------------------------------------------------------------- */
    buf_i_start[1] = 0;
    size_t len_d1 = 0;
    if (m >= 1) {                     /* (0,1) */
        H[0 * cols + 1] = -(gap_open + (int32_t)0 * gap_extend);
        E[0 * cols + 1] = H[0 * cols + 1];
        F[0 * cols + 1] = NEG_INF;
        buf_H[1][0] = H[0 * cols + 1];
        buf_E[1][0] = E[0 * cols + 1];
        buf_F[1][0] = F[0 * cols + 1];
        ++len_d1;
    }
    if (n >= 1) {                     /* (1,0) */
        H[1 * cols + 0] = -(gap_open + (int32_t)0 * gap_extend);
        E[1 * cols + 0] = NEG_INF;
        F[1 * cols + 0] = H[1 * cols + 0];
        buf_H[1][len_d1] = H[1 * cols + 0];
        buf_E[1][len_d1] = E[1 * cols + 0];
        buf_F[1][len_d1] = F[1 * cols + 0];
        ++len_d1;
    }

    /* --------------------------------------------------------------
     *  Scratch buffer for substitution scores (max VL for LMUL=4)
     * -------------------------------------------------------------- */
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    int32_t *sub_buf = (int32_t *)malloc(vlmax * sizeof(int32_t));

    /* --------------------------------------------------------------
     *  Main anti‑diagonal loop (d = 2 … n+m)
     * -------------------------------------------------------------- */
    for (size_t d = 2; d <= n + m; ++d) {
        size_t i_start = (d > m) ? (d - m) : 1;
        size_t i_end   = (d - 1 < n) ? (d - 1) : n;
        if (i_start > i_end) continue;               /* empty diagonal */

        size_t cur_idx   = d % 3;
        size_t prev_idx  = (d - 1) % 3;
        size_t prev2_idx = (d - 2) % 3;

        buf_i_start[cur_idx] = i_start;               /* remember for next round */

        size_t processed = 0;
        while (processed < (i_end - i_start + 1)) {
            size_t i = i_start + processed;
            size_t j = d - i;

            /* --------------------------------------------------
             *  Border cells (i==1 or j==1) – scalar path
             * -------------------------------------------------- */
            if (i == 1 || j == 1) {
                int8_t a1 = idx1[i - 1];
                int8_t a2 = idx2[j - 1];
                int32_t sub = (a1 >= 0 && a2 >= 0) ?
                              blosum_flat[(int32_t)a1 * BLOSUM62_SIZE + (int32_t)a2]
                              : -1;

                int32_t H_left = H[i * cols + (j - 1)];
                int32_t E_left = E[i * cols + (j - 1)];
                int32_t H_up   = H[(i - 1) * cols + j];
                int32_t F_up   = F[(i - 1) * cols + j];
                int32_t H_diag = H[(i - 1) * cols + (j - 1)];

                int32_t E_cur = (H_left - gap_open > E_left - gap_extend)
                                ? H_left - gap_open
                                : E_left - gap_extend;
                int32_t F_cur = (H_up - gap_open > F_up - gap_extend)
                                ? H_up - gap_open
                                : F_up - gap_extend;
                int32_t H_cur = H_diag + sub;
                if (E_cur > H_cur) H_cur = E_cur;
                if (F_cur > H_cur) H_cur = F_cur;

                H[i * cols + j] = H_cur;
                E[i * cols + j] = E_cur;
                F[i * cols + j] = F_cur;

                size_t off_cur = i - i_start;
                buf_H[cur_idx][off_cur] = H_cur;
                buf_E[cur_idx][off_cur] = E_cur;
                buf_F[cur_idx][off_cur] = F_cur;

                ++processed;
                continue;
            }

            /* --------------------------------------------------
             *  Vectorised interior cells (i>1 && j>1)
             * -------------------------------------------------- */
            size_t avl = (i_end - i + 1);          /* remaining cells in diagonal */
            size_t max_allowed = j - 1;            /* must keep j-1 > 0 for all lanes */
            if (avl > max_allowed) avl = max_allowed;
            size_t vl = __riscv_vsetvl_e32m4(avl);

            /* ---- substitution scores (profile lookup) ---- */
            for (size_t k = 0; k < vl; ++k) {
                int8_t a1 = idx1[i + k - 1];
                int8_t a2 = idx2[j - k - 1];
                sub_buf[k] = (a1 >= 0 && a2 >= 0) ?
                             blosum_flat[(int32_t)a1 * BLOSUM62_SIZE + (int32_t)a2]
                             : -1;
            }
            vint32m4_t v_sub = __riscv_vle32_v_i32m4(sub_buf, vl);

            /* ---- load neighbours from rolling buffers (unit stride) ---- */
            size_t off_left = i - buf_i_start[prev_idx];
            size_t off_up   = (i - 1) - buf_i_start[prev_idx];
            size_t off_diag = (i - 1) - buf_i_start[prev2_idx];

            vint32m4_t vH_left = __riscv_vle32_v_i32m4(buf_H[prev_idx] + off_left, vl);
            vint32m4_t vE_left = __riscv_vle32_v_i32m4(buf_E[prev_idx] + off_left, vl);
            vint32m4_t vH_up   = __riscv_vle32_v_i32m4(buf_H[prev_idx] + off_up,   vl);
            vint32m4_t vF_up   = __riscv_vle32_v_i32m4(buf_F[prev_idx] + off_up,   vl);
            vint32m4_t vH_diag = __riscv_vle32_v_i32m4(buf_H[prev2_idx] + off_diag, vl);

            /* ---- E[i,j] = max( H[i,j‑1] - gap_open , E[i,j‑1] - gap_extend ) ---- */
            vint32m4_t e_from_h = __riscv_vsub_vx_i32m4(vH_left, gap_open, vl);
            vint32m4_t e_from_e = __riscv_vsub_vx_i32m4(vE_left, gap_extend, vl);
            vint32m4_t vE_cur   = __riscv_vmax_vv_i32m4(e_from_h, e_from_e, vl);

            /* ---- F[i,j] = max( H[i‑1,j] - gap_open , F[i‑1,j] - gap_extend ) ---- */
            vint32m4_t f_from_h = __riscv_vsub_vx_i32m4(vH_up, gap_open, vl);
            vint32m4_t f_from_f = __riscv_vsub_vx_i32m4(vF_up, gap_extend, vl);
            vint32m4_t vF_cur   = __riscv_vmax_vv_i32m4(f_from_h, f_from_f, vl);

            /* ---- H[i,j] = max( H[i‑1,j‑1] + sub , E[i,j] , F[i,j] ) ---- */
            vint32m4_t diag_score = __riscv_vadd_vv_i32m4(vH_diag, v_sub, vl);
            vint32m4_t tmp        = __riscv_vmax_vv_i32m4(diag_score, vE_cur, vl);
            vint32m4_t vH_cur     = __riscv_vmax_vv_i32m4(tmp, vF_cur, vl);

            /* ---- store to full DP matrices (strided) ---- */
            int32_t *H_ptr = &H[i * cols + j];
            int32_t *E_ptr = &E[i * cols + j];
            int32_t *F_ptr = &F[i * cols + j];
            __riscv_vsse32_v_i32m4(H_ptr, stride_dp, vH_cur, vl);
            __riscv_vsse32_v_i32m4(E_ptr, stride_dp, vE_cur, vl);
            __riscv_vsse32_v_i32m4(F_ptr, stride_dp, vF_cur, vl);

            /* ---- store to rolling buffers (contiguous) ---- */
            size_t off_cur = i - i_start;
            __riscv_vse32_v_i32m4(buf_H[cur_idx] + off_cur, vH_cur, vl);
            __riscv_vse32_v_i32m4(buf_E[cur_idx] + off_cur, vE_cur, vl);
            __riscv_vse32_v_i32m4(buf_F[cur_idx] + off_cur, vF_cur, vl);

            processed += vl;
        }
    }

    /* --------------------------------------------------------------
     *  Clean‑up
     * -------------------------------------------------------------- */
    free(idx1);
    free(idx2);
    free(sub_buf);
    for (int k = 0; k < 3; ++k) {
        free(buf_H[k]);
        free(buf_E[k]);
        free(buf_F[k]);
    }
}