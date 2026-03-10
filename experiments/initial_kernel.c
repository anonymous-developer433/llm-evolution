#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "blosum62.h"

static inline int32_t imax3(int32_t a, int32_t b, int32_t c)
{
  int32_t ab = a > b ? a : b;
  return ab > c ? ab : c;
}

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
  // Protein pairwise global alignment (Needleman-Wunsch) with affine gap
  // penalties and BLOSUM62 substitution scoring.
  //
  // Three DP matrices, each (n+1) x (m+1), stored row-major:
  //   H[i,j] = best alignment score for seq1[0..i-1] vs seq2[0..j-1]
  //   E[i,j] = best score ending with a gap in seq1 (horizontal / deletion)
  //   F[i,j] = best score ending with a gap in seq2 (vertical / insertion)
  //
  // Recurrence (maximization):
  //   E[i,j] = max(H[i,j-1] - gap_open, E[i,j-1] - gap_extend)
  //   F[i,j] = max(H[i-1,j] - gap_open, F[i-1,j] - gap_extend)
  //   H[i,j] = max(H[i-1,j-1] + BLOSUM62(seq1[i-1], seq2[j-1]), E[i,j], F[i,j])
  //
  // gap_open is the cost to open a new gap (first position).
  // gap_extend is the cost to extend an existing gap by one more position.

  const int32_t NEG_INF = -1000000;
  size_t cols = m + 1;

  // Initialize base cases.
  H[0] = 0;
  E[0] = NEG_INF;
  F[0] = NEG_INF;

  for (size_t j = 1; j <= m; ++j)
  {
    H[j] = -(gap_open + (int32_t)(j - 1) * gap_extend);
    E[j] = H[j];
    F[j] = NEG_INF;
  }
  for (size_t i = 1; i <= n; ++i)
  {
    H[i * cols] = -(gap_open + (int32_t)(i - 1) * gap_extend);
    E[i * cols] = NEG_INF;
    F[i * cols] = H[i * cols];
  }

  // Wavefront (anti-diagonal) traversal.
  // For a fixed anti-diagonal d = i + j, all cells are independent because
  // they only depend on diagonals d-1 and d-2.
  //
  // Vectorization: within an anti-diagonal, as i increases by 1, j decreases
  // by 1, so DP addresses advance by stride (cols - 1) elements.
  ptrdiff_t stride_dp = (ptrdiff_t)(cols - 1) * (ptrdiff_t)sizeof(int32_t);

  for (size_t d = 2; d <= n + m; ++d)
  {
    size_t i_min = (d > m) ? (d - m) : 1;
    if (i_min < 1)
      i_min = 1;
    size_t i_max = (d - 1 < n) ? (d - 1) : n;

    if (i_min > i_max)
      continue;

    size_t diag_len = i_max - i_min + 1;

    size_t i = i_min;
    while (diag_len > 0)
    {
      size_t vl = __riscv_vsetvl_e32m4(diag_len);
      size_t j = d - i;

      // --- Build BLOSUM62 substitution score vector for this chunk ---
      // For each lane k: score = blosum_flat[idx1 * 20 + idx2]
      // where idx1 = aa_index(seq1[i+k-1]), idx2 = aa_index(seq2[j-k-1]).
      // We use indexed (gather) loads from the flattened BLOSUM62 table.
      //
      // Build index vector: for lane k, index = idx1[i+k-1]*20 + idx2[j-k-1]
      // Since we cannot easily do this purely in vector, we prepare a small
      // scalar buffer of indices and load it.
      int32_t sub_buf[vl];
      for (size_t k = 0; k < vl; ++k)
      {
        int a1 = blosum62_aa_to_idx(seq1[i + k - 1]);
        int a2 = blosum62_aa_to_idx(seq2[j - k - 1]);
        if (a1 >= 0 && a2 >= 0)
          sub_buf[k] = blosum_flat[a1 * BLOSUM62_SIZE + a2];
        else
          sub_buf[k] = -1;
      }
      vint32m4_t v_sub = __riscv_vle32_v_i32m4(sub_buf, vl);

      // --- E matrix: gap in seq1 (horizontal, from left) ---
      // E[i,j] = max(H[i,j-1] - gap_open, E[i,j-1] - gap_extend)
      int32_t *H_left = &H[i * cols + (j - 1)];
      int32_t *E_left = &E[i * cols + (j - 1)];

      vint32m4_t vH_left = __riscv_vlse32_v_i32m4(H_left, stride_dp, vl);
      vint32m4_t vE_left = __riscv_vlse32_v_i32m4(E_left, stride_dp, vl);

      vint32m4_t e_from_h = __riscv_vsub_vx_i32m4(vH_left, gap_open, vl);
      vint32m4_t e_from_e = __riscv_vsub_vx_i32m4(vE_left, gap_extend, vl);
      vint32m4_t vE_cur = __riscv_vmax_vv_i32m4(e_from_h, e_from_e, vl);

      // Store E[i,j]
      int32_t *E_cur = &E[i * cols + j];
      __riscv_vsse32_v_i32m4(E_cur, stride_dp, vE_cur, vl);

      // --- F matrix: gap in seq2 (vertical, from above) ---
      // F[i,j] = max(H[i-1,j] - gap_open, F[i-1,j] - gap_extend)
      int32_t *H_up = &H[(i - 1) * cols + j];
      int32_t *F_up = &F[(i - 1) * cols + j];

      vint32m4_t vH_up = __riscv_vlse32_v_i32m4(H_up, stride_dp, vl);
      vint32m4_t vF_up = __riscv_vlse32_v_i32m4(F_up, stride_dp, vl);

      vint32m4_t f_from_h = __riscv_vsub_vx_i32m4(vH_up, gap_open, vl);
      vint32m4_t f_from_f = __riscv_vsub_vx_i32m4(vF_up, gap_extend, vl);
      vint32m4_t vF_cur = __riscv_vmax_vv_i32m4(f_from_h, f_from_f, vl);

      // Store F[i,j]
      int32_t *F_cur = &F[i * cols + j];
      __riscv_vsse32_v_i32m4(F_cur, stride_dp, vF_cur, vl);

      // --- H matrix: best overall ---
      // H[i,j] = max(H[i-1,j-1] + sub_score, E[i,j], F[i,j])
      int32_t *H_diag = &H[(i - 1) * cols + (j - 1)];
      vint32m4_t vH_diag = __riscv_vlse32_v_i32m4(H_diag, stride_dp, vl);

      vint32m4_t diag_score = __riscv_vadd_vv_i32m4(vH_diag, v_sub, vl);
      vint32m4_t tmp = __riscv_vmax_vv_i32m4(diag_score, vE_cur, vl);
      vint32m4_t vH_cur = __riscv_vmax_vv_i32m4(tmp, vF_cur, vl);

      // Store H[i,j]
      int32_t *H_cur_ptr = &H[i * cols + j];
      __riscv_vsse32_v_i32m4(H_cur_ptr, stride_dp, vH_cur, vl);

      i += vl;
      diag_len -= vl;
    }
  }

  (void)imax3;
}
