#ifndef CES_MATCHED_FILTER_H
#define CES_MATCHED_FILTER_H

#include <stdint.h>
#include "parameters.h"

// LLR convention used here:
//   LLR = log(P(bit=0)/P(bit=1))
// so:  LLR large => likely 0,  LLR small/negative => likely 1.

typedef struct {
    int   beta_mode;   // 0: use user_beta
                     // 1: beta = 1 / robust_sigma_MAD(S)
                     // 2: beta = 2 / robust_sigma_MAD(S)^2  (stronger)
    float user_beta;

    // for updating c from model:
    float lambda;      // [0..1], trust in model (can be adaptive)
} ces_cfg_t;

// Build b = mG + r2*s + e   (GF(2), cyclic via vect_mul)
void ces_build_b(uint64_t *b,
                 const uint64_t *mG,
                 const uint64_t *r2,
                 const uint64_t *s,
                 const uint64_t *e);

// Build A = r2*h + r1   (GF(2), cyclic via vect_mul)
void ces_build_A(uint64_t *A,
                 const uint64_t *r2,
                 const uint64_t *h,
                 const uint64_t *r1);

// Update y LLR via CES (matched filter):
// Inputs:
//   llr_c        : LLR for c_obs (length PARAM_N)
//   llr_y_prior  : prior LLR for y   (length PARAM_N)
//   A_bits       : A = r2*h + r1  bitvector (VEC_N_SIZE_64)
//   b_bits       : b = mG + r2*s + e bitvector (VEC_N_SIZE_64)
// Outputs:
//   llr_y_new    : updated y LLR (length PARAM_N)
//   y_hat_bits   : optional hard y (weight=PARAM_OMEGA), can be NULL
int ces_update_y_llr(float *llr_y_new,
                     uint64_t *y_hat_bits,
                     const float *llr_c,
                     const float *llr_y_prior,
                     const uint64_t *A_bits,
                     const uint64_t *b_bits,
                     const ces_cfg_t *cfg);

// Compute c_model = b + A*y_hat (GF2), using vect_mul
void ces_compute_c_model(uint64_t *c_model,
                         const uint64_t *A_bits,
                         const uint64_t *b_bits,
                         const uint64_t *y_hat_bits);

// Update c LLR by fusing with model bits:
// llr_c_new = llr_c_prior + lambda*L0*sgn(c_model)
// where sgn = +1 if bit=0 else -1
void ces_update_c_llr_from_model(float *llr_c_new,
                                 const float *llr_c_prior,
                                 const uint64_t *c_model_bits,
                                 float lambda);

// Utility: compute L0 = median(|llr|) (useful if you want adaptive strength)
float ces_median_abs_llr(const float *llr, uint32_t n);

// Optional: adaptive lambda from residual hard mismatch rate
// residual = hard(c_obs from llr_c_prior) XOR c_model
float ces_lambda_from_residual(const float *llr_c_prior,
                               const uint64_t *c_model_bits);

#endif
