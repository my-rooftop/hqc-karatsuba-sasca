#define _CRT_SECURE_NO_WARNINGS
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "code.h"
#include "gf2x.h"
#include "parameters.h"
#include "vector.h"

#ifndef CEIL_DIVIDE
#define CEIL_DIVIDE(a, b) (((a) + (b) - 1) / (b))
#endif

#ifndef VEC_2N_SIZE_BITS
#define VEC_2N_SIZE_BITS (2u * (uint32_t)PARAM_N)
#endif

#ifndef VEC_2N_SIZE_64
#define VEC_2N_SIZE_64 CEIL_DIVIDE(VEC_2N_SIZE_BITS, 64u)
#endif

static inline uint8_t vec_get_bit_u64(const uint64_t *v, uint32_t i) {
    return (uint8_t)((v[i >> 6] >> (i & 63u)) & 1ULL);
}
static inline void vec_set_bit_u64(uint64_t *v, uint32_t i, uint8_t b) {
    uint64_t mask = 1ULL << (i & 63u);
    if (b)
        v[i >> 6] |= mask;
    else
        v[i >> 6] &= ~mask;
}
static inline void vec_zero_u64(uint64_t *v, size_t nwords) {
    memset(v, 0, nwords * sizeof(uint64_t));
}
static inline uint32_t popcount_u64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_popcountll((unsigned long long)x);
#else
    uint32_t c = 0;
    while (x) {
        x &= (x - 1);
        c++;
    }
    return c;
#endif
}
static inline void clear_unused_tail_bits(uint64_t *v, uint32_t n_bits) {
    uint32_t rem = n_bits & 63u;
    if (rem == 0)
        return;
    uint64_t mask = (rem == 64) ? ~0ULL : ((1ULL << rem) - 1ULL);
    uint32_t last = (n_bits >> 6);
    v[last] &= mask;
}
static uint32_t vec_weight_n_bits(const uint64_t *v, uint32_t n_bits) {
    uint32_t full_words = n_bits >> 6;
    uint32_t rem = n_bits & 63u;
    uint32_t w = 0;
    for (uint32_t i = 0; i < full_words; i++) w += popcount_u64(v[i]);
    if (rem) {
        uint64_t mask = (1ULL << rem) - 1ULL;
        w += popcount_u64(v[full_words] & mask);
    }
    return w;
}

static inline float clipf(float x, float lo, float hi) {
    if (x < lo)
        return lo;
    if (x > hi)
        return hi;
    return x;
}
static inline float safe_tanh_half(float L) {
    const float LIM = 20.0f;
    L = clipf(L, -LIM, +LIM);
    return tanhf(0.5f * L);
}
static inline float safe_atanh(float x) {
    const float EPS = 1e-7f;
    if (x > 1.0f - EPS)
        x = 1.0f - EPS;
    if (x < -1.0f + EPS)
        x = -1.0f + EPS;
    return atanhf(x);
}

typedef struct {
    uint32_t t;
    uint32_t idx[512];
} supp_t;

static void build_support(supp_t *S, const uint64_t *r_bits) {
    const uint32_t n = (uint32_t)PARAM_N;
    uint32_t t = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (vec_get_bit_u64(r_bits, i)) {
            if (t < (uint32_t)(sizeof(S->idx) / sizeof(S->idx[0])))
                S->idx[t++] = i;
        }
    }
    if (t == 0) {
        t = 1;
        S->idx[0] = 0;
    }
    S->t = t;
}

static int detect_mul_shift_dir(const uint64_t *r_bits, uint32_t n) {
    uint32_t nwords = (n + 63u) >> 6;
    uint64_t e1[VEC_N_SIZE_64];
    uint64_t u[VEC_N_SIZE_64];
    vec_zero_u64(e1, nwords);
    vec_zero_u64(u, nwords);

    if (n < 2)
        return +1;
    vec_set_bit_u64(e1, 1, 1);
    vect_mul(u, r_bits, e1);
    clear_unused_tail_bits(u, n);

    int first_r = -1, first_u = -1;
    for (uint32_t i = 0; i < n; i++)
        if (vec_get_bit_u64(r_bits, i)) {
            first_r = (int)i;
            break;
        }
    for (uint32_t i = 0; i < n; i++)
        if (vec_get_bit_u64(u, i)) {
            first_u = (int)i;
            break;
        }
    if (first_r < 0 || first_u < 0)
        return +1;

    int diff = (first_u - first_r);
    while (diff < 0) diff += (int)n;
    diff %= (int)n;

    if (diff == 1)
        return +1;
    if (diff == (int)n - 1)
        return -1;

    printf("[SCBP][WARN] detect_mul_shift_dir ambiguous diff=%d, fallback +1\n", diff);
    return +1;
}

static inline uint32_t add_mod_n(uint32_t a, uint32_t b, uint32_t n) {
    uint32_t x = a + b;
    if (x >= n)
        x -= n;
    return x;
}
static inline uint32_t sub_mod_n(uint32_t a, uint32_t b, uint32_t n) {
    return (a >= b) ? (a - b) : (a + n - b);
}
static inline uint32_t check_to_var(uint32_t j, uint32_t p, uint32_t n, int dir) {
    return (dir > 0) ? sub_mod_n(j, p, n) : add_mod_n(j, p, n);
}
static inline size_t edge_index(uint32_t i, uint32_t k, uint32_t t) {
    return (size_t)i * (size_t)t + (size_t)k;
}

static void build_llr_from_absL_and_hard(float *L_out, const float *absL, const uint64_t *hard_bits, uint32_t n_bits) {
    for (uint32_t i = 0; i < n_bits; i++) {
        float a = absL[i];
        uint8_t b = vec_get_bit_u64(hard_bits, i);
        L_out[i] = b ? (-a) : (+a);
    }
}

typedef struct {
    uint32_t idx;
    float key;
} item_t;
static int cmp_item_desc(const void *A, const void *B) {
    float a = ((const item_t *)A)->key;
    float b = ((const item_t *)B)->key;
    return (a > b) ? -1 : (a < b);
}
static void hard_top_omega_from_L(uint64_t *out_bits, const float *L, uint32_t n, uint32_t omega) {
    uint32_t nwords = (n + 63u) >> 6;
    vec_zero_u64(out_bits, nwords);
    if (omega > n)
        omega = n;

    item_t *arr = (item_t *)malloc(sizeof(item_t) * (size_t)n);
    if (!arr) {
        printf("[SCBP][FATAL] alloc fail hard_top_omega\n");
        exit(1);
    }

    for (uint32_t i = 0; i < n; i++) {
        arr[i].idx = i;
        arr[i].key = -L[i];
    }
    qsort(arr, n, sizeof(item_t), cmp_item_desc);

    for (uint32_t k = 0; k < omega; k++) vec_set_bit_u64(out_bits, arr[k].idx, 1);
    free(arr);
}

typedef struct {
    uint32_t idx;
    float key;
} ck_t;
static int cmp_ck_desc(const void *A, const void *B) {
    float a = ((const ck_t *)A)->key;
    float b = ((const ck_t *)B)->key;
    return (a > b) ? -1 : (a < b);
}
static uint32_t build_conf_mask(uint8_t *mask, const float *Lc, uint32_t n, float frac_keep) {
    if (!(frac_keep > 0.0f))
        frac_keep = 1.0f;
    if (frac_keep > 1.0f)
        frac_keep = 1.0f;

    uint32_t K = (uint32_t)floorf(frac_keep * (float)n);
    if (K < 1)
        K = 1;
    if (K > n)
        K = n;

    ck_t *arr = (ck_t *)malloc(sizeof(ck_t) * (size_t)n);
    if (!arr) {
        printf("[SCBP][FATAL] alloc fail conf mask\n");
        exit(1);
    }

    for (uint32_t i = 0; i < n; i++) {
        mask[i] = 0;
        arr[i].idx = i;
        arr[i].key = fabsf(Lc[i]);
    }
    qsort(arr, n, sizeof(ck_t), cmp_ck_desc);
    for (uint32_t t = 0; t < K; t++) mask[arr[t].idx] = 1;

    free(arr);
    return K;
}

static void var_update(float *m_v2c, float *Lpost, const float *m_c2v, const float *Lprior, uint32_t n, uint32_t t,
                       float damping_v, float llr_max) {
    for (uint32_t i = 0; i < n; i++) {
        float sum_all = 0.0f;
        size_t base = (size_t)i * (size_t)t;
        for (uint32_t k = 0; k < t; k++) sum_all += m_c2v[base + k];

        Lpost[i] = clipf(Lprior[i] + sum_all, -llr_max, +llr_max);

        for (uint32_t k = 0; k < t; k++) {
            float excl = sum_all - m_c2v[base + k];
            float newm = clipf(Lprior[i] + excl, -llr_max, +llr_max);
            float oldm = m_v2c[base + k];
            m_v2c[base + k] = (1.0f - damping_v) * newm + damping_v * oldm;
        }
    }
}

static void check_update_single(float *m_c2v, const float *m_v2c, const float *Lb, const uint8_t *mask_c,
                                const supp_t *S, int dir, uint32_t n, float damping_c, float llr_max) {
    const float TINY = 1e-12f;
    float tanhv[512];

    for (uint32_t j = 0; j < n; j++) {
        float Tb = (mask_c && !mask_c[j]) ? 0.0f : safe_tanh_half(Lb[j]);

        if (Tb == 0.0f) {
            for (uint32_t k = 0; k < S->t; k++) {
                uint32_t p = S->idx[k];
                uint32_t i = check_to_var(j, p, n, dir);
                size_t e = edge_index(i, k, S->t);
                float oldm = m_c2v[e];
                m_c2v[e] = (1.0f - damping_c) * 0.0f + damping_c * oldm;
            }
            continue;
        }

        float P_all = 1.0f;
        for (uint32_t k = 0; k < S->t; k++) {
            uint32_t p = S->idx[k];
            uint32_t i = check_to_var(j, p, n, dir);
            size_t e = edge_index(i, k, S->t);
            float tv = safe_tanh_half(m_v2c[e]);
            tanhv[k] = tv;
            P_all *= tv;
        }

        for (uint32_t k = 0; k < S->t; k++) {
            uint32_t p = S->idx[k];
            uint32_t i = check_to_var(j, p, n, dir);
            size_t e = edge_index(i, k, S->t);

            float denom = tanhv[k];
            float P_excl;
            if (fabsf(denom) < TINY) {
                P_excl = 1.0f;
                for (uint32_t kk = 0; kk < S->t; kk++)
                    if (kk != k)
                        P_excl *= tanhv[kk];
            } else {
                P_excl = P_all / denom;
            }

            float arg = Tb * P_excl;
            float newm = 2.0f * safe_atanh(arg);
            newm = clipf(newm, -llr_max, +llr_max);

            float oldm = m_c2v[e];
            m_c2v[e] = (1.0f - damping_c) * newm + damping_c * oldm;
        }
    }
}

static void run_bp_single_equation(float *Lpost_out, const float *Lb_obs, const float *Lprior, const uint8_t *mask_c,
                                   const supp_t *S, int dir, uint32_t n, uint32_t iters, float damping_v,
                                   float damping_c, float llr_max, int verbose, const char *tag) {
    size_t E = (size_t)n * (size_t)S->t;
    float *m_v2c = (float *)malloc(sizeof(float) * E);
    float *m_c2v = (float *)malloc(sizeof(float) * E);
    if (!m_v2c || !m_c2v) {
        printf("[SCBP][FATAL] alloc fail BP messages\n");
        exit(1);
    }

    for (uint32_t i = 0; i < n; i++) {
        size_t base = (size_t)i * (size_t)S->t;
        for (uint32_t k = 0; k < S->t; k++) {
            m_v2c[base + k] = clipf(Lprior[i], -llr_max, +llr_max);
            m_c2v[base + k] = 0.0f;
        }
    }

    for (uint32_t it = 0; it < iters; it++) {
        check_update_single(m_c2v, m_v2c, Lb_obs, mask_c, S, dir, n, damping_c, llr_max);
        var_update(m_v2c, Lpost_out, m_c2v, Lprior, n, S->t, damping_v, llr_max);

        if (verbose && (it == 0 || ((it + 1) % 10 == 0) || (it + 1 == iters))) {
            double a = 0.0;
            for (uint32_t i = 0; i < n; i++) a += fabs(Lpost_out[i]);
            a /= (double)n;
            printf("[SCBP][%s] it=%u/%u avg|L|=%.3f\n", tag ? tag : "BP", it + 1, iters, (float)a);
            fflush(stdout);
        }
    }

    free(m_v2c);
    free(m_c2v);
}

static void llr_xor_known(float *Ldst, const float *Lsrc, const uint64_t *known_bits, uint32_t n) {
    for (uint32_t j = 0; j < n; j++) {
        float v = Lsrc[j];
        if (vec_get_bit_u64(known_bits, j))
            v = -v;
        Ldst[j] = v;
    }
}

static void dump_prob_txt_from_llr(const char *path, const float *L, uint32_t n) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        printf("[SCBP][WARN] cannot open %s\n", path);
        return;
    }

    const double EPS = 1e-12;
    for (uint32_t i = 0; i < n; i++) {
        double llr = (double)L[i];
        double p1 = 1.0 / (1.0 + exp(llr));
        if (p1 < EPS)
            p1 = EPS;
        if (p1 > 1.0 - EPS)
            p1 = 1.0 - EPS;
        double p0 = 1.0 - p1;
        fprintf(fp, "%.12f %.12f\n", p0, p1);
    }
    fclose(fp);
}

typedef struct {
    uint32_t omega;
    uint32_t iters_x;
    uint32_t iters_y;
    uint32_t outer_rounds;
    float damping_v;
    float damping_c;
    float llr_max;
    float prior_wt_strength;
    float y_prior_scale;
    float conf_frac_c;
    int verbose;
} SCBP_CFG;

typedef struct {
    int found;
    float score;
    uint64_t x_hat[VEC_N_SIZE_64];
    uint64_t y_hat[VEC_N_SIZE_64];
} best2_t;

void run_softcancel_bp_keyrecovery(best2_t *best, const uint64_t *r1, const uint64_t *r2, const uint64_t *base_bits,
                                   const uint64_t *c_sca, const float *absL_c, const uint64_t *y_sca,
                                   const float *absL_y, const uint64_t *x_true, const uint64_t *y_true,
                                   const SCBP_CFG *cfg, int dump_probs) {
    const uint32_t n = (uint32_t)PARAM_N;
    const uint32_t nwords = (n + 63u) >> 6;

    uint64_t base_local[VEC_N_SIZE_64];
    uint64_t csca_local[VEC_N_SIZE_64];
    memcpy(base_local, base_bits, sizeof(uint64_t) * nwords);
    memcpy(csca_local, c_sca, sizeof(uint64_t) * nwords);
    clear_unused_tail_bits(base_local, n);
    clear_unused_tail_bits(csca_local, n);

    supp_t S1, S2;
    build_support(&S1, r1);
    build_support(&S2, r2);
    int dir1 = detect_mul_shift_dir(r1, n);
    int dir2 = detect_mul_shift_dir(r2, n);

    float *Lc = (float *)malloc(sizeof(float) * (size_t)n);
    float *Ly_raw = (float *)malloc(sizeof(float) * (size_t)n);
    float *Lb = (float *)malloc(sizeof(float) * (size_t)n);
    float *Lb_p = (float *)malloc(sizeof(float) * (size_t)n);
    float *Lb_y = (float *)malloc(sizeof(float) * (size_t)n);

    float *Lx_prior = (float *)malloc(sizeof(float) * (size_t)n);
    float *Ly_prior = (float *)malloc(sizeof(float) * (size_t)n);
    float *Lx_post = (float *)malloc(sizeof(float) * (size_t)n);
    float *Ly_post = (float *)malloc(sizeof(float) * (size_t)n);

    uint8_t *mask_c = (uint8_t *)malloc(sizeof(uint8_t) * (size_t)n);

    if (!Lc || !Ly_raw || !Lb || !Lb_p || !Lb_y || !Lx_prior || !Ly_prior || !Lx_post || !Ly_post || !mask_c) {
        printf("[SCBP][FATAL] alloc fail arrays\n");
        exit(1);
    }

    build_llr_from_absL_and_hard(Lc, absL_c, csca_local, n);
    build_llr_from_absL_and_hard(Ly_raw, absL_y, y_sca, n);
    llr_xor_known(Lb, Lc, base_local, n);

    float keep = cfg->conf_frac_c;
    if (!(keep > 0.0f) || keep > 1.0f)
        keep = 0.30f;
    uint32_t Kmask = build_conf_mask(mask_c, Lc, n, keep);

    float Lw = 0.0f;
    {
        double p1 = (double)cfg->omega / (double)n;
        if (p1 < 1e-12)
            p1 = 1e-12;
        if (p1 > 1.0 - 1e-12)
            p1 = 1.0 - 1e-12;
        Lw = (float)log((1.0 - p1) / p1);
        Lw *= (cfg->prior_wt_strength > 0.0f) ? cfg->prior_wt_strength : 0.0f;
    }

    for (uint32_t i = 0; i < n; i++) {
        Lx_prior[i] = clipf(+Lw, -cfg->llr_max, +cfg->llr_max);
        Ly_prior[i] = clipf(cfg->y_prior_scale * Ly_raw[i] + Lw, -cfg->llr_max, +cfg->llr_max);
        Ly_post[i] = Ly_prior[i];
        Lx_post[i] = Lx_prior[i];
    }

    if (cfg->verbose) {
        printf("[SCBP] n=%u omega=%u supp(r1)=%u supp(r2)=%u dir1=%+d dir2=%+d\n", n, cfg->omega, S1.t, S2.t, dir1,
               dir2);
        printf("[SCBP] conf_frac_c=%.3f => Kmask=%u ; prior_wt_strength=%.3f (Lw=%.3f)\n", keep, Kmask,
               cfg->prior_wt_strength, Lw);
        fflush(stdout);
    }

    uint64_t y_hat[VEC_N_SIZE_64];
    uint64_t x_hat[VEC_N_SIZE_64];
    uint64_t ty[VEC_N_SIZE_64];
    uint64_t tx[VEC_N_SIZE_64];

    uint32_t R = (cfg->outer_rounds >= 1) ? cfg->outer_rounds : 1;
    for (uint32_t round = 0; round < R; round++) {
        hard_top_omega_from_L(y_hat, Ly_post, n, cfg->omega);
        clear_unused_tail_bits(y_hat, n);

        vec_zero_u64(ty, nwords);
        vect_mul(ty, r1, y_hat);
        clear_unused_tail_bits(ty, n);

        for (uint32_t j = 0; j < n; j++) {
            float v = Lb[j];
            if (vec_get_bit_u64(ty, j))
                v = -v;
            Lb_p[j] = v;
        }

        run_bp_single_equation(Lx_post, Lb_p, Lx_prior, mask_c, &S2, dir2, n, cfg->iters_x, cfg->damping_v,
                               cfg->damping_c, cfg->llr_max, cfg->verbose, "X");

        hard_top_omega_from_L(x_hat, Lx_post, n, cfg->omega);
        clear_unused_tail_bits(x_hat, n);

        if (cfg->iters_y > 0) {
            vec_zero_u64(tx, nwords);
            vect_mul(tx, r2, x_hat);
            clear_unused_tail_bits(tx, n);

            for (uint32_t j = 0; j < n; j++) {
                float v = Lb[j];
                if (vec_get_bit_u64(tx, j))
                    v = -v;
                Lb_y[j] = v;
            }

            run_bp_single_equation(Ly_post, Lb_y, Ly_prior, mask_c, &S1, dir1, n, cfg->iters_y, cfg->damping_v,
                                   cfg->damping_c, cfg->llr_max, cfg->verbose, "Y");
        }

        if (cfg->verbose) {
            uint32_t wx = vec_weight_n_bits(x_hat, n);
            uint32_t wy = vec_weight_n_bits(y_hat, n);
            printf("[SCBP] round=%u/%u wt(x_hat)=%u wt(y_hat)=%u\n", round + 1, R, wx, wy);
            fflush(stdout);
        }
    }

    hard_top_omega_from_L(x_hat, Lx_post, n, cfg->omega);
    hard_top_omega_from_L(y_hat, Ly_post, n, cfg->omega);
    clear_unused_tail_bits(x_hat, n);
    clear_unused_tail_bits(y_hat, n);

    best->found = 1;
    best->score = 0.0f;
    memcpy(best->x_hat, x_hat, sizeof(uint64_t) * nwords);
    memcpy(best->y_hat, y_hat, sizeof(uint64_t) * nwords);

    {
        uint64_t bx[VEC_N_SIZE_64], by[VEC_N_SIZE_64], bhat[VEC_N_SIZE_64], bhard[VEC_N_SIZE_64];
        vec_zero_u64(bx, nwords);
        vec_zero_u64(by, nwords);

        vect_mul(bx, r2, best->x_hat);
        vect_mul(by, r1, best->y_hat);
        for (uint32_t w = 0; w < nwords; w++) bhat[w] = bx[w] ^ by[w];
        clear_unused_tail_bits(bhat, n);

        for (uint32_t w = 0; w < nwords; w++) bhard[w] = csca_local[w] ^ base_local[w];
        clear_unused_tail_bits(bhard, n);

        uint32_t wr = 0;
        for (uint32_t i = 0; i < n; i++) wr += (uint32_t)(vec_get_bit_u64(bhat, i) ^ vec_get_bit_u64(bhard, i));
        printf("[SCBP][SANITY] wr = %u\n", wr);
    }

    if (cfg->verbose) {
        printf("[SCBP] DONE wt(x)=%u wt(y)=%u\n", vec_weight_n_bits(best->x_hat, n), vec_weight_n_bits(best->y_hat, n));
        if (x_true && y_true) {
            uint32_t ex = 0, ey = 0;
            for (uint32_t i = 0; i < n; i++) {
                ex += (uint32_t)(vec_get_bit_u64(best->x_hat, i) ^ vec_get_bit_u64(x_true, i));
                ey += (uint32_t)(vec_get_bit_u64(best->y_hat, i) ^ vec_get_bit_u64(y_true, i));
            }
            printf("[SCBP] err(x)=%u err(y)=%u\n", ex, ey);
        }
        fflush(stdout);
    }

    if (dump_probs) {
        dump_prob_txt_from_llr("scbp_x_prob.txt", Lx_post, n);
        dump_prob_txt_from_llr("scbp_y_prob.txt", Ly_post, n);
        if (cfg->verbose) {
            printf("[SCBP] dumped: scbp_x_prob.txt / scbp_y_prob.txt\n");
            fflush(stdout);
        }
    }

    free(Lc);
    free(Ly_raw);
    free(Lb);
    free(Lb_p);
    free(Lb_y);
    free(Lx_prior);
    free(Ly_prior);
    free(Lx_post);
    free(Ly_post);
    free(mask_c);
}

static void bytes_to_vec_mem(uint64_t *dst64, const uint8_t *src_bytes, size_t src_len_bytes) {
    memset(dst64, 0, sizeof(uint64_t) * VEC_N_SIZE_64);
    memcpy((uint8_t *)dst64, src_bytes, src_len_bytes);
    dst64[VEC_N_SIZE_64 - 1] &= BITMASK(PARAM_N, 64);
}
static int hexval(char c) {
    if ('0' <= c && c <= '9')
        return c - '0';
    if ('a' <= c && c <= 'f')
        return 10 + (c - 'a');
    if ('A' <= c && c <= 'F')
        return 10 + (c - 'A');
    return -1;
}
static int hex_to_bytes_prefix(const char *hex, uint8_t *out, size_t out_len) {
    while (*hex == ' ' || *hex == '\t' || *hex == '\n' || *hex == '\r') hex++;
    if (hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X'))
        hex += 2;

    size_t hl = 0;
    for (const char *p = hex; *p; p++) {
        if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
            continue;
        if (hexval(*p) < 0)
            return -1;
        hl++;
    }
    if (hl < out_len * 2)
        return -2;

    size_t bi = 0;
    int high = -1;
    for (const char *p = hex; *p && bi < out_len; p++) {
        if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
            continue;
        int v = hexval(*p);
        if (high < 0)
            high = v;
        else {
            out[bi++] = (uint8_t)((high << 4) | v);
            high = -1;
        }
    }
    return (bi == out_len) ? 0 : -3;
}

static int read_u64_array_from_txt(const char *filename, uint64_t *out, size_t out_len) {
    FILE *fp = fopen(filename, "r");
    if (!fp)
        return -1;

    size_t i = 0;
    char buf[256];

    while (i < out_len) {
        int ret = fscanf(fp, "%255s", buf);
        if (ret == 1) {
            errno = 0;
            char *end = NULL;
            unsigned long long v = strtoull(buf, &end, 0);
            if (end == buf || *end != '\0') {
                fclose(fp);
                return -3;
            }
            if (errno == ERANGE) {
                fclose(fp);
                return -4;
            }
            out[i++] = (uint64_t)v;
        } else if (ret == EOF) {
            fclose(fp);
            return -2;
        } else {
            fclose(fp);
            return -3;
        }
    }

    fclose(fp);
    return 0;
}

int load_y_sca_from_prob_txt(const char *path, uint64_t *y_sca, uint32_t n_bits) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    uint32_t n_words = (n_bits + 63) >> 6;
    memset(y_sca, 0, n_words * sizeof(uint64_t));

    double p0, p1;
    uint32_t bit_idx = 0;

    while (bit_idx < n_bits && fscanf(fp, "%lf %lf", &p0, &p1) == 2) {
        if (p1 > p0) {
            uint32_t w = bit_idx >> 6;
            uint32_t b = bit_idx & 63;
            y_sca[w] |= (1ULL << b);
        }
        bit_idx++;
    }

    fclose(fp);
    if (bit_idx < n_bits)
        return -2;
    return 0;
}

typedef struct {
    uint32_t idx;
    double p0;
} p0_item_t;
static int cmp_p0_asc(const void *a, const void *b) {
    double x = ((const p0_item_t *)a)->p0;
    double y = ((const p0_item_t *)b)->p0;
    return (x < y) ? -1 : (x > y);
}
int load_y_sca_from_prob_txt_top66_p0smallest(const char *path, uint64_t *y_sca, uint32_t n_bits, uint32_t omega) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    uint32_t n_words = (n_bits + 63) >> 6;
    memset(y_sca, 0, n_words * sizeof(uint64_t));

    p0_item_t *arr = (p0_item_t *)malloc(sizeof(p0_item_t) * n_bits);
    if (!arr) {
        fclose(fp);
        return -3;
    }

    const double EPS = 1e-300;
    double p0, p1;
    uint32_t bit_idx = 0;

    while (bit_idx < n_bits && fscanf(fp, "%lf %lf", &p0, &p1) == 2) {
        if (p0 < EPS)
            p0 = EPS;
        if (p1 < EPS)
            p1 = EPS;
        arr[bit_idx].idx = bit_idx;
        arr[bit_idx].p0 = p0;
        bit_idx++;
    }

    fclose(fp);
    if (bit_idx < n_bits) {
        free(arr);
        return -2;
    }

    if (omega > n_bits)
        omega = n_bits;
    qsort(arr, n_bits, sizeof(p0_item_t), cmp_p0_asc);

    for (uint32_t k = 0; k < omega; k++) {
        uint32_t i = arr[k].idx;
        y_sca[i >> 6] |= (1ULL << (i & 63));
    }

    free(arr);
    return 0;
}

static int load_absL_y_from_prob_txt(const char *path, float *absL_y, uint32_t n_bits) {
    FILE *fp = fopen(path, "r");
    if (!fp)
        return -1;

    const double EPS = 1e-300;
    double p0, p1;
    uint32_t i = 0;

    while (i < n_bits && fscanf(fp, "%lf %lf", &p0, &p1) == 2) {
        if (p0 < EPS)
            p0 = EPS;
        if (p1 < EPS)
            p1 = EPS;
        double llr = log(p0) - log(p1);
        absL_y[i++] = (float)fabs(llr);
    }

    fclose(fp);
    return (i == n_bits) ? 0 : -2;
}

uint32_t count_bit_mismatch_n(const uint64_t *a, const uint64_t *b, uint32_t n_bits) {
    uint32_t mism = 0;
    for (uint32_t i = 0; i < n_bits; i++) mism += (uint32_t)(vec_get_bit_u64(a, i) ^ vec_get_bit_u64(b, i));
    return mism;
}

static inline uint8_t vec_get_bit_local(const uint64_t *v, uint32_t i) {
    return (uint8_t)((v[i >> 6] >> (i & 63)) & 1ULL);
}
static uint32_t vec_weight_n_local(const uint64_t *v) {
    uint32_t w = 0;
    for (uint32_t i = 0; i < PARAM_N; i++) w += vec_get_bit_local(v, i);
    return w;
}
/*
 * This experiment uses a collection of input files and runtime buffers that
 * represent binary vectors of length PARAM_N, all stored in bit-packed
 * uint64_t arrays (VEC_N_SIZE_64 words). Each bit corresponds to one coordinate
 * of the underlying vector, and any unused tail bits must be cleared after
 * loading from file.
 *
 * The files include noisy side-channel predictions of the ciphertext c and the
 * secret vectors x and y (PRED_C_TXT, PRED_X_TXT, PRED_Y_TXT), which are used as
 * initial estimates or soft/hard decisions in the key-recovery process.
 * Ground-truth reference files (Y_TRUE_TXT, X_TRUE_TXT, C_TRUE_TXT) provide the
 * true values of y, x, and c, and are used only for validation, mismatch
 * counting, and debugging.
 *
 * The remaining input files correspond to public or known HQC parameters and
 * derived vectors: r1 and r2 are sparse vectors used to form the products
 * r1 * y and r2 * x; e is a known fixed-weight error vector (not random noise);
 * h is a public vector; s is defined as s = x + h * y; and m is a
 * message-related vector used in the construction of mG. All of these are
 * stored and processed as uint64_t bit vectors.
 *
 * At runtime, the arrays y_true, x_true, and c_true hold the true secret and
 * ciphertext vectors, while h, s, r1, r2, m, and e store the corresponding
 * known parameters and intermediate values. A fixed random seed is used to
 * ensure reproducibility of all randomized components of the experiment
 * (e.g., shuffling, random restarts, or beam search).
 */
int main(void) {
    const char *PRED_C_TXT =
        "C:/Users/***/Desktop/word/hqc-vs/src/ref/predictions/predictions/predictions_c_noise6.txt";
    const char *PRED_Y_TXT =
        "C:/Users/***/Desktop/word/hqc-vs/src/ref/predictions/predictions/predictions_y_noise_new_6.txt";
    const char *PRED_X_TXT = "C:/User/***/Desktop/word/hqc-vs/src/ref/x_probabilities_noise2.txt";

    const char *Y_TRUE_TXT = "C:/Users/***/Desktop/word/hqc-vs/src/ref/y.txt";
    const char *X_TRUE_TXT = "C:/Users/***/Desktop/word/hqc-vs/src/ref/x.txt";
    const char *C_TRUE_TXT = "C:/Users/***/Desktop/word/hqc-vs/src/ref/c.txt";
    const char *R1_TXT = "C:/Users/***/Desktop/word/hqc-vs/src/ref/r1.txt";
    const char *R2_TXT = "C:/Users/***/Desktop/word/hqc-vs/src/ref/r2.txt";
    const char *E_TXT = "C:/Users/***/Desktop/word/hqc-vs/src/ref/e.txt";
    const char *H_TXT = "C:/Users/***/Desktop/word/hqc-vs/src/ref/h.txt";
    const char *S_TXT = "C:/Users/***/Desktop/word/hqc-vs/src/ref/s.txt";
    const char *M_TXT = "C:/Users/***/Desktop/word/hqc-vs/src/ref/m.txt";

    uint64_t seed = 20251222;

    uint64_t y_true[VEC_N_SIZE_64], x_true[VEC_N_SIZE_64];
    uint64_t c_true[VEC_N_SIZE_64];
    uint64_t h[VEC_N_SIZE_64], s[VEC_N_SIZE_64], r1[VEC_N_SIZE_64], r2[VEC_N_SIZE_64], m[VEC_N_SIZE_64],
        e[VEC_N_SIZE_64];

    if (read_u64_array_from_txt(Y_TRUE_TXT, y_true, VEC_N_SIZE_64) != 0) {
        printf("parse y fail\n");
        return 1;
    }
    if (read_u64_array_from_txt(X_TRUE_TXT, x_true, VEC_N_SIZE_64) != 0) {
        printf("parse x fail\n");
        return 1;
    }
    if (read_u64_array_from_txt(C_TRUE_TXT, c_true, VEC_N_SIZE_64) != 0) {
        printf("parse c fail\n");
        return 1;
    }
    if (read_u64_array_from_txt(R1_TXT, r1, VEC_N_SIZE_64) != 0) {
        printf("parse r1 fail\n");
        return 1;
    }
    if (read_u64_array_from_txt(R2_TXT, r2, VEC_N_SIZE_64) != 0) {
        printf("parse r2 fail\n");
        return 1;
    }
    if (read_u64_array_from_txt(E_TXT, e, VEC_N_SIZE_64) != 0) {
        printf("parse e fail\n");
        return 1;
    }
    if (read_u64_array_from_txt(H_TXT, h, VEC_N_SIZE_64) != 0) {
        printf("parse h fail\n");
        return 1;
    }
    if (read_u64_array_from_txt(S_TXT, s, VEC_N_SIZE_64) != 0) {
        printf("parse s fail\n");
        return 1;
    }
    if (read_u64_array_from_txt(M_TXT, m, 16) != 0) {
        printf("parse m fail\n");
        return 1;
    }


    printf("[info] wt(y)=%u\n", vec_weight_n_local(y_true));

    uint64_t y_sca[VEC_N_SIZE_64], x_sca[VEC_N_SIZE_64], c_sca[VEC_N_SIZE_64];

    load_y_sca_from_prob_txt(PRED_C_TXT, c_sca, 17664);
    load_y_sca_from_prob_txt_top66_p0smallest(PRED_Y_TXT, y_sca, PARAM_N, 66);
    load_y_sca_from_prob_txt_top66_p0smallest(PRED_X_TXT, x_sca, PARAM_N, 66);

    uint32_t diff_y0 = count_bit_mismatch_n(y_true, y_sca, 17669);
    printf("init y mismatch bits = %u\n", diff_y0);

    uint32_t diff_c0 = count_bit_mismatch_n(c_true, c_sca, 17669);
    printf("c mismatch bits = %u\n", diff_c0);

    float *absL_y = (float *)malloc(sizeof(float) * PARAM_N);
    float *absL_x = (float *)malloc(sizeof(float) * PARAM_N);
    float *absL_c = (float *)malloc(sizeof(float) * PARAM_N);

    load_absL_y_from_prob_txt(PRED_Y_TXT, absL_y, PARAM_N);
    load_absL_y_from_prob_txt(PRED_X_TXT, absL_x, PARAM_N);
    load_absL_y_from_prob_txt(PRED_C_TXT, absL_c, PARAM_N);

    best2_t best;
    uint64_t base[VEC_N_SIZE_64];
    uint64_t mG[VEC_N_SIZE_64];
    code_encode(mG, m);
    vect_add(base, mG, e, VEC_N_SIZE_64);
    /*
     * SCBP (Soft-Cancel Belief Propagation) parameters.
     *
     * A turbo-like decoding scheme alternating between BP on r2*x and r1*y with
     * decision-directed interference cancellation. The parameter omega enforces
     * the target Hamming weight of x and y via hard projection and a sparsity prior.
     * iters_x / iters_y specify the number of BP iterations for the x- and y-decoders
     * per turbo round, and outer_rounds controls the total number of turbo iterations.
     * damping_v and damping_c apply message damping for BP stabilization on MDPC
     * graphs. llr_max limits LLR magnitude to avoid numerical issues. prior_wt_strength
     * and y_prior_scale control the influence of sparsity and side-channel priors.
     * conf_frac_c selects the fraction of reliable ciphertext constraints used in BP.
     */
    SCBP_CFG cfg = {.omega = 66,
                    .iters_x = 10,
                    .iters_y = 10,
                    .outer_rounds = 3,
                    .damping_v = 0.35f,
                    .damping_c = 0.35f,
                    .llr_max = 32.0f,
                    .prior_wt_strength = 1.0f,
                    .y_prior_scale = 1.0f,
                    .conf_frac_c = 0.50f,
                    .verbose = 1};

    run_softcancel_bp_keyrecovery(&best, r1, r2, base, c_sca, absL_c, y_sca, absL_y, NULL, NULL, &cfg, 1);

    printf("[y_hat] ");
    vect_print(best.y_hat, VEC_N_SIZE_BYTES);
    printf("\n");
    printf("[x_hat] ");
    vect_print(best.x_hat, VEC_N_SIZE_BYTES);
    printf("\n");

    uint32_t diff_y = count_bit_mismatch_n(y_true, best.y_hat, 17669);
    printf("y_hat mismatch bits = %u\n", diff_y);

    uint32_t diff_x = count_bit_mismatch_n(x_true, best.x_hat, 17669);
    printf("x_hat mismatch bits = %u\n", diff_x);

    free(absL_y);
    free(absL_x);
    free(absL_c);
    return 0;
}
