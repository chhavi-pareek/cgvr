/* OpenACC port of the PARITY Lagrangian allocator: same kernel as alloc/cuda.
 *
 * Host drives the bracketing/bisection loop; each evaluation is one
 * parallel-loop region with a sum reduction, so lambda round-trips to the host
 * by construction of the OpenACC model (contrast the CUDA version, where the
 * last block advances the state machine on the device). Compile with FMA
 * contraction off (-Mnofma for nvc, -ffp-contract=off for clang/gcc) so
 * score = (s*q) - (lam*c) keeps the oracle's three roundings.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENACC
#include <openacc.h>
#endif

int parity_acc_backend(void)
{
#ifdef _OPENACC
    return 1;
#else
    return 0;
#endif
}

static double eval_lambda(int n, int m, const double *s, const double *q,
                          const double *c, const double *e, const double *h,
                          double lam, int *abuf, int buf)
{
    double total = 0.0;
    int off = buf * n;
#pragma acc parallel loop present(s[0:n], q[0:m], c[0:m], e[0:m], h[0:n], abuf[0:3 * n]) reduction(+ : total)
    for (int i = 0; i < n; i++) {
        double si = s[i], hi = h[i];
        double best = -INFINITY, bc = INFINITY, bq = -INFINITY;
        int bj = -1;
        for (int j = 0; j < m; j++) {
            if (e[j] <= hi) {
                double cj = c[j], qj = q[j];
                double u = si * qj;
                double lc = lam * cj;
                double sc = u - lc;
                if (sc > best || (sc == best && (cj < bc || (cj == bc && qj > bq)))) {
                    best = sc;
                    bj = j;
                    bc = cj;
                    bq = qj;
                }
            }
        }
        abuf[off + i] = bj;
        total += bc;
    }
    return total;
}

/* One greedy fill step. Returns the cost delta applied, or -1 if none fits. */
static double fill_step(int n, int m, const double *s, const double *q,
                        const double *c, const double *e, const double *h,
                        int *abuf, int buf, double slack, double *br, int *bj)
{
    double bestr = -INFINITY;
    int off = buf * n;
#pragma acc parallel loop present(s[0:n], q[0:m], c[0:m], e[0:m], h[0:n], abuf[0:3 * n], br[0:n], bj[0:n]) reduction(max : bestr)
    for (int i = 0; i < n; i++) {
        double si = s[i], hi = h[i];
        int ai = abuf[off + i];
        double cu = si * q[ai];
        double cc = c[ai];
        double best = -INFINITY, bc = INFINITY, bq = -INFINITY;
        int bidx = -1;
        for (int j = 0; j < m; j++) {
            if (e[j] <= hi) {
                double cj = c[j], qj = q[j];
                double u = si * qj;
                double du = u - cu;
                double dc = cj - cc;
                if (du > 0.0 && dc > 0.0 && dc <= slack) {
                    double r = du / dc;
                    if (r > best || (r == best && (cj < bc || (cj == bc && qj > bq)))) {
                        best = r;
                        bidx = j;
                        bc = cj;
                        bq = qj;
                    }
                }
            }
        }
        br[i] = best;
        bj[i] = bidx;
        if (best > bestr) bestr = best;
    }
    if (bestr == -INFINITY) return -1.0;
    int besti = n;
#pragma acc parallel loop present(br[0:n]) reduction(min : besti)
    for (int i = 0; i < n; i++)
        if (br[i] == bestr && i < besti) besti = i;
#pragma acc update self(bj[besti:1], abuf[off + besti:1])
    int j = bj[besti];
    double d = c[j] - c[abuf[off + besti]];
    abuf[off + besti] = j;
#pragma acc update device(abuf[off + besti:1])
    return d;
}

#define SWAP(x, y) do { int _t = (x); (x) = (y); (y) = _t; } while (0)

/* stats: [0]=lam [1]=cost [2]=evals [3]=infeasible [4]=fill_steps */
int parity_alloc_acc(int n, int m, const double *s, const double *q,
                     const double *c, const double *e, const double *h,
                     double budget, double lam_warm, double lam_max,
                     double rtol, double btol, int max_iter, int do_fill,
                     int *a_out, double *stats)
{
    int *abuf = (int *)malloc(sizeof(int) * 3 * (size_t)n);
    double *br = (double *)malloc(sizeof(double) * (size_t)n);
    int *bj = (int *)malloc(sizeof(int) * (size_t)n);
    if (!abuf || !br || !bj) return -1;
    int evals = 0, infeasible = 0, res = 0, steps = 0;
    double lam = 0.0, t = 0.0;

#pragma acc data copyin(s[0:n], q[0:m], c[0:m], e[0:m], h[0:n]) create(abuf[0:3 * n], br[0:n], bj[0:n])
    {
        evals = 1;
        double t0 = eval_lambda(n, m, s, q, c, e, h, 0.0, abuf, 0);
        if (t0 <= budget) {
            res = 0; lam = 0.0; t = t0;
        } else if (lam_max < 0.0) {
            res = 0; lam = 0.0; t = t0; infeasible = 1;
        } else {
            evals++;
            int a_hi = 1, a_lo = 0;
            double t_hi = eval_lambda(n, m, s, q, c, e, h, lam_max, abuf, a_hi);
            if (t_hi > budget) {
                res = a_hi; lam = lam_max; t = t_hi; infeasible = 1;
            } else {
                double lo, hi, t_lo;
                if (lam_warm <= 0.0) {
                    lo = 0.0; hi = lam_max; t_lo = t0;
                } else {
                    lo = fmin(lam_warm / 2, lam_max);
                    hi = fmin(lam_warm * 2, lam_max);
                    t_lo = eval_lambda(n, m, s, q, c, e, h, lo, abuf, a_lo);
                    t_hi = eval_lambda(n, m, s, q, c, e, h, hi, abuf, a_hi);
                    evals += 2;
                    while (t_hi > budget) {
                        lo = hi; t_lo = t_hi; SWAP(a_lo, a_hi);
                        hi = fmin(hi * 2, lam_max);
                        t_hi = eval_lambda(n, m, s, q, c, e, h, hi, abuf, a_hi);
                        evals++;
                    }
                    while (t_lo <= budget) {
                        hi = lo; t_hi = t_lo; SWAP(a_lo, a_hi);
                        lo = lo > 1e-12 ? lo / 2 : 0.0;
                        t_lo = eval_lambda(n, m, s, q, c, e, h, lo, abuf, a_lo);
                        evals++;
                        if (lo == 0.0) break;
                    }
                }
                int free_b = 3 - a_lo - a_hi;
                for (int it = 0; it < max_iter; it++) {
                    if (hi - lo <= rtol * hi || budget - t_hi <= btol * budget) break;
                    double mid = 0.5 * (lo + hi);
                    double tm = eval_lambda(n, m, s, q, c, e, h, mid, abuf, free_b);
                    evals++;
                    if (tm <= budget) { hi = mid; t_hi = tm; SWAP(a_hi, free_b); }
                    else { lo = mid; t_lo = tm; SWAP(a_lo, free_b); }
                }
                res = a_hi; lam = hi; t = t_hi;
            }
        }
        if (do_fill && !infeasible) {
            double slack = budget - t;
            for (int k = 0; k < n; k++) {
                double d = fill_step(n, m, s, q, c, e, h, abuf, res, slack, br, bj);
                if (d < 0.0) break;
                slack -= d;
                steps++;
            }
        }
#pragma acc update self(abuf[res * n:n])
    }
    memcpy(a_out, abuf + (size_t)res * n, sizeof(int) * (size_t)n);
    stats[0] = lam; stats[1] = t; stats[2] = evals; stats[3] = infeasible; stats[4] = steps;
    free(abuf); free(br); free(bj);
    return 0;
}
