// Port of alloc/serial.py: Lagrangian bisection with a warm-started multiplier, then a
// greedy fill of the integrality gap. The error budget is NOT a second multiplier -- it is a
// hard per-agent feasibility mask (rows whose divergence rate exceeds the agent's headroom
// are unselectable), which is what makes invariant 3 exact. See STATE.md: two different
// two-stage strawmen reach this solver's exact allocation, so what the joint solve buys is
// allocator time, not utility.
//
// The fill is capped at FillMax steps, which is unity/PORT_SPEC.md's engine decision; the
// residual slack is reported rather than hidden.
// Everything lives in Assembly-CSharp on purpose: no .asmdef, no packages, nothing to install.
// The jobs run on Unity's worker threads as managed code, which carries ~1500 agents at 60 Hz.
// To go faster: install com.unity.burst, then add PARITY_BURST to
// Player Settings > Other Settings > Scripting Define Symbols. Nothing else changes.
using Unity.Collections;
using Unity.Jobs;
// `Allocator` is Unity's allocation-kind enum, so the solver is FidelityAllocator and the
// enum is aliased; naming the class Allocator shadows it and every NativeArray ctor breaks.
using Alloc = Unity.Collections.Allocator;
#if PARITY_BURST
using Unity.Burst;
#endif

namespace Parity
{
    public struct AllocResult
    {
        public float Lambda;
        public float Cost;       // predicted ms
        public float Utility;
        public int Evals;
        public int FillSteps;
        public float Slack;
        public bool Infeasible;  // budget below the error-feasible floor; mask still honoured
        public bool Forced;      // hit EvalMax before the stopping rule
    }

#if PARITY_BURST
    [BurstCompile]
#endif
    struct EvalJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> Salience, Headroom, Q, C, E;
        [ReadOnly] public float Lam;
        [ReadOnly] public int M;
        [WriteOnly] public NativeArray<int> Assign;
        [WriteOnly] public NativeArray<float> ChosenCost, ChosenUtil;

        public void Execute(int i)
        {
            float s = Salience[i], hr = Headroom[i];
            int best = -1;
            float bestScore = 0f, bestCost = 0f, bestUtil = 0f;
            for (int j = 0; j < M; j++)
            {
                if (E[j] > hr) continue;                  // hard error-budget mask
                float u = s * Q[j];
                float score = u - Lam * C[j];
                // strict > keeps the first maximum, and rows are (cost asc, quality desc),
                // so ties break toward the cheaper row exactly as np.argmax does.
                if (best < 0 || score > bestScore)
                {
                    best = j; bestScore = score; bestCost = C[j]; bestUtil = u;
                }
            }
            Assign[i] = best; ChosenCost[i] = bestCost; ChosenUtil[i] = bestUtil;
        }
    }

#if PARITY_BURST
    [BurstCompile]
#endif
    struct FillScanJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> Salience, Headroom, Q, C, E, CurCost, CurUtil;
        [ReadOnly] public float Slack;
        [ReadOnly] public int M;
        [WriteOnly] public NativeArray<float> BestRatio;
        [WriteOnly] public NativeArray<int> BestRow;

        public void Execute(int i)
        {
            float s = Salience[i], hr = Headroom[i], cu = CurUtil[i], cc = CurCost[i];
            float bestR = float.NegativeInfinity;
            int bestJ = -1;
            for (int j = 0; j < M; j++)
            {
                if (E[j] > hr) continue;
                float dc = C[j] - cc;
                if (dc <= 0f || dc > Slack) continue;
                float du = s * Q[j] - cu;
                if (du <= 0f) continue;
                float r = du / dc;
                if (r > bestR) { bestR = r; bestJ = j; }
            }
            BestRatio[i] = bestR; BestRow[i] = bestJ;
        }
    }

    /// <summary>
    /// Allocates one configuration row per agent under a frame-time budget and a per-agent
    /// error headroom. Reused across frames so the multiplier stays warm.
    /// </summary>
    public sealed class FidelityAllocator : System.IDisposable
    {
        public const int EvalMax = 40;      // PORT_SPEC.md: phase 3 measured 22 mean / 28 max cold
        public int FillMax = 8;
        public float Rtol = 1e-4f, Btol = 1e-3f;

        readonly int m;
        NativeArray<float> q, c, e;         // table columns in (cost asc, quality desc) order
        NativeArray<int> rowOf;             // sorted index -> original row index
        NativeArray<float> chosenCost, chosenUtil, fillRatio;
        NativeArray<int> fillRow;
        NativeArray<int> assignSorted;
        int capacity;
        float warmLam = -1f;                // < 0 means cold

        public float LastLambda => warmLam;

        public FidelityAllocator(ParityTable table)
        {
            m = table.M;
            q = new NativeArray<float>(m, Alloc.Persistent);
            c = new NativeArray<float>(m, Alloc.Persistent);
            e = new NativeArray<float>(m, Alloc.Persistent);
            rowOf = new NativeArray<int>(m, Alloc.Persistent);
        }

        void Ensure(int n)
        {
            if (capacity >= n) return;
            Dispose(chosenCost); Dispose(chosenUtil); Dispose(fillRatio);
            if (fillRow.IsCreated) fillRow.Dispose();
            if (assignSorted.IsCreated) assignSorted.Dispose();
            capacity = n;
            chosenCost = new NativeArray<float>(n, Alloc.Persistent);
            chosenUtil = new NativeArray<float>(n, Alloc.Persistent);
            fillRatio = new NativeArray<float>(n, Alloc.Persistent);
            fillRow = new NativeArray<int>(n, Alloc.Persistent);
            assignSorted = new NativeArray<int>(n, Alloc.Persistent);
        }

        static void Dispose(NativeArray<float> a) { if (a.IsCreated) a.Dispose(); }

        /// <summary>Re-sorts the table for this frame's cost vector: (cost asc, quality desc).</summary>
        public void SetCosts(ParityTable table, float[] rowCostMs)
        {
            var idx = new int[m];
            for (int i = 0; i < m; i++) idx[i] = i;
            System.Array.Sort(idx, (a, b) =>
            {
                int k = rowCostMs[a].CompareTo(rowCostMs[b]);
                if (k != 0) return k;
                return table.Quality[b].CompareTo(table.Quality[a]);
            });
            for (int i = 0; i < m; i++)
            {
                rowOf[i] = idx[i];
                q[i] = table.Quality[idx[i]];
                c[i] = rowCostMs[idx[i]];
                e[i] = table.Err[idx[i]];
            }
        }

        struct Eval { public float Cost, Util; }

        Eval RunEval(NativeArray<float> s, NativeArray<float> hr, int n, float lam, NativeArray<int> outAssign)
        {
            new EvalJob
            {
                Salience = s, Headroom = hr, Q = q, C = c, E = e, Lam = lam, M = m,
                Assign = outAssign, ChosenCost = chosenCost, ChosenUtil = chosenUtil,
            }.Schedule(n, 128).Complete();
            double tc = 0.0, tu = 0.0;
            for (int i = 0; i < n; i++) { tc += chosenCost[i]; tu += chosenUtil[i]; }
            return new Eval { Cost = (float)tc, Util = (float)tu };
        }

        /// <param name="assign">[n] out, row index into the original (unsorted) table.</param>
        public AllocResult Solve(NativeArray<float> salience, NativeArray<float> headroom,
                                 int n, float budgetMs, NativeArray<int> assign)
        {
            Ensure(n);
            int evals = 0;

            // lam = 0 first: if the crowd already fits, nothing has to be priced.
            var at0 = RunEval(salience, headroom, n, 0f, assignSorted); evals++;
            float lo = 0f, hi;
            Eval eLo = at0, eHi;

            if (at0.Cost <= budgetMs)
            {
                warmLam = 0f;
                return Finish(salience, headroom, n, budgetMs, assign, at0, 0f, evals, false, false);
            }

            // lam_max: the largest price at which the cheapest row always wins.
            float dcMin = float.MaxValue;
            for (int i = 1; i < m; i++) { float d = c[i] - c[i - 1]; if (d > 0f && d < dcMin) dcMin = d; }
            if (dcMin == float.MaxValue)
            {
                // every row costs the same and the floor is already over budget
                return Finish(salience, headroom, n, budgetMs, assign, at0, 0f, evals, true, false);
            }
            float uMax = 0f;
            for (int i = 0; i < n; i++) { float u = salience[i]; if (u > uMax) uMax = u; }
            float qMax = 0f;
            for (int j = 0; j < m; j++) if (q[j] > qMax) qMax = q[j];
            // Python takes max over the realised U matrix; max(s)*max(q) is an upper bound on
            // that, and any larger lam_max still drives every agent to the cheapest row.
            float lamMax = UnityEngine.Mathf.Max(uMax * qMax / dcMin, 1e-12f);

            eHi = RunEval(salience, headroom, n, lamMax, assignSorted); evals++;
            hi = lamMax;
            if (eHi.Cost > budgetMs)
            {
                warmLam = lamMax;
                return Finish(salience, headroom, n, budgetMs, assign, eHi, lamMax, evals, true, false);
            }

            // warm-started bracket: the multiplier moves slowly frame to frame, so last
            // frame's value is almost always inside [lam/2, lam*2].
            if (warmLam > 0f)
            {
                lo = UnityEngine.Mathf.Min(warmLam * 0.5f, lamMax);
                hi = UnityEngine.Mathf.Min(warmLam * 2f, lamMax);
                eLo = RunEval(salience, headroom, n, lo, assignSorted); evals++;
                eHi = RunEval(salience, headroom, n, hi, assignSorted); evals++;
                while (eHi.Cost > budgetMs && evals < EvalMax)
                {
                    lo = hi; eLo = eHi;
                    hi = UnityEngine.Mathf.Min(hi * 2f, lamMax);
                    eHi = RunEval(salience, headroom, n, hi, assignSorted); evals++;
                }
                while (eLo.Cost <= budgetMs && evals < EvalMax)
                {
                    hi = lo; eHi = eLo;
                    lo = lo > 1e-12f ? lo * 0.5f : 0f;
                    eLo = RunEval(salience, headroom, n, lo, assignSorted); evals++;
                    if (lo == 0f) break;
                }
            }

            // invariant here: cost(lo) > budget >= cost(hi)
            bool forced = true;
            while (evals < EvalMax)
            {
                if (hi - lo <= Rtol * hi || budgetMs - eHi.Cost <= Btol * budgetMs) { forced = false; break; }
                float mid = 0.5f * (lo + hi);
                var em = RunEval(salience, headroom, n, mid, assignSorted); evals++;
                if (em.Cost <= budgetMs) { hi = mid; eHi = em; } else { lo = mid; eLo = em; }
            }

            // re-evaluate at the feasible bracket end so assignSorted holds that assignment
            eHi = RunEval(salience, headroom, n, hi, assignSorted); evals++;
            warmLam = hi;
            return Finish(salience, headroom, n, budgetMs, assign, eHi, hi, evals, false, forced);
        }

        AllocResult Finish(NativeArray<float> s, NativeArray<float> hr, int n, float budget,
                           NativeArray<int> assign, Eval ev, float lam, int evals,
                           bool infeasible, bool forced)
        {
            float slack = budget - ev.Cost;
            float util = ev.Util;
            int steps = 0;
            if (!infeasible && FillMax > 0)
            {
                for (; steps < FillMax; steps++)
                {
                    new FillScanJob
                    {
                        Salience = s, Headroom = hr, Q = q, C = c, E = e,
                        CurCost = chosenCost, CurUtil = chosenUtil, Slack = slack, M = m,
                        BestRatio = fillRatio, BestRow = fillRow,
                    }.Schedule(n, 128).Complete();
                    int bi = -1; float br = float.NegativeInfinity;
                    for (int i = 0; i < n; i++) if (fillRow[i] >= 0 && fillRatio[i] > br) { br = fillRatio[i]; bi = i; }
                    if (bi < 0) break;
                    int bj = fillRow[bi];
                    float du = s[bi] * q[bj] - chosenUtil[bi];
                    float dc = c[bj] - chosenCost[bi];
                    assignSorted[bi] = bj;
                    chosenUtil[bi] += du;
                    chosenCost[bi] += dc;
                    slack -= dc; util += du;
                }
            }
            for (int i = 0; i < n; i++) assign[i] = rowOf[assignSorted[i]];
            return new AllocResult
            {
                Lambda = lam, Cost = budget - slack, Utility = util, Evals = evals,
                FillSteps = steps, Slack = slack, Infeasible = infeasible, Forced = forced,
            };
        }

        public void Dispose()
        {
            if (q.IsCreated) q.Dispose();
            if (c.IsCreated) c.Dispose();
            if (e.IsCreated) e.Dispose();
            if (rowOf.IsCreated) rowOf.Dispose();
            Dispose(chosenCost); Dispose(chosenUtil); Dispose(fillRatio);
            if (fillRow.IsCreated) fillRow.Dispose();
            if (assignSorted.IsCreated) assignSorted.Dispose();
        }
    }

}
