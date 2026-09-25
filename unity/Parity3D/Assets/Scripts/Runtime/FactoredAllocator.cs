// Port of alloc/factored.py: the two-salience objective
//
//     sum_i  a_i * q_state(behaviour, navigation)  +  b_i * q_view(animation, geometry)
//
// solved over the table's factorisation into 12 state pairs x 15 view pairs. For a multiplier
// lam each agent's best row is its best state pair plus its best view pair, each a shared menu
// at shared prices, so each half is a hull walked by a sort (alloc/hull.py) and one bisection
// on lam prices both. Per evaluation O(K log n) for K hull vertices, no scan over the table.
//
// Needs the FULL table: pruning on combined quality drops rows a two-salience optimum uses
// (tests/test_factored.py shows one). Each hull prunes its own half, which stays valid.
// Arithmetic is fp64 throughout to track the Python oracle; the result reports fp32.
using System;
using System.Collections.Generic;

namespace Parity
{
    public sealed class FactoredAllocator
    {
        public int MaxIter = 64;
        public double Rtol = 1e-4;

        readonly int nS, nV;
        readonly int[] rowOf;            // [s * nV + v]
        readonly double[] qS, qV, eS;
        readonly double[] cS, cV;
        double[] levels;                 // distinct state error rates, ascending

        // per-solve work, grown on demand
        Agent[] sortS = new Agent[0], sortV = new Agent[0];
        double[] salS = new double[0], salV = new double[0];
        int[] pickS = new int[0], pickV = new int[0];

        struct Agent { public double Key; public int Index; }
        sealed class DescOrder : IComparer<Agent>
        {
            public int Compare(Agent x, Agent y)
            {
                int c = y.Key.CompareTo(x.Key);                  // salience descending
                return c != 0 ? c : x.Index.CompareTo(y.Index);  // stable, as numpy's argsort
            }
        }
        static readonly DescOrder Desc = new DescOrder();

        // reused every frame so the solve allocates nothing once warm
        List<int>[] byLevel;
        readonly List<int> menuS = new List<int>(), menuV = new List<int>(), hullWork = new List<int>();

        sealed class Part
        {
            public int Lo, Hi;           // slice of the sorted agent array
            public int[] Hull;           // menu indices, increasing cost
            public double[] Theta;       // switch points, decreasing
        }
        readonly List<Part> partsS = new List<Part>(), partsV = new List<Part>();

        FactoredAllocator(int s, int v)
        {
            nS = s; nV = v;
            rowOf = new int[s * v];
            qS = new double[s]; qV = new double[v]; eS = new double[s];
            cS = new double[s]; cV = new double[v];
        }

        /// <summary>Null if the table is not the full state x view product (a pruned table).</summary>
        public static FactoredAllocator TryCreate(ParityTable t)
        {
            var sKeys = new SortedSet<int>();
            var vKeys = new SortedSet<int>();
            for (int r = 0; r < t.M; r++)
            {
                sKeys.Add(t.TierOf(r, 0) * ParityTable.NTiers + t.TierOf(r, 1));
                vKeys.Add(t.TierOf(r, 2) * ParityTable.NTiers + t.TierOf(r, 3));
            }
            if (sKeys.Count * vKeys.Count != t.M) return null;
            var f = new FactoredAllocator(sKeys.Count, vKeys.Count);
            var sIdx = new Dictionary<int, int>();
            var vIdx = new Dictionary<int, int>();
            foreach (int k in sKeys) sIdx[k] = sIdx.Count;
            foreach (int k in vKeys) vIdx[k] = vIdx.Count;
            for (int i = 0; i < f.rowOf.Length; i++) f.rowOf[i] = -1;
            for (int r = 0; r < t.M; r++)
            {
                int b = t.TierOf(r, 0), n = t.TierOf(r, 1), a = t.TierOf(r, 2), g = t.TierOf(r, 3);
                int s = sIdx[b * ParityTable.NTiers + n], v = vIdx[a * ParityTable.NTiers + g];
                f.rowOf[s * f.nV + v] = r;
                f.qS[s] = (t.AxisQ[0, b] + t.AxisQ[1, n]) / ParityTable.NAxes;
                f.qV[v] = (t.AxisQ[2, a] + t.AxisQ[3, g]) / ParityTable.NAxes;
            }
            for (int i = 0; i < f.rowOf.Length; i++) if (f.rowOf[i] < 0) return null;
            var lv = new SortedSet<double>();
            for (int s = 0; s < f.nS; s++)
            {
                f.eS[s] = t.Err[f.rowOf[s * f.nV]];
                for (int v = 0; v < f.nV; v++)
                    if (Math.Abs(t.Err[f.rowOf[s * f.nV + v]] - f.eS[s]) > 1e-9)
                        throw new InvalidOperationException("error rate depends on the view half");
                lv.Add(f.eS[s]);
            }
            f.levels = new double[lv.Count];
            lv.CopyTo(f.levels);
            return f;
        }

        public int RowOf(int s, int v) => rowOf[s * nV + v];

        /// <summary>Split an additive row cost into state (carrying the core) and view halves.</summary>
        public void SetCosts(float[] rowCost)
        {
            for (int s = 0; s < nS; s++) cS[s] = rowCost[rowOf[s * nV]];
            for (int v = 0; v < nV; v++) cV[v] = rowCost[rowOf[v]] - rowCost[rowOf[0]];
            double scale = 0.0;
            for (int r = 0; r < rowCost.Length; r++) scale = Math.Max(scale, Math.Abs(rowCost[r]));
            double tol = 1e-5 * scale + 1e-9;
            for (int s = 0; s < nS; s++)
                for (int v = 0; v < nV; v++)
                    if (Math.Abs(cS[s] + cV[v] - rowCost[rowOf[s * nV + v]]) > tol)
                        throw new InvalidOperationException("row cost is not additive over state and view");
        }

        int[] UpperHull(double[] c, double[] q, List<int> menu)
        {
            menu.Sort((x, y) => { int k = c[x].CompareTo(c[y]); return k != 0 ? k : q[y].CompareTo(q[x]); });
            var h = hullWork;
            h.Clear();
            foreach (int j in menu)
            {
                if (h.Count > 0 && q[j] <= q[h[h.Count - 1]]) continue;
                while (h.Count >= 2)
                {
                    int a = h[h.Count - 2], b = h[h.Count - 1];
                    double lhs = (q[b] - q[a]) * (c[j] - c[a]);
                    double rhs = (q[j] - q[a]) * (c[b] - c[a]);
                    if (lhs <= rhs) h.RemoveAt(h.Count - 1); else break;
                }
                h.Add(j);
            }
            return h.ToArray();
        }

        static double[] Thetas(double[] c, double[] q, int[] h)
        {
            var t = new double[Math.Max(h.Length - 1, 0)];
            for (int k = 0; k < t.Length; k++) t[k] = (q[h[k + 1]] - q[h[k]]) / (c[h[k + 1]] - c[h[k]]);
            return t;
        }

        /// <summary>Number of agents in sorted[lo..hi) with salience >= s (salience descending).</summary>
        static int CountAtLeast(double[] sal, int lo, int hi, double s)
        {
            int a = lo, b = hi;
            while (a < b)
            {
                int m = (a + b) >> 1;
                if (sal[m] >= s) a = m + 1; else b = m;
            }
            return a - lo;
        }

        /// <summary>Walks the blocks of one part at lam: agents [bounds[k], bounds[k+1]) take
        /// hull vertex K-1-k. Returns the part's total cost; writes picks when pick != null.</summary>
        static double Blocks(Part p, double lam, double[] sal, double[] c, Agent[] sorted, int[] pick)
        {
            int n = p.Hi - p.Lo, K = p.Hull.Length;
            if (n == 0) return 0.0;
            double total = 0.0;
            int prev = 0;
            // most salient first: vertex K-1 goes to agents with s >= every threshold
            for (int blk = 0; blk < K; blk++)
            {
                int vtx = K - 1 - blk;
                int end;
                if (vtx == 0) end = n;
                else
                {
                    double S = lam / Math.Max(p.Theta[vtx - 1], 1e-300);
                    end = CountAtLeast(sal, p.Lo, p.Hi, S);
                }
                end = Math.Min(Math.Max(end, prev), n);
                int size = end - prev;
                if (size > 0)
                {
                    total += size * c[p.Hull[vtx]];
                    if (pick != null)
                        for (int k = prev; k < end; k++) pick[sorted[p.Lo + k].Index] = p.Hull[vtx];
                }
                prev = end;
            }
            return total;
        }

        void Prepare(Agent[] sorted, double[] sal, float[] salience, List<int> idx, int count, int start,
                     double[] c, double[] q, List<int> menu, List<Part> parts)
        {
            for (int k = 0; k < count; k++)
            {
                int i = idx == null ? k : idx[k];
                sorted[start + k] = new Agent { Key = salience[i], Index = i };
            }
            Array.Sort(sorted, start, count, Desc);
            for (int k = 0; k < count; k++) sal[start + k] = sorted[start + k].Key;
            var h = UpperHull(c, q, menu);
            parts.Add(new Part { Lo = start, Hi = start + count, Hull = h, Theta = Thetas(c, q, h) });
        }

        double Total(double lam)
        {
            double t = 0.0;
            foreach (var p in partsS) t += Blocks(p, lam, salS, cS, sortS, null);
            foreach (var p in partsV) t += Blocks(p, lam, salV, cV, sortV, null);
            return t;
        }

        /// <summary>stateSal / viewSal / headroom are per agent; assign receives full-table rows.
        /// A negative view salience array entry is treated as 0.</summary>
        public AllocResult Solve(float[] stateSal, float[] viewSal, float[] headroom, int n, float budget,
                                 int[] assign)
        {
            if (sortS.Length < n)
            {
                sortS = new Agent[n]; sortV = new Agent[n];
                salS = new double[n]; salV = new double[n];
                pickS = new int[n]; pickV = new int[n];
            }
            partsS.Clear(); partsV.Clear();
            var res = new AllocResult();

            // group agents by how many error levels their headroom affords
            int L = levels.Length;
            if (byLevel == null)
            {
                byLevel = new List<int>[L + 1];
                for (int g = 0; g <= L; g++) byLevel[g] = new List<int>();
            }
            for (int g = 0; g <= L; g++) byLevel[g].Clear();
            for (int i = 0; i < n; i++)
            {
                double h = headroom[i];
                int g = 0;
                while (g < L && levels[g] <= h) g++;   // numpy searchsorted(side="right")
                if (g == 0) { res.Starved++; g = 1; }   // cannot happen while a rate-0 pair exists
                byLevel[g].Add(i);
            }
            int start = 0;
            for (int g = 1; g <= L; g++)
            {
                if (byLevel[g].Count == 0) continue;
                menuS.Clear();
                for (int s = 0; s < nS; s++) if (eS[s] <= levels[g - 1] + 1e-12) menuS.Add(s);
                var idx = byLevel[g];
                Prepare(sortS, salS, stateSal, idx, idx.Count, start, cS, qS, menuS, partsS);
                start += idx.Count;
            }
            menuV.Clear();
            for (int v = 0; v < nV; v++) menuV.Add(v);
            Prepare(sortV, salV, viewSal, null, n, 0, cV, qV, menuV, partsV);
            for (int i = 0; i < n; i++) if (salV[i] < 0.0) salV[i] = 0.0;

            int evals = 1;
            double lam;
            if (Total(0.0) <= budget) lam = 0.0;
            else
            {
                double hi = 1.0;
                evals++;
                while (Total(hi) > budget && hi < 1e18) { hi *= 4.0; evals++; }
                double lo = 0.0;
                for (int it = 0; it < MaxIter; it++)
                {
                    if (hi - lo <= Rtol * Math.Max(hi, 1e-12)) break;
                    double mid = 0.5 * (lo + hi);
                    evals++;
                    if (Total(mid) <= budget) hi = mid; else lo = mid;
                }
                lam = hi;
            }

            foreach (var p in partsS) Blocks(p, lam, salS, cS, sortS, pickS);
            foreach (var p in partsV) Blocks(p, lam, salV, cV, sortV, pickV);
            double cost = 0.0, util = 0.0;
            for (int i = 0; i < n; i++)
            {
                int s = pickS[i], v = pickV[i];
                assign[i] = rowOf[s * nV + v];
                cost += cS[s] + cV[v];
                util += stateSal[i] * qS[s] + Math.Max(viewSal[i], 0f) * qV[v];
            }
            res.Lambda = (float)lam;
            res.Cost = (float)cost;
            res.Utility = (float)util;
            res.Evals = evals;
            res.Slack = (float)(budget - cost);
            res.Infeasible = cost > budget * (1.0 + 1e-9);
            return res;
        }
    }
}
