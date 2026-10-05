// sim/scenes.py Palace in the engine: the Mysore Palace forecourt on Jamboo Savari day. A seated
// crowd stands up at once when the procession leaves; each row empties to its nearer end, and
// each block side follows one route to its gate (PalaceSite). Seat blocks, gate widths, the
// speed spread and the elderly share are ASSUMPTIONS (docs/site/mysore_palace.md, section 6);
// the mean walking speed is LIT. Agents that reach a gate's sink stand outside the walls near
// that gate; when everyone is out the crowd sits down again and the cycle repeats.
//
// The core: every route leg is a straight, wall-free segment (string-pulled on clear cells),
// and on passing a waypoint the core is re-seated from that waypoint rather than from wherever
// the crowd pressed the agent, so a surrogate riding the core never cuts a building corner --
// the same rule as CorridorScene's doorway.
using System.Collections.Generic;
using UnityEngine;

namespace Parity
{
    public enum PalacePlan { Nearest, Balanced, Measured }

    public sealed class PalaceScene : CrowdScene
    {
        public const int Rows = 20, Cols = 40, PerBlock = Rows * Cols;
        public const float SeatPitch = 0.5f, RowPitch = 0.9f, AisleW = 1.2f, Pass = 2.5f;
        public const float SpeedMean = 1.06f, SpeedSd = 0.15f, ElderlyShare = 0.10f, ElderlyScale = 0.75f;
        public const float StallSeconds = 5f, StallMetres = 0.5f;
        public static float ReleaseSeconds = 8f;   // seated, then the procession leaves
        public static float ResetSeconds = 15f;    // after the last one is out
        public static PalacePlan Plan = PalacePlan.Balanced;
        /// <summary>Sustained persons per minute per gate, measured at 48,000 seats by
        /// bench/palace.py; the Measured plan shares seats by these. Empty until the run with the
        /// audited gates (docs/site/mysore_palace.md, 5b) has measured them -- the earlier rates
        /// belonged to a gate set the audit corrected.</summary>
        public static readonly Dictionary<string, float> MeasuredRates = new Dictionary<string, float>();

        public PalaceSite Site { get; private set; }
        Vector2[] seat, rowEnd;
        int[] side, first, k, gate;
        bool[] done, follow;
        Vector2[] mark;
        Vector2[][] routes;
        int[] sideGate;
        int releaseFrame, clearFrame = -1, nb;
        // parking outside the walls, nearest each gate first
        Vector2[] slots;
        int[][] slotOrder;
        int[] slotNext;
        bool[] slotTaken;

        public int Exits, Cycles, Out;
        public int[] ExitsByGate;
        public float LastClearSeconds = -1f;

        public PalaceScene(SceneSpec s) : base(s) { }

        public override void Begin(CrowdWorld w)
        {
            Site = PalaceSite.Load();
            if ((Site.Size - Spec.Size).magnitude > 0.05f)
                Debug.LogWarning($"palace: SceneSpec size {Spec.Size} differs from the site file's {Site.Size}");
            int n = w.N;
            nb = (n + PerBlock - 1) / PerBlock;
            if (nb > Site.Blocks.Count) throw new System.ArgumentException($"palace: {n} seats need {nb} blocks, {Site.Blocks.Count} fit");
            seat = new Vector2[n]; rowEnd = new Vector2[n]; mark = new Vector2[n];
            side = new int[n]; first = new int[n]; k = new int[n]; gate = new int[n];
            done = new bool[n]; follow = new bool[n];
            ExitsByGate = new int[Site.Gates.Length];
            float bw = Rows * RowPitch, bh = Cols * SeatPitch;
            var starts = new Vector2[2 * nb];
            for (int b = 0; b < nb; b++)
            {
                starts[b] = Site.Blocks[b] + new Vector2(bw / 2, -AisleW / 2);
                starts[b + nb] = Site.Blocks[b] + new Vector2(bw / 2, bh + AisleW / 2);
            }
            sideGate = Assign(starts);
            routes = new Vector2[2 * nb][];
            for (int s = 0; s < starts.Length; s++)
            {
                routes[s] = Site.Route(starts[s], sideGate[s]);
                if (routes[s] == null) throw new System.InvalidOperationException($"palace: block side {s} cannot reach its gate");
            }
            // the last block is filled at random seats, as sim/scenes.py does
            var order = new int[PerBlock];
            for (int q = 0; q < PerBlock; q++) order[q] = q;
            var pr = new Rng(777u);
            for (int q = PerBlock - 1; q > 0; q--) { int j = pr.Range(0, q + 1); (order[q], order[j]) = (order[j], order[q]); }
            for (int i = 0; i < n; i++)
            {
                int b = i / PerBlock, s = b == nb - 1 && n % PerBlock != 0 ? order[i % PerBlock] : i % PerBlock;
                int r = s / Cols, c = s % Cols;
                var corner = Site.Blocks[b];
                seat[i] = corner + new Vector2((r + 0.5f) * RowPitch, (c + 0.5f) * SeatPitch);
                bool north = c >= Cols / 2;
                side[i] = b + (north ? nb : 0);
                rowEnd[i] = new Vector2(seat[i].x, north ? corner.y + bh + AisleW / 2 : corner.y - AisleW / 2);
                var rt = routes[side[i]];
                first[i] = rt.Length > 1 && Site.Visible(rowEnd[i].x, rowEnd[i].y, rt[1].x, rt[1].y) ? 1 : 0;
                gate[i] = sideGate[side[i]];
            }
            BuildSlots();
            releaseFrame = Mathf.RoundToInt(ReleaseSeconds / CrowdWorld.Dt);
        }

        int[] Assign(Vector2[] starts)
        {
            var open = Site.Open;
            var outG = new int[starts.Length];
            var d = new double[starts.Length, open.Length];
            var best = new double[starts.Length];
            for (int s = 0; s < starts.Length; s++)
            {
                best[s] = double.PositiveInfinity;
                for (int j = 0; j < open.Length; j++)
                {
                    d[s, j] = Site.Distance(starts[s], open[j]);
                    if (d[s, j] < best[s]) { best[s] = d[s, j]; outG[s] = open[j]; }
                }
            }
            if (Plan == PalacePlan.Nearest) return outG;
            var byBest = new int[starts.Length];
            for (int s = 0; s < byBest.Length; s++) byBest[s] = s;
            System.Array.Sort((double[])best.Clone(), byBest);   // stable enough: distinct distances
            var load = new double[open.Length];
            if (Plan == PalacePlan.Balanced)
            {
                // sim/scenes.py Palace._balance: room per gate in proportion to its width
                double tw = 0;
                foreach (int g in open) tw += Site.Gates[g].Width;
                foreach (int s in byBest)
                {
                    int pick = -1;
                    var idx = new int[open.Length];
                    var row = new double[open.Length];
                    for (int j = 0; j < open.Length; j++) { idx[j] = j; row[j] = d[s, j]; }
                    System.Array.Sort(row, idx);
                    foreach (int j in idx)
                    {
                        double room = System.Math.Ceiling(Site.Gates[open[j]].Width / tw * starts.Length);
                        if (load[j] + 1 <= room && !double.IsInfinity(d[s, j])) { pick = j; break; }
                    }
                    if (pick < 0) pick = idx[idx.Length - 1];
                    load[pick] += 1;
                    outG[s] = open[pick];
                }
                return outG;
            }
            // Measured: sim/scenes.py Palace._by_rate (nearest gate until rates are measured)
            if (MeasuredRates.Count == 0) return outG;
            foreach (int s in byBest)
            {
                int pick = -1;
                double bf = double.PositiveInfinity;
                for (int j = 0; j < open.Length; j++)
                {
                    if (!MeasuredRates.TryGetValue(Site.Gates[open[j]].Key, out float rate) || rate <= 0f) continue;
                    double fin = (load[j] + PerBlock / 2.0) / rate + d[s, j] / SpeedMean / 60.0;
                    if (fin < bf) { bf = fin; pick = j; }
                }
                load[pick] += PerBlock / 2.0;
                outG[s] = open[pick];
            }
            return outG;
        }

        void BuildSlots()
        {
            const float pitch = 1.6f;
            var list = new List<Vector2>();
            for (float x = 1f; x < Spec.Size.x - 1f; x += pitch)
                for (float y = 1f; y < Spec.Size.y - 1f; y += pitch)
                {
                    var p = new Vector2(x, y);
                    if (!PalaceSite.Inside(Site.Compound, p) && !Site.Building[Site.CellI(x) * Site.Ny + Site.CellJ(y)]) list.Add(p);
                }
            slots = list.ToArray();
            slotTaken = new bool[slots.Length];
            slotOrder = new int[Site.Gates.Length][];
            slotNext = new int[Site.Gates.Length];
            foreach (int g in Site.Open)
            {
                var c = Site.Gates[g].At + Site.Gates[g].Out * 8f;
                var key = new float[slots.Length];
                var idx = new int[slots.Length];
                for (int q = 0; q < slots.Length; q++) { key[q] = (slots[q] - c).sqrMagnitude; idx[q] = q; }
                System.Array.Sort(key, idx);
                slotOrder[g] = idx;
            }
        }

        Vector2 Park(int g)
        {
            var o = slotOrder[g];
            while (slotNext[g] < o.Length && slotTaken[o[slotNext[g]]]) slotNext[g]++;
            if (slotNext[g] >= o.Length) return Vector2.one;   // out of slots: a corner, never reached in practice
            int q = o[slotNext[g]];
            slotTaken[q] = true;
            return slots[q];
        }

        public override void SpawnOne(CrowdWorld w, int i, ref Rng r)
        {
            w.Pos[i] = seat[i];
            w.SetGoal(i, seat[i]);
            float u1 = 1f - r.NextFloat(), u2 = r.NextFloat();
            float z = Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
            float sp = Mathf.Clamp(SpeedMean + SpeedSd * z, 0.6f, 1.5f);
            if (r.NextFloat() < ElderlyShare) sp *= ElderlyScale;
            w.Speed[i] = sp;
            k[i] = 0;
        }

        Vector2 Waypoint(int i, int kk)
        {
            if (kk == 0) return seat[i];
            if (kk == 1) return rowEnd[i];
            var rt = routes[side[i]];
            return rt[Mathf.Min(first[i] + kk - 2, rt.Length - 1)];
        }

        int Count(int i) => 2 + routes[side[i]].Length - first[i];

        public override void Constrain(CrowdWorld w)
        {
            for (int i = 0; i < w.N; i++)
            {
                if (k[i] == 0 && !done[i]) { w.Pos[i] = seat[i]; w.Vel[i] = Vector2.zero; }
                else if (done[i]) { w.Pos[i] = w.Goal[i]; w.Vel[i] = Vector2.zero; }
                else w.Pos[i] = Site.Constrain(w.Pos[i], w.PrevPos[i]);
            }
        }

        public override void AfterStep(CrowdWorld w, int frame, ref Rng r)
        {
            int every = Mathf.RoundToInt(StallSeconds / CrowdWorld.Dt);
            bool check = frame % every == 0;
            int outNow = 0;
            for (int i = 0; i < w.N; i++)
            {
                if (done[i]) { outNow++; continue; }
                if (k[i] == 0)
                {
                    if (frame < releaseFrame) continue;
                    k[i] = 1;
                    mark[i] = w.Pos[i];
                    w.SetGoal(i, rowEnd[i]);
                    continue;
                }
                if (Site.InSink(w.Pos[i]))
                {
                    done[i] = true;
                    outNow++;
                    Exits++;
                    ExitsByGate[gate[i]]++;
                    var p = Park(gate[i]);
                    w.Pos[i] = p;
                    w.SetGoal(i, p);
                    continue;
                }
                if (check && frame >= releaseFrame + every)
                {
                    if ((w.Pos[i] - mark[i]).magnitude < StallMetres) follow[i] = true;
                    mark[i] = w.Pos[i];
                }
                if (follow[i])
                {
                    w.SetGoal(i, w.Pos[i] + 3f * Site.Steer(w.Pos[i], gate[i]));
                    continue;
                }
                bool last = k[i] == Count(i) - 1;
                if (!last && (w.Goal[i] - w.Pos[i]).magnitude < Pass)
                {
                    var from = Waypoint(i, k[i]);
                    k[i]++;
                    var to = Waypoint(i, k[i]);
                    w.SetGoal(i, to);
                    // re-seat the core on the waypoint just passed: the leg is wall-free
                    w.CoreFrom[i] = from; w.CoreP[i] = from;
                    w.CoreLen[i] = Mathf.Max((to - from).magnitude, 0.001f);
                }
                else if (CoreDone(w, i) && !last) w.SetGoal(i, w.Goal[i]);
            }
            Out = outNow;
            if (outNow == w.N)
            {
                if (clearFrame < 0)
                {
                    clearFrame = frame;
                    LastClearSeconds = (frame - releaseFrame) * CrowdWorld.Dt;
                    Cycles++;
                }
                else if (frame - clearFrame >= Mathf.RoundToInt(ResetSeconds / CrowdWorld.Dt)) Reseat(w, frame);
            }
        }

        void Reseat(CrowdWorld w, int frame)
        {
            for (int i = 0; i < w.N; i++)
            {
                done[i] = follow[i] = false;
                k[i] = 0;
                w.Pos[i] = seat[i];
                w.Vel[i] = Vector2.zero;
                w.SetGoal(i, seat[i]);
            }
            System.Array.Clear(slotTaken, 0, slotTaken.Length);
            System.Array.Clear(slotNext, 0, slotNext.Length);
            clearFrame = -1;
            releaseFrame = frame + Mathf.RoundToInt(ReleaseSeconds / CrowdWorld.Dt);
        }

        public int GateOf(int i) => gate[i];
        public bool IsOut(int i) => done[i];

        /// <summary>Walking agents standing off walkable ground: must stay 0.</summary>
        public int WallViolations(CrowdWorld w)
        {
            int n = 0;
            for (int i = 0; i < w.N; i++)
                if (k[i] > 0 && !done[i] && !Site.Walkable(w.Pos[i])) n++;
            return n;
        }

        public int SeatedCount()
        {
            int n = 0;
            for (int i = 0; i < k.Length; i++) if (k[i] == 0 && !done[i]) n++;
            return n;
        }
    }
}
