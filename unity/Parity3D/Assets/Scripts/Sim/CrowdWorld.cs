// One crowd, stepped under one fidelity policy. Two of these run side by side from the same
// seed and the same camera path: the MassLOD baseline and PARITY.
//
// The four table axes are wired to things you can actually see:
//   behaviour  0-2 live steering at 1 / 3 / 10 frame stride, 3 = surrogate along the core
//   navigation 0-2 full / sparse / no neighbour separation,  3 = core only
//   animation  0-2 gait phase advanced at 1 / 3 / 10 frame stride, 3 = frozen
//   geometry   0-3 mesh detail, handed to the renderer
//
// The view-invariant core (invariant 4) is the straight route to the goal plus progress s.
// It is stepped for every agent every frame at every tier and is never allocated -- it is
// what a restored agent is snapped back onto, and what keeps a surrogate agent in the right
// place even while its behaviour has diverged.
//
// The behaviour model here is a 3-D stand-in for sim/behaviour.py's latent jump process, not
// a port of it: the allocator, the ledger and the MassLOD baseline are faithful ports, the
// crowd itself is an analogue. Divergence numbers from this scene are the scene's own.
using UnityEngine;

namespace Parity
{
    public enum Policy { Baseline, Parity }

    public sealed class CrowdWorld : System.IDisposable
    {
        public const float Dt = 1f / 60f;
        public readonly Policy Mode;
        public int N;

        public Vector2[] Pos, Goal, CoreP;   // CoreP: the core's authoritative position
        public float[] Speed, Sig, Phase, Dmeas, CoreS, CoreLen;
        public Vector2[] CoreFrom;
        public bool[] InView;
        public sbyte[] SimTier, VisTier;
        public int[] Row;                    // parity: table row; baseline: -1
        public sbyte[] Beh, Nav, Anim, Geo;

        public readonly ParityTable Table;
        public readonly MassLodAssigner MassLod = new MassLodAssigner();
        public FidelityAllocator Alloc;
        public ErrorLedger Ledger;
        public readonly RlsCostModel Cost = new RlsCostModel();

        public float BudgetMs = 16.7f;
        public float Cap = 4.0f;
        public AllocResult Last;
        public float AllocMs, StepMs;
        public int Restorations, Promotes, Demotes;
        /// <summary>Agents whose ledger passed the cap. The feasibility mask makes this
        /// impossible by construction, so it is displayed as a live assertion.</summary>
        public int CapBreaches;

        readonly float[] rowCost;
        readonly double[,] thetaAxis = new double[ParityTable.NAxes, ParityTable.NTiers];
        readonly int[,] counts = new int[ParityTable.NAxes, ParityTable.NTiers];
        Unity.Collections.NativeArray<float> salience, headroom;
        Unity.Collections.NativeArray<int> assign;

        Rng rng;
        readonly Vector2 size;
        Grid grid;
        int frame;

        public CrowdWorld(Policy mode, int n, Vector2 sceneSize, uint seed, ParityTable table,
                          float budgetMs = 16.7f, float cap = 4.0f)
        {
            Mode = mode; size = sceneSize; Table = table;
            BudgetMs = budgetMs; Cap = cap;   // before Allocate: the ledger seeds D from Cap
            rowCost = new float[table.M];
            rng = new Rng(seed);
            Allocate(n);
        }

        public void Allocate(int n)
        {
            N = n;
            Pos = new Vector2[n]; Goal = new Vector2[n]; CoreP = new Vector2[n];
            CoreFrom = new Vector2[n];
            Speed = new float[n]; Sig = new float[n]; Phase = new float[n];
            Dmeas = new float[n]; CoreS = new float[n]; CoreLen = new float[n];
            InView = new bool[n]; SimTier = new sbyte[n]; VisTier = new sbyte[n];
            Row = new int[n]; Beh = new sbyte[n]; Nav = new sbyte[n]; Anim = new sbyte[n]; Geo = new sbyte[n];

            var r = new Rng(12345u);
            for (int i = 0; i < n; i++)
            {
                Pos[i] = new Vector2(r.Range(0f, size.x), r.Range(0f, size.y));
                NewGoal(i, ref r);
                Speed[i] = r.Range(1.0f, 1.6f);
                Phase[i] = r.NextFloat() * Mathf.PI * 2f;
            }
            grid = new Grid(size, 4f);

            Alloc?.Dispose(); Ledger?.Dispose();
            DisposeNative();
            if (Mode == Policy.Parity)
            {
                Alloc = new FidelityAllocator(Table);
                Ledger = new ErrorLedger(n, Cap, 7u);
                salience = new Unity.Collections.NativeArray<float>(n, Unity.Collections.Allocator.Persistent);
                headroom = new Unity.Collections.NativeArray<float>(n, Unity.Collections.Allocator.Persistent);
                assign = new Unity.Collections.NativeArray<int>(n, Unity.Collections.Allocator.Persistent);
            }
            MassLod.Resize(n);
            // seed the cost model so frame 0 has a usable cost vector; RLS takes over from
            // the first measured frame and nothing authored survives (invariant 1)
            Cost.Theta[0] = 0.0008;
            for (int f = 1; f < RlsCostModel.NFeat; f++) Cost.Theta[f] = 0.0022 / (1 + (f - 1) % 3);
            frame = 0;
        }

        void NewGoal(int i, ref Rng r)
        {
            Goal[i] = new Vector2(r.Range(2f, size.x - 2f), r.Range(2f, size.y - 2f));
            CoreFrom[i] = Pos[i];
            CoreP[i] = Pos[i];
            CoreS[i] = 0f;
            CoreLen[i] = Mathf.Max((Goal[i] - CoreFrom[i]).magnitude, 0.001f);
        }

        // -- per-frame -----------------------------------------------------------------

        public void Step(Vector2 camPos, float camYaw, float measuredFrameMs)
        {
            float t0 = Time.realtimeSinceStartup;
            MassLod.Step(Pos, N, camPos, camYaw, Sig, InView, SimTier, VisTier);

            if (Mode == Policy.Baseline) ApplyBaselineTiers();
            else RunAllocator(measuredFrameMs);

            StepCore();
            StepFine();
            AccrueDivergence();
            StepMs = (Time.realtimeSinceStartup - t0) * 1000f;
            frame++;
        }

        void ApplyBaselineTiers()
        {
            // MassLOD gives one level per agent and applies it to every axis at once.
            for (int i = 0; i < N; i++)
            {
                sbyte t = SimTier[i];
                Beh[i] = t; Nav[i] = t; Anim[i] = VisTier[i]; Geo[i] = VisTier[i];
                Row[i] = -1;
            }
        }

        void RunAllocator(float measuredFrameMs)
        {
            // invariant 1: the cost of a tier comes from measured frame time, never authored
            Histogram();
            if (frame > 2 && measuredFrameMs > 0f) Cost.Update(counts, N, measuredFrameMs);
            Cost.ThetaAxis(thetaAxis);
            Cost.RowCosts(Table, thetaAxis, rowCost);

            Ledger.Cap = Cap;
            Ledger.Refresh(N);
            for (int i = 0; i < N; i++)
            {
                salience[i] = 1f / (1f + Sig[i] / 20f);
                headroom[i] = Ledger.Headroom[i];
            }

            float t0 = Time.realtimeSinceStartup;
            Alloc.SetCosts(Table, rowCost);
            Last = Alloc.Solve(salience, headroom, N, BudgetMs, assign);
            AllocMs = (Time.realtimeSinceStartup - t0) * 1000f;

            Promotes = 0; Demotes = 0;
            for (int i = 0; i < N; i++)
            {
                int row = assign[i];
                Row[i] = row;
                sbyte nb = (sbyte)Table.TierOf(row, 0);
                if (Beh[i] == 3 && nb < 3)
                {
                    // reconciliation: the surrogate is abandoned and the agent is snapped back
                    // onto the core it never stopped tracking (invariant 4), and its ledger is
                    // the only thing a reset ever touches (invariant 3).
                    Pos[i] = CoreP[i];
                    Ledger.Reset(i);
                    Dmeas[i] = 0f;
                    Restorations++; Promotes++;
                }
                else if (Beh[i] < 3 && nb == 3) Demotes++;
                Beh[i] = nb;
                Nav[i] = (sbyte)Table.TierOf(row, 1);
                Anim[i] = (sbyte)Table.TierOf(row, 2);
                Geo[i] = (sbyte)Table.TierOf(row, 3);
            }
        }

        /// <summary>Invariant 4: stepped for every agent at every tier, never allocated.</summary>
        void StepCore()
        {
            for (int i = 0; i < N; i++)
            {
                CoreS[i] += Speed[i] * Dt;
                float u = Mathf.Clamp01(CoreS[i] / CoreLen[i]);
                CoreP[i] = Vector2.Lerp(CoreFrom[i], Goal[i], u);
            }
        }

        static readonly int[] Stride = { 1, 3, 10, 1 };

        void StepFine()
        {
            grid.Build(Pos, N);
            for (int i = 0; i < N; i++)
            {
                int beh = Beh[i];
                if (beh == 3)
                {
                    // surrogate: ride the core, no local behaviour at all
                    Pos[i] = CoreP[i];
                }
                else if (frame % Stride[beh] == 0)
                {
                    Vector2 to = Goal[i] - Pos[i];
                    float d = to.magnitude;
                    Vector2 dir = d > 1e-4f ? to / d : Vector2.zero;
                    Vector2 sep = Vector2.zero;
                    int nav = Nav[i];
                    if (nav <= 1)
                    {
                        int budget = nav == 0 ? 12 : 4;   // orca vs orca_sparse
                        sep = grid.Separation(Pos, i, 1.6f, budget);
                    }
                    Vector2 v = (dir + sep * 1.4f).normalized * Speed[i] * Stride[beh] * Dt;
                    Pos[i] += v;
                    Pos[i] = new Vector2(Mathf.Clamp(Pos[i].x, 0f, size.x), Mathf.Clamp(Pos[i].y, 0f, size.y));
                }
                int anim = Anim[i];
                if (anim < 3 && frame % Stride[anim] == 0) Phase[i] += Speed[i] * Dt * Stride[anim] * 6f;

                if ((Goal[i] - Pos[i]).sqrMagnitude < 1.0f || CoreS[i] >= CoreLen[i]) NewGoal(i, ref rng);
            }
        }

        void AccrueDivergence()
        {
            if (Mode == Policy.Parity)
            {
                for (int i = 0; i < N; i++)
                {
                    float rate = Table.Err[Row[i]];
                    Ledger.Accrue(i, rate);
                    Dmeas[i] = Ledger.D[i];
                    if (Ledger.D[i] > Cap + 1e-4f) CapBreaches++;
                }
            }
            else
            {
                // same metric on both sides, driven by the baseline's own tier, so the two
                // panels are directly comparable. Nothing resets it: MassLOD has no ledger.
                for (int i = 0; i < N; i++)
                {
                    float rate = (float)(ParityTable.AxisErr[0, Beh[i]] + ParityTable.AxisErr[1, Nav[i]]);
                    Dmeas[i] += rate;
                }
            }
        }

        public void Histogram()
        {
            System.Array.Clear(counts, 0, counts.Length);
            for (int i = 0; i < N; i++)
            {
                counts[0, Beh[i]]++; counts[1, Nav[i]]++; counts[2, Anim[i]]++; counts[3, Geo[i]]++;
            }
        }

        public int[,] Counts { get { Histogram(); return counts; } }

        public float MaxDivergence()
        {
            float m = 0f;
            for (int i = 0; i < N; i++) if (Dmeas[i] > m) m = Dmeas[i];
            return m;
        }

        void DisposeNative()
        {
            if (salience.IsCreated) salience.Dispose();
            if (headroom.IsCreated) headroom.Dispose();
            if (assign.IsCreated) assign.Dispose();
        }

        public void Dispose() { Alloc?.Dispose(); Ledger?.Dispose(); DisposeNative(); }
    }

    /// <summary>Uniform grid for neighbour separation; the cost of a full query is what the
    /// navigation axis is buying.</summary>
    sealed class Grid
    {
        readonly int nx, ny;
        readonly float cell;
        readonly int[] head;
        int[] next;

        public Grid(Vector2 size, float cellSize)
        {
            cell = cellSize;
            nx = Mathf.Max(1, Mathf.CeilToInt(size.x / cell));
            ny = Mathf.Max(1, Mathf.CeilToInt(size.y / cell));
            head = new int[nx * ny];
        }

        int Idx(Vector2 p)
        {
            int cx = Mathf.Clamp((int)(p.x / cell), 0, nx - 1);
            int cy = Mathf.Clamp((int)(p.y / cell), 0, ny - 1);
            return cy * nx + cx;
        }

        public void Build(Vector2[] pos, int n)
        {
            if (next == null || next.Length < n) next = new int[n];
            for (int i = 0; i < head.Length; i++) head[i] = -1;
            for (int i = 0; i < n; i++) { int c = Idx(pos[i]); next[i] = head[c]; head[c] = i; }
        }

        public Vector2 Separation(Vector2[] pos, int i, float radius, int budget)
        {
            Vector2 acc = Vector2.zero;
            int cx = Mathf.Clamp((int)(pos[i].x / cell), 0, nx - 1);
            int cy = Mathf.Clamp((int)(pos[i].y / cell), 0, ny - 1);
            float r2 = radius * radius;
            int seen = 0;
            for (int dy = -1; dy <= 1 && seen < budget; dy++)
            {
                int y = cy + dy; if (y < 0 || y >= ny) continue;
                for (int dx = -1; dx <= 1 && seen < budget; dx++)
                {
                    int x = cx + dx; if (x < 0 || x >= nx) continue;
                    for (int j = head[y * nx + x]; j >= 0 && seen < budget; j = next[j])
                    {
                        if (j == i) continue;
                        Vector2 d = pos[i] - pos[j];
                        float m2 = d.sqrMagnitude;
                        if (m2 > r2 || m2 < 1e-6f) continue;
                        acc += d / m2;
                        seen++;
                    }
                }
            }
            return acc;
        }
    }
}
