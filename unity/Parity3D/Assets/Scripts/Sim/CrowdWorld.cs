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
        public float[] Latent;               // [n * LatentDim] the behaviour latent being decoded
        public float[] SpeedScale, LatBias;  // decoded behaviour heads
        public float[] JointBlend;           // decoded gait, consumed by the renderer
        public float[] Speed, Sig, Phase, Dmeas, CoreS, CoreLen;
        public Vector2[] CoreFrom;
        public bool[] InView;
        /// <summary>Context band from local crowd density, standing in for sim/invariant.py's
        /// free / approach / choke / queue contexts. It selects the accrual rate.</summary>
        public sbyte[] Ctx;
        public sbyte[] SimTier, VisTier;
        public int[] Row;                    // parity: table row; baseline: -1
        public sbyte[] Beh, Nav, Anim, Geo;

        public readonly ParityTable Table;
        public readonly MassLodAssigner MassLod = new MassLodAssigner();
        public FidelityAllocator Alloc;
        public ErrorLedger Ledger;
        public readonly RlsCostModel Cost = new RlsCostModel();

        // sim/tiered.py sets the budget as n * (cmin + frac * (cmax - cmin)), so it scales with
        // the crowd and with whatever the cost model has learned. An absolute millisecond budget
        // is the honest thing to show eventually, but at a few hundred capsule agents it is never
        // binding, the allocator returns at lambda = 0 every frame and the demo shows nothing.
        /// <summary>Online RLS off by default: the cost vector comes from direct calibration,
        /// exactly as sim/tiered.py builds it from measure_costs(). The RLS is a faithful port
        /// and is exercised by the oracle harness, but it is not what prices this crowd.</summary>
        public bool UseOnlineRls;
        public float BudgetFrac = 0.25f;
        public bool AbsoluteBudget;
        public float TargetMs = 16.7f;
        public float BudgetMs = 16.7f;   // recomputed every frame unless AbsoluteBudget
        // cap = 300 * e_sur, as sim/tiered.py builds it from the occupancy-weighted mean rate
        public float Cap = (float)(300.0 * ParityTable.PlazaESur);
        /// <summary>Measured per-context surrogate KL rate. The ledger spends the ACTUAL rate
        /// while the table admitted against the worst one, so D can only ever land under the
        /// cap -- that asymmetry is what makes invariant 3 exact rather than probable.</summary>
        public static readonly float[] ERate = System.Array.ConvertAll(ParityTable.PlazaERate, v => (float)v);
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
                          float budgetMs = 16.7f, float cap = -1f)
        {
            Mode = mode; size = sceneSize; Table = table;
            BudgetMs = budgetMs;
            if (cap > 0f) Cap = cap;          // before Allocate: the ledger seeds D from Cap
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
            InView = new bool[n]; SimTier = new sbyte[n]; VisTier = new sbyte[n]; Ctx = new sbyte[n];
            Latent = new float[n * LatentDim];
            SpeedScale = new float[n]; LatBias = new float[n]; JointBlend = new float[n];
            Row = new int[n]; Beh = new sbyte[n]; Nav = new sbyte[n]; Anim = new sbyte[n]; Geo = new sbyte[n];

            var r = new Rng(12345u);
            for (int i = 0; i < n; i++)
            {
                Pos[i] = new Vector2(r.Range(0f, size.x), r.Range(0f, size.y));
                NewGoal(i, ref r);
                Speed[i] = r.Range(1.0f, 1.6f);
                Phase[i] = r.NextFloat() * Mathf.PI * 2f;
                SpeedScale[i] = 1f;
                // nested-dropout ordering: leading dimensions carry the most energy, so a
                // truncated decode loses the tail first. That is the whole point of the axis.
                for (int d = 0; d < LatentDim; d++) Latent[i * LatentDim + d] = r.Range(-1f, 1f) / (1f + d);
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

        // -- calibration ---------------------------------------------------------------

        /// <summary>Pin every agent at one axis-tier and time the fine step, per tier, as
        /// sim/tiered.py::measure_costs does. Runs on a throwaway crowd so the live one is not
        /// perturbed. Returns ms per agent per frame; the caller subtracts the all-tier-3 floor.
        /// Nothing authored enters the result, which is invariant 1.</summary>
        public static double CalibrateAxes(Vector2 size, ParityTable table, int n,
                                           out double floorMsPerAgent, int reps = 24)
        {
            var w = new CrowdWorld(Policy.Parity, n, size, 99u, table);
            double Time4(int b, int nv, int a, int g)
            {
                for (int i = 0; i < w.N; i++)
                {
                    w.Beh[i] = (sbyte)b; w.Nav[i] = (sbyte)nv; w.Anim[i] = (sbyte)a; w.Geo[i] = (sbyte)g;
                }
                w.grid.Build(w.Pos, w.N);
                w.StepCore(); w.StepFine(); w.frame++;            // warm up
                float t0 = Time.realtimeSinceStartup;
                // frame MUST advance: the behaviour and animation tiers are update strides, so
                // holding frame at 0 makes every tier run every step and measures only the
                // decode width. That reads back as a non-monotone cost column.
                for (int r = 0; r < reps; r++) { w.StepCore(); w.StepFine(); w.frame++; }
                return (Time.realtimeSinceStartup - t0) * 1000.0 / (reps * (double)w.N);
            }

            floorMsPerAgent = Time4(3, 3, 3, 3);
            var th = new double[ParityTable.NAxes, ParityTable.NTiers];
            for (int ax = 0; ax < ParityTable.NAxes; ax++)
                for (int t = 0; t < ParityTable.NTiers - 1; t++)
                {
                    int[] pin = { 3, 3, 3, 3 };
                    pin[ax] = t;
                    double ms = Time4(pin[0], pin[1], pin[2], pin[3]);
                    th[ax, t] = System.Math.Max(ms - floorMsPerAgent, 0.0);
                }
            w.Dispose();
            CalibratedTheta = th;
            return floorMsPerAgent;
        }

        public static double[,] CalibratedTheta;

        /// <summary>Price the full table from a calibration and drop the dominated rows.</summary>
        public static ParityTable PricedAndPruned(ParityTable full, double floorMs, double[,] theta,
                                                  out int keptCount)
        {
            var m = new RlsCostModel();
            m.SetCalibrated(floorMs, theta);
            var thAxis = new double[ParityTable.NAxes, ParityTable.NTiers];
            m.ThetaAxis(thAxis);
            var cost = new float[full.M];
            m.RowCosts(full, thAxis, cost);
            int[] kept;
            var pruned = full.PruneDominated(cost, out kept);
            keptCount = kept.Length;
            return pruned;
        }

        public void ApplyCalibration(double floorMsPerAgent, double[,] theta)
        {
            Cost.SetCalibrated(floorMsPerAgent, theta);
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
            if (UseOnlineRls && frame > 2 && measuredFrameMs > 0f) Cost.Update(counts, N, measuredFrameMs);
            Cost.ThetaAxis(thetaAxis);
            Cost.RowCosts(Table, thetaAxis, rowCost);

            float cmin = float.MaxValue, cmax = 0f;
            for (int c = 0; c < rowCost.Length; c++)
            {
                if (rowCost[c] < cmin) cmin = rowCost[c];
                if (rowCost[c] > cmax) cmax = rowCost[c];
            }
            BudgetMs = AbsoluteBudget ? TargetMs : N * (cmin + BudgetFrac * (cmax - cmin));

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

        // MassLOD's mechanism, and ONLY MassLOD's: tick the agent every k frames. Skipping
        // frames is what makes the baseline diverge, and it has no ledger to pay it back.
        static readonly int[] Stride = { 1, 3, 10, 1 };

        // Fraction of the reference process a baseline agent misses at each tick period, the
        // analogue of sim/tiered.py::_kl_baseline: 1 - 1/k for k = 1, 3, 10, and never.
        static readonly float[] BaselineMiss = { 0f, 2f / 3f, 0.9f, 1f };

        public const int LatentDim = 16;

        // The axes have to cost what they are named after, or the RLS correctly concludes that
        // every configuration costs the same and the allocator has nothing to decide.
        //
        //   behaviour  latent16 / latent8 / latent4 / core  -> decode over k dims
        //   animation  skeletal_ik / skeletal / vat / none  -> evaluate j joints
        //   navigation orca / orca_sparse / field / core    -> neighbour query budget
        //   geometry                                        -> render only, see Director
        //
        // The decode and the joint evaluation are real: their outputs steer the agent and pose
        // it, so neither can be folded away, and the cost ratio between tier 0 and tier 3 is
        // what the cost model actually measures.
        static readonly int[] DecodeDims = { LatentDim, 8, 4, 0 };
        static readonly int[] JointCount = { 12, 8, 3, 0 };
        static readonly int[] NavBudget = { 24, 8, 0, 0 };

        static readonly float[] DecSpeedW = new float[LatentDim];
        static readonly float[] DecLatW = new float[LatentDim];

        static CrowdWorld()
        {
            // fixed pseudo-random decode weights; the point is the arithmetic, not the values
            var r = new Rng(0xC0FFEEu);
            for (int d = 0; d < LatentDim; d++)
            {
                DecSpeedW[d] = r.Range(-1f, 1f) / LatentDim;
                DecLatW[d] = r.Range(-1f, 1f) / LatentDim;
            }
        }

        /// <summary>Behaviour decode: k retained latent dimensions -> speed scale and lateral
        /// bias. k = 0 is the surrogate, which decodes nothing and rides the core.</summary>
        void Decode(int i, int k)
        {
            float sp = 0f, lat = 0f;
            int b = i * LatentDim;
            for (int d = 0; d < k; d++)
            {
                float z = Latent[b + d];
                sp += z * DecSpeedW[d];
                lat += z * DecLatW[d];
            }
            SpeedScale[i] = 1f + 0.18f * sp;
            LatBias[i] = 0.45f * lat;
        }

        /// <summary>Gait decode: j joints blended into the pose the renderer swings limbs by.</summary>
        void Joints(int i, int j)
        {
            if (j == 0) return;             // tier 3: the pose is frozen
            float acc = 0f;
            for (int k = 0; k < j; k++) acc += Mathf.Sin(Phase[i] * (1f + k * 0.37f) + k);
            JointBlend[i] = acc / j;
        }

        static sbyte CtxOf(int neighbours)
        {
            if (neighbours >= 8) return 3;   // queue
            if (neighbours >= 4) return 2;   // choke: the expensive context to be wrong in
            if (neighbours >= 1) return 1;   // approach
            return 0;                        // free
        }

        void StepFine()
        {
            grid.Build(Pos, N);
            for (int i = 0; i < N; i++)
            {
                // context is read from the core's neighbourhood, so it is available at every
                // tier including the surrogate, which never runs a neighbour query of its own
                Ctx[i] = CtxOf(grid.Count(Pos, i, 2.2f, 12));
                int beh = Beh[i];
                if (beh == 3)
                {
                    // surrogate: ride the core, no local behaviour at all
                    Pos[i] = CoreP[i];
                }
                else if (Mode == Policy.Baseline ? frame % Stride[beh] == 0 : true)
                {
                    // PARITY's behaviour tiers are latent WIDTH (latent16 / latent8 / latent4):
                    // every one of them runs every frame, and the only way to stop paying for
                    // the fine step is to drop to the surrogate -- which costs divergence. The
                    // baseline instead buys its saving by skipping frames outright.
                    Decode(i, DecodeDims[beh]);
                    Vector2 to = Goal[i] - Pos[i];
                    float d = to.magnitude;
                    Vector2 dir = d > 1e-4f ? to / d : Vector2.zero;
                    Vector2 sep = Vector2.zero;
                    int budget = NavBudget[Nav[i]];
                    if (budget > 0) sep = grid.Separation(Pos, i, 1.6f, budget);
                    dir = new Vector2(dir.x - dir.y * LatBias[i], dir.y + dir.x * LatBias[i]);
                    float adv = Mode == Policy.Baseline ? Stride[beh] : 1;
                    Vector2 v = (dir + sep * 1.4f).normalized * Speed[i] * SpeedScale[i] * adv * Dt;
                    Pos[i] += v;
                    Pos[i] = new Vector2(Mathf.Clamp(Pos[i].x, 0f, size.x), Mathf.Clamp(Pos[i].y, 0f, size.y));
                }
                int anim = Anim[i];
                int astride = Mode == Policy.Baseline ? Stride[anim] : 1;
                if (anim < 3 && frame % astride == 0)
                {
                    Phase[i] += Speed[i] * Dt * astride * 6f;
                    Joints(i, JointCount[anim]);
                }

                if ((Goal[i] - Pos[i]).sqrMagnitude < 1.0f || CoreS[i] >= CoreLen[i]) NewGoal(i, ref rng);
            }
        }

        void AccrueDivergence()
        {
            // Only a surrogate diverges. Behaviour tiers 0-2 run the live process, so they
            // accrue nothing -- see ParityTable's error-column note.
            if (Mode == Policy.Parity)
            {
                // Two separate quantities, as in sim/tiered.py. Ledger.D is the admission
                // ledger: seeded with a desynchronising draw so restorations do not arrive as
                // one wave, and reset by reconciliation. Dmeas is the *measured* divergence the
                // figures quote, which starts at zero on both sides so the panels compare.
                for (int i = 0; i < N; i++)
                {
                    if (Beh[i] == 3)
                    {
                        float rate = ERate[Ctx[i]];
                        Ledger.Accrue(i, rate);
                        Dmeas[i] += rate;
                    }
                    if (Ledger.D[i] > Cap + 1e-4f) CapBreaches++;
                }
            }
            else
            {
                // Every baseline tier above 0 misses part of the reference process, not just
                // the lowest one -- that is what ticking at 1/3/10 frames costs. Nothing ever
                // resets it: MassLOD has no ledger, so the total only goes up.
                for (int i = 0; i < N; i++)
                    Dmeas[i] += ERate[Ctx[i]] * BaselineMiss[Beh[i]];
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

        public int Count(Vector2[] pos, int i, float radius, int cap)
        {
            int cx = Mathf.Clamp((int)(pos[i].x / cell), 0, nx - 1);
            int cy = Mathf.Clamp((int)(pos[i].y / cell), 0, ny - 1);
            float r2 = radius * radius;
            int seen = 0;
            for (int dy = -1; dy <= 1 && seen < cap; dy++)
            {
                int y = cy + dy; if (y < 0 || y >= ny) continue;
                for (int dx = -1; dx <= 1 && seen < cap; dx++)
                {
                    int x = cx + dx; if (x < 0 || x >= nx) continue;
                    for (int j = head[y * nx + x]; j >= 0 && seen < cap; j = next[j])
                        if (j != i && (pos[i] - pos[j]).sqrMagnitude <= r2) seen++;
                }
            }
            return seen;
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
