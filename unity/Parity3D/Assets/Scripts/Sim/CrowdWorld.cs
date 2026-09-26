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
        public Vector2[] PrevPos, Final;     // pre-step position (wall crossing); corridor exit
        public int[] Slot;                   // hub queue slot, -1 when not queued
        public bool[] Queuer;
        public sbyte[] Stage;                // corridor route leg: mouth / doorway / exit
        /// <summary>Facing, from actual displacement, and how much of a walk the agent is
        /// doing (0 standing .. 1 full stride). Both are presentation only.</summary>
        public float[] Heading, Walk;
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
        /// <summary>The two-salience allocator (alloc/factored.py). Null unless the table is the
        /// full state x view product; a pruned table falls back to the single-salience one.</summary>
        public FactoredAllocator Factored;
        /// <summary>Weight animation and geometry by projected AREA instead of by the one
        /// salience behaviour uses: 1 within ViewD0 metres, (ViewD0 / d)^2 beyond, 0 out of view.
        /// Area, because the geometry quality it multiplies is the pixel judge's count of wrong
        /// pixels at ViewD0, and a figure's pixel count falls as 1 / d^2.</summary>
        public bool ViewAware = true;
        public float ViewD0 = 10f;
        /// <summary>Weight of the new frame in the view salience's running average. Coverage
        /// flickers as people pass in front of each other; unsmoothed, that flicker is re-decided
        /// every frame and shows up as popping.</summary>
        public float ViewSmoothing = 0.25f;
        float[] stateSalM, headroomM;

        // -- the view side: several viewers, the pop ledger, occlusion ------------------
        /// <summary>1, or 2 for two viewers on this one simulation (co-op split screen): one
        /// behaviour per agent, a mesh and gait tier per agent per viewer.</summary>
        public int Viewers = 1;
        public const int MaxViewers = 2;
        public Vector2 Cam2; public float Yaw2;
        public readonly MassLodAssigner MassLod2 = new MassLodAssigner();
        public float[] Sig2; public bool[] InView2; sbyte[] simTier2, visTier2;
        /// <summary>[viewer][agent] render tiers. GeoV[0] is Geo; AnimV[0] mirrors Anim for one
        /// viewer, while Anim itself is the most detailed gait any viewer needs (it is computed once).</summary>
        public sbyte[][] GeoV, AnimV;
        public readonly PopTracker[] Pops = { new PopTracker(), new PopTracker() };
        /// <summary>Hold view pairs of agents with no pop token (PARITY only; MassLOD is only counted).</summary>
        public bool PopLedger = true;
        /// <summary>Price on changing an agent's view pair, per unit of its view salience
        /// (alloc/factored.py switch_cost): a preference in the objective that lowers the average
        /// pop rate, where the pop ledger only bounds the worst. 0 turns it off. Off by default:
        /// judged in pixels (ParityFrameBench image and pop), it bought few pops for much image,
        /// where the ledger alone cut pops ~17x for far less.</summary>
        public float SwitchCost = 0f;
        /// <summary>The same price on changing an agent's state pair, per unit of its state
        /// salience (alloc/factored.py state_switch_cost): frame-time noise moves the budget every
        /// frame, and this keeps marginal agents from flipping with it. Off by default: in the dense
        /// plaza the flips are the ledger forcing restorations, which it must not resist.</summary>
        public float StateSwitchCost;
        /// <summary>Ablation: PARITY with no divergence cap (Python's parity_nocap).</summary>
        public bool Uncapped;
        int[] visit, keyStart;
        int[] prevS;
        /// <summary>View salience from visible pixels in a coverage buffer instead of distance alone.</summary>
        public bool Occlusion = true;
        /// <summary>View salience is the agent's visible pixels in units of one agent's footprint
        /// at ViewD0 -- a nearer agent is worth its larger image, as Funkhouser's benefit and the
        /// pixel judge both count it. The cap is a numeric guard only.</summary>
        public float SalienceCap = 64f;
        public readonly CoverageBuffer Coverage = new CoverageBuffer();
        readonly Matrix4x4[] viewProj = new Matrix4x4[MaxViewers];
        readonly Vector2[] proj = new Vector2[MaxViewers];
        readonly bool[] hasView = new bool[MaxViewers];
        public float[][] VisiblePx;
        /// <summary>In-frustum agents at most a quarter visible, per viewer, last frame.</summary>
        public int[] Occluded = new int[MaxViewers];
        float[][] viewSalV;
        readonly float[][] smoothed = new float[MaxViewers][];
        int[][] holdV, assignV, prevV;
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
        /// <summary>When set (>= 0), the budget IS this: the baseline's own predicted spend this
        /// frame, so both policies are held to the same milliseconds.</summary>
        public float MatchBudgetMs = -1f;
        /// <summary>When set (> 0), a target FRAME time, so the whole frame -- sim, allocator,
        /// render, GPU -- settles on the target on any machine without re-tuning. Model-based: the
        /// budget is the target less a smoothed estimate of what the frame costs beyond the
        /// allocation's own predicted spend (measured frame minus the predicted spend of the
        /// allocation that frame ran). Exact in one step when the cost model is, and stable for any
        /// cost-model scale error below 2x -- an integral controller on the frame time is not, once
        /// the decision is pipelined a frame ahead. FrameGain is the smoothing rate.</summary>
        public float FrameTargetMs;
        public float FrameGain = 0.1f;
        /// <summary>The target is a PERCENTILE of frame time, not its mean: the budget leaves
        /// z * std of headroom (z = 1.65 is the 95th percentile of a normal).</summary>
        public float FrameZ = 1.65f;
        float frameEma, frameVar, overheadEma = float.NaN, appliedCost = -1f;

        /// <summary>Frames overlap -- the CPU prepares frame t + 1 while the GPU draws frame t --
        /// so a frame costs max(CPU, GPU), not their sum, and a millisecond is not one currency.
        /// With DualResource each row has a CPU cost (behaviour, navigation, animation, and the
        /// draw submission its mesh needs) and a GPU cost (the rest of its mesh), each resource
        /// its own budget from its own measured time, and the allocator prices a row at
        /// cpu + Rho * gpu with Rho the ratio of the two multipliers. The existing single-budget
        /// solve is exact for a fixed Rho (the combined budget is B_cpu + Rho * B_gpu); Rho is
        /// searched each frame, warm from the last, until both budgets hold. A resource with
        /// slack gets price 0 -- which is how CPU fidelity becomes free while the GPU is the
        /// bottleneck, and the other way round.</summary>
        public bool DualResource;
        /// <summary>Last frame's main-thread CPU time and GPU time, set by the caller before Step.</summary>
        public float MeasuredCpuMs = -1f, MeasuredGpuMs = -1f;
        /// <summary>MeasuredGpuMs is only an upper bound: without a GPU timer, an overlapped frame
        /// that never waited for the GPU shows how long the GPU had, not how long it took.</summary>
        public bool GpuCensored;
        public float Rho = 1f, BudgetCpuMs, BudgetGpuMs, SpendCpuMs, SpendGpuMs;
        /// <summary>GPU share of each geometry tier's cost (ms per agent, relative to tier 3),
        /// from the render calibration; the rest of the calibrated geometry cost is CPU.</summary>
        public static readonly double[] GeoGpuTheta = new double[ParityTable.NTiers];
        float[] rowCpu, rowGpu, rowEff;
        float ovCpu = float.NaN, ovGpu = float.NaN, cpuEma, cpuVar, gpuEma, gpuVar, appliedCpu = -1f, appliedGpu = -1f;
        public float TargetMs = 16.7f;
        public float BudgetMs = 16.7f;   // recomputed every frame unless AbsoluteBudget
        // cap = 300 * e_sur, as sim/tiered.py builds it from the occupancy-weighted mean rate
        public float Cap;
        /// <summary>Measured per-context surrogate KL rate. The ledger spends the ACTUAL rate
        /// while the table admitted against the worst one, so D can only ever land under the
        /// cap -- that asymmetry is what makes invariant 3 exact rather than probable.</summary>
        public readonly float[] ERate;
        public readonly SceneSpec Spec;
        public readonly CrowdScene Scene;
        public AllocResult Last;

        /// <summary>A global render setting (shadows, say) priced in the same Lagrangian as the
        /// per-agent tiers. Everything is measured against option 0, the full setting: its frame
        /// cost, and by the pixel judge its geometry quality and its loss on the scene alone.</summary>
        public sealed class GlobalOption
        {
            public string Name;
            public float FixedMs;          // scene cost over option 0's, ms per frame
            public float DrawnMs;          // cost over option 0's per drawn agent at the cheapest tier
            public double[] GeoTheta;      // ms per drawn agent of geometry tiers over tier 3
            public double[] GeoQuality;    // per geometry tier, against option 0's tier 0
            public float SceneLoss;        // the scene's pixels lost, in objective units
        }

        /// <summary>Null: no global choice. Else PARITY picks Option every frame.</summary>
        public GlobalOption[] Options;
        public int Option;
        public float OptionHysteresis = 0.01f;
        /// <summary>The option is set from outside (a fixed setting, or a baseline's own scaler);
        /// only that one is solved.</summary>
        public bool OptionExternal;
        public long[] OptionFrames;
        int decidedOption;
        float[] optCost;
        readonly double[,] optTheta = new double[ParityTable.NAxes, ParityTable.NTiers];
        public float AllocMs, StepMs;
        /// <summary>Predicted cost of the tier work the budget actually governs. The rest of
        /// StepMs is the assigner and the spatial grid, which are tier-independent and which
        /// both policies pay identically.</summary>
        public float AllocatableMs;
        public int Restorations, Promotes, Demotes;
        /// <summary>Agents whose ledger passed the cap. The feasibility mask makes this
        /// impossible by construction, so it is displayed as a live assertion.</summary>
        public int CapBreaches;
        /// <summary>One agent followed for the trace. The max over N agents is flat once N is
        /// large -- someone is always near the cap -- so the sawtooth only shows per agent.</summary>
        public int Tracked;
        public float TrackedD => Tracked < N ? Dmeas[Tracked] : 0f;
        public int TrackedRestores;

        readonly float[] rowCost;
        readonly double[,] thetaAxis = new double[ParityTable.NAxes, ParityTable.NTiers];
        readonly int[,] counts = new int[ParityTable.NAxes, ParityTable.NTiers];
        Unity.Collections.NativeArray<float> salience, headroom;
        Unity.Collections.NativeArray<int> assign;

        Rng rng;
        readonly Vector2 size;
        Grid grid;
        int frame;

        public CrowdWorld(Policy mode, int n, SceneSpec spec, uint seed, ParityTable table,
                          float budgetMs = 16.7f, float cap = -1f)
        {
            Mode = mode; Spec = spec; size = spec.Size; Table = table;
            Scene = CrowdScene.Make(spec);
            ERate = System.Array.ConvertAll(spec.ERate, v => (float)v);
            BudgetMs = budgetMs;
            Cap = cap > 0f ? cap : spec.Cap;  // before Allocate: the ledger seeds D from Cap
            rowCost = new float[table.M];
            rng = new Rng(seed);
            Allocate(n);
        }

        public void Allocate(int n)
        {
            N = n;
            Pos = new Vector2[n]; Goal = new Vector2[n]; CoreP = new Vector2[n];
            CoreFrom = new Vector2[n]; PrevPos = new Vector2[n]; Final = new Vector2[n];
            Slot = new int[n]; Queuer = new bool[n]; Stage = new sbyte[n];
            Heading = new float[n]; Walk = new float[n];
            Speed = new float[n]; Sig = new float[n]; Phase = new float[n];
            Dmeas = new float[n]; CoreS = new float[n]; CoreLen = new float[n];
            InView = new bool[n]; SimTier = new sbyte[n]; VisTier = new sbyte[n]; Ctx = new sbyte[n];
            Latent = new float[n * LatentDim];
            SpeedScale = new float[n]; LatBias = new float[n]; JointBlend = new float[n];
            Row = new int[n]; Beh = new sbyte[n]; Nav = new sbyte[n]; Anim = new sbyte[n]; Geo = new sbyte[n];

            var r = new Rng(12345u);
            for (int i = 0; i < n; i++) Slot[i] = -1;
            Scene.Begin(this);
            for (int i = 0; i < n; i++)
            {
                // position, goal and speed are the scene's; the plaza draws them in the order
                // this loop always did, so its crowd is bit-identical to the plaza-only build
                Scene.SpawnOne(this, i, ref r);
                Phase[i] = r.NextFloat() * Mathf.PI * 2f;
                Walk[i] = 1f;
                Vector2 to = Goal[i] - Pos[i];
                Heading[i] = Mathf.Atan2(to.x, to.y);
                SpeedScale[i] = 1f;
                // nested-dropout ordering: leading dimensions carry the most energy, so a
                // truncated decode loses the tail first. That is the whole point of the axis.
                for (int d = 0; d < LatentDim; d++) Latent[i * LatentDim + d] = r.Range(-1f, 1f) / (1f + d);
            }
            grid = new Grid(size, 2.5f);          // >= the largest query radius (2.2 m)

            Alloc?.Dispose(); Ledger?.Dispose();
            DisposeNative();
            // both modes: the baseline's render-knapsack variant allocates the view half too
            {
                Alloc = new FidelityAllocator(Table);
                Factored = FactoredAllocator.TryCreate(Table);
                if (Factored != null) { Factored.Warm = true; Factored.Btol = 1e-3; }
                stateSalM = new float[n]; headroomM = new float[n]; prevS = new int[n];
                Ledger = new ErrorLedger(n, Cap, 7u);
                salience = new Unity.Collections.NativeArray<float>(n, Unity.Collections.Allocator.Persistent);
                headroom = new Unity.Collections.NativeArray<float>(n, Unity.Collections.Allocator.Persistent);
                assign = new Unity.Collections.NativeArray<int>(n, Unity.Collections.Allocator.Persistent);
            }
            MassLod.Resize(n);
            MassLod2.Resize(n);
            Sig2 = new float[n]; InView2 = new bool[n]; simTier2 = new sbyte[n]; visTier2 = new sbyte[n];
            GeoV = new[] { Geo, new sbyte[n] };
            AnimV = new[] { new sbyte[n], new sbyte[n] };
            viewSalV = new[] { new float[n], new float[n] };
            holdV = new[] { new int[n], new int[n] };
            prevV = new[] { new int[n], new int[n] };
            assignV = new[] { new int[n], new int[n] };
            VisiblePx = new[] { new float[n], new float[n] };
            foreach (var p in Pops) p.Resize(n);
            frame = 0;
        }

        /// <summary>Retarget an agent and re-seat its core on where the agent actually is.</summary>
        public void SetGoal(int i, Vector2 g)
        {
            Goal[i] = g;
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
        public static double CalibrateAxes(SceneSpec spec, ParityTable table, int n,
                                           out double floorMsPerAgent, int reps = 8)
        {
            CalibratedTheta = MeasureAxes(spec, table, n, Policy.Parity, out floorMsPerAgent, reps);
            return floorMsPerAgent;
        }

        /// <summary>The same measurement under either policy's mechanism. MassLOD's tiers are
        /// frame strides and PARITY's are latent widths, so the same tier number costs each of
        /// them something different, and a cost-matched comparison has to price each policy's
        /// spend with its own measured costs.</summary>
        public static double[,] MeasureAxes(SceneSpec spec, ParityTable table, int n, Policy mode,
                                            out double floorMsPerAgent, int reps = 8, int blocks = 6)
        {
            var w = new CrowdWorld(mode, n, spec, 99u, table);
            void Pin(int[] c)
            {
                for (int i = 0; i < w.N; i++)
                {
                    w.Beh[i] = (sbyte)c[0]; w.Nav[i] = (sbyte)c[1]; w.Anim[i] = (sbyte)c[2]; w.Geo[i] = (sbyte)c[3];
                }
            }
            // Each configuration is a pin of all four axes. The one-axis-at-a-time contrast is
            // taken against the all-floor pin, except navigation: separation runs only for a
            // live agent, so against a surrogate every navigation tier reads as free. Its
            // contrast is taken on the cheapest live behaviour tier instead. The baseline never
            // splits the two (MassLOD sets navigation = behaviour), so its behaviour column is
            // the joint contrast and its navigation column 0.
            bool joint = mode == Policy.Baseline;
            var cfg = new System.Collections.Generic.List<int[]> { new[] { 3, 3, 3, 3 } };
            for (int t = 0; t < ParityTable.NTiers - 1; t++) cfg.Add(joint ? new[] { t, t, 3, 3 } : new[] { t, 3, 3, 3 });
            if (!joint)
            {
                cfg.Add(new[] { 2, 3, 3, 3 });
                for (int t = 0; t < ParityTable.NTiers - 1; t++) cfg.Add(new[] { 2, t, 3, 3 });
            }
            for (int ax = 2; ax < ParityTable.NAxes; ax++)
                for (int t = 0; t < ParityTable.NTiers - 1; t++)
                {
                    var c = new[] { 3, 3, 3, 3 }; c[ax] = t; cfg.Add(c);
                }
            // the configurations are timed in interleaved blocks and each keeps its fastest
            // block: the parallel step at large N is short enough that scheduler noise would
            // otherwise swamp the differences being measured
            var best = new double[cfg.Count];
            for (int k = 0; k < cfg.Count; k++) best[k] = double.MaxValue;
            for (int blk = 0; blk < blocks; blk++)
                for (int k = 0; k < cfg.Count; k++)
                {
                    Pin(cfg[k]);
                    w.grid.Build(w.Pos, w.N);
                    w.StepCore(); w.StepFine(); w.frame++;            // warm up
                    float t0 = Time.realtimeSinceStartup;
                    // frame MUST advance: the baseline's behaviour and animation tiers are update
                    // strides, so holding frame at 0 makes every tier run every step
                    for (int r = 0; r < reps; r++) { w.StepCore(); w.StepFine(); w.frame++; }
                    best[k] = System.Math.Min(best[k], (Time.realtimeSinceStartup - t0) * 1000.0 / (reps * (double)w.N));
                }

            floorMsPerAgent = best[0];
            var th = new double[ParityTable.NAxes, ParityTable.NTiers];
            int at = 1;
            for (int t = 0; t < ParityTable.NTiers - 1; t++) th[0, t] = System.Math.Max(best[at++] - best[0], 0.0);
            if (!joint)
            {
                double live = best[at++];
                for (int t = 0; t < ParityTable.NTiers - 1; t++) th[1, t] = System.Math.Max(best[at++] - live, 0.0);
            }
            for (int ax = 2; ax < ParityTable.NAxes; ax++)
                for (int t = 0; t < ParityTable.NTiers - 1; t++) th[ax, t] = System.Math.Max(best[at++] - best[0], 0.0);
            w.Dispose();
            return th;
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

        /// <summary>The render camera of viewer k, for the coverage pass. Without one the view
        /// salience falls back to the (d0 / d)^2 area model.</summary>
        public void SetView(int k, Matrix4x4 viewProjection, float projX, float projY)
        {
            Join();
            viewProj[k] = viewProjection;
            proj[k] = new Vector2(projX, projY);
            hasView[k] = true;
        }

        public bool[] SeenBy(int k) => k == 0 ? InView : InView2;

        /// <summary>Cumulative ms per stage, for profiling: see ProfNames.</summary>
        /// Main thread: masslod (view part), decide (wait + any serial decision + apply), core,
        /// fine, accrue, pops. The decision itself, wherever it runs: prep, view, coverage, solve,
        /// and total (the multiplier search inside solve). wait is the part of decide spent
        /// blocked on a pipelined decision.
        public readonly double[] Prof = new double[12];
        public static readonly string[] ProfNames = { "masslod", "decide", "core", "fine", "accrue", "pops", "view", "prep", "coverage", "total", "solve", "wait" };
        static readonly double TickMs = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        long tick, atick;
        void Lap(int k) { long now = System.Diagnostics.Stopwatch.GetTimestamp(); Prof[k] += (now - tick) * TickMs; tick = now; }
        void ALap(int k) { long now = System.Diagnostics.Stopwatch.GetTimestamp(); Prof[k] += (now - atick) * TickMs; atick = now; }

        /// <summary>Decide the next frame's tiers on a worker thread while this frame renders, for
        /// either policy. The decision sees the ledger exactly as the next frame starts (it is
        /// taken after accrual) and the camera one frame late. Set before the first Step.</summary>
        public bool Pipelined;
        System.Threading.Tasks.Task decision;
        bool decided;
        bool CanPipeline => Mode == Policy.Baseline || (ViewAware && Factored != null);

        /// <summary>Waits for a pipelined decision in flight. Call before touching the world
        /// from outside Step.</summary>
        public void Join()
        {
            if (decision == null) return;
            decision.Wait();
            decision = null;
        }

        float decCpu, decGpu, decPairCpu, decPairGpu;
        bool decGpuCensored;

        void Decide(float measuredFrameMs, float pairedCost)
        {
            if (Mode == Policy.Parity) RunAllocator(measuredFrameMs, pairedCost);
            else
            {
                if (Pipelined) MassLod.Tiers(Sig, InView, SimTier, VisTier);
                if (RenderKnapsack) RunAllocator(measuredFrameMs, pairedCost);
            }
        }

        public void Step(Vector2 camPos, float camYaw, float measuredFrameMs)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            tick = t0;
            Join();
            Prof[11] += (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * TickMs;
            bool pipe = Pipelined && CanPipeline;
            bool viewOnly = Mode == Policy.Parity || pipe;
            MassLod.Step(Pos, N, camPos, camYaw, Sig, InView, SimTier, VisTier, viewOnly);
            if (Viewers > 1) MassLod2.Step(Pos, N, Cam2, Yaw2, Sig2, InView2, simTier2, visTier2, viewOnly);
            Lap(0);

            // the predicted spend of the allocation the measured (previous) frame ran
            float paired = appliedCost;
            decCpu = MeasuredCpuMs; decGpu = MeasuredGpuMs; decGpuCensored = GpuCensored; decPairCpu = appliedCpu; decPairGpu = appliedGpu;
            if (!decided) Decide(measuredFrameMs, paired);
            decided = false;
            if (Mode == Policy.Baseline) { ApplyBaselineTiers(); appliedCost = AllocatableMs; }
            else { ApplyDecision(); appliedCost = AllocatableMs; appliedCpu = SpendCpuMs; appliedGpu = SpendGpuMs; }
            Lap(1);

            StepCore();
            Lap(2);
            StepFine();
            Lap(3);
            AccrueDivergence();
            Lap(4);
            int views = Mode == Policy.Parity ? Viewers : 1;
            for (int k = 0; k < views; k++)
            {
                // the same area measure for both policies: (d0 / d)^2 from the LOD camera distance
                var sigK = k == 0 ? Sig : Sig2;
                if (popArea == null || popArea.Length != N) popArea = new float[N];
                for (int i = 0; i < N; i++)
                {
                    float r = ViewD0 / Mathf.Max(sigK[i], 1e-3f);
                    popArea[i] = Mathf.Min(1f, r * r);
                }
                Pops[k].Update(AnimV[k], GeoV[k], SeenBy(k), N, popArea);
            }
            Lap(5);
            frame++;
            if (pipe)
            {
                float ms = measuredFrameMs;
                decided = true;
                decision = System.Threading.Tasks.Task.Run(() => Decide(ms, paired));
            }
            StepMs = (float)((System.Diagnostics.Stopwatch.GetTimestamp() - t0) * TickMs);
        }

        /// <summary>Baseline variants (Mode == Baseline). TimeSlice = 1, 3 or 10: uniform AI time
        /// slicing instead of MassLOD's simulation levels -- every agent, near or far, updated every
        /// TimeSlice frames, staggered (the simulation level with that stride, for all). RenderKnapsack: the view half chosen by a
        /// frame-budget knapsack (Funkhouser and Sequin 1993) over MassLOD's simulation levels,
        /// which it takes as given -- visual and simulation tiers still derived independently.</summary>
        public int TimeSlice;
        public bool RenderKnapsack;
        int[] stateLock;

        int BehStride(int beh) => TimeSlice > 0 ? TimeSlice : Stride[beh];
        sbyte SliceTier => (sbyte)(TimeSlice >= 10 ? 2 : TimeSlice >= 3 ? 1 : 0);

        void ApplyBaselineTiers()
        {
            // MassLOD gives one level per agent and applies it to every axis at once.
            for (int i = 0; i < N; i++)
            {
                sbyte t = TimeSlice > 0 ? SliceTier : SimTier[i];
                Beh[i] = t; Nav[i] = t; Anim[i] = VisTier[i]; Geo[i] = VisTier[i];
                if (RenderKnapsack)
                {
                    int row = assignV[0][i];
                    Anim[i] = (sbyte)Table.TierOf(row, 2);
                    Geo[i] = (sbyte)Table.TierOf(row, 3);
                }
                AnimV[0][i] = Anim[i];
                Row[i] = -1;
            }
        }

        bool decidedFactored;
        int decidedViewers;

        public int RhoSolves = 3;
        public float RhoTol = 0.02f;

        // one resource's overhead (measured minus the predicted spend of the allocation that ran)
        // and its spread, as the single-budget controller keeps them for the whole frame
        void Track(float measured, float paired, ref float overhead, ref float ema, ref float var, bool censored = false)
        {
            if (measured <= 0f || paired < 0f) return;
            float o = measured - paired;
            // a censored reading bounds the overhead from above and says nothing about the spread:
            // it can only pull an estimate down (a generous budget then makes the resource bind,
            // and the next exact reading pulls it back up)
            if (censored && !float.IsNaN(overhead))
            {
                if (o < overhead) overhead += FrameGain * (o - overhead);
                return;
            }
            overhead = float.IsNaN(overhead) ? o : overhead + FrameGain * (o - overhead);
            if (ema <= 0f) ema = measured;
            float dev = measured - ema;
            ema += 0.2f * dev;
            var = Mathf.Lerp(var, dev * dev, 0.05f);
        }

        void RunAllocator(float measuredFrameMs, float pairedCost)
        {
            atick = System.Diagnostics.Stopwatch.GetTimestamp();
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
            bool factored = ViewAware && Factored != null;
            int V = factored ? Mathf.Clamp(Viewers, 1, MaxViewers) : 1;
            if (factored && V > 1)
            {
                // each extra viewer can draw every agent once more
                Factored.SetCosts(rowCost);
                cmax += (float)((V - 1) * Factored.MaxViewCost);
            }
            bool dual = DualResource && FrameTargetMs > 0f && factored;
            if (dual)
            {
                if (rowCpu == null || rowCpu.Length != rowCost.Length)
                {
                    rowCpu = new float[rowCost.Length]; rowGpu = new float[rowCost.Length]; rowEff = new float[rowCost.Length];
                }
                for (int r = 0; r < rowCost.Length; r++)
                {
                    rowGpu[r] = (float)GeoGpuTheta[Table.TierOf(r, 3)];
                    rowCpu[r] = rowCost[r] - rowGpu[r];
                }
                Track(decCpu, decPairCpu, ref ovCpu, ref cpuEma, ref cpuVar);
                Track(decGpu, decPairGpu, ref ovGpu, ref gpuEma, ref gpuVar, decGpuCensored);
                float gmin = float.MaxValue, gmax = 0f, pmin = float.MaxValue, pmax = 0f;
                for (int r = 0; r < rowCost.Length; r++)
                {
                    gmin = Mathf.Min(gmin, rowGpu[r]); gmax = Mathf.Max(gmax, rowGpu[r]);
                    pmin = Mathf.Min(pmin, rowCpu[r]); pmax = Mathf.Max(pmax, rowCpu[r]);
                }
                BudgetCpuMs = float.IsNaN(ovCpu) ? N * pmin : Mathf.Clamp(FrameTargetMs - FrameZ * Mathf.Sqrt(cpuVar) - ovCpu, N * pmin, N * pmax);
                BudgetGpuMs = float.IsNaN(ovGpu) ? N * gmin : Mathf.Clamp(FrameTargetMs - FrameZ * Mathf.Sqrt(gpuVar) - ovGpu, N * gmin, N * gmax);
            }
            else if (FrameTargetMs > 0f)
            {
                if (measuredFrameMs > 0f && pairedCost >= 0f)
                {
                    float o = measuredFrameMs - pairedCost;
                    overheadEma = float.IsNaN(overheadEma) ? o : overheadEma + FrameGain * (o - overheadEma);
                    if (frameEma <= 0f) frameEma = measuredFrameMs;
                    float dev = measuredFrameMs - frameEma;
                    frameEma += 0.2f * dev;
                    frameVar = Mathf.Lerp(frameVar, dev * dev, 0.05f);
                }
                float b = float.IsNaN(overheadEma) ? N * cmin
                        : FrameTargetMs - FrameZ * Mathf.Sqrt(frameVar) - overheadEma;
                BudgetMs = Mathf.Clamp(b, N * cmin, N * cmax);
            }
            else
            BudgetMs = AbsoluteBudget ? TargetMs
                     : MatchBudgetMs >= 0f ? MatchBudgetMs
                     : N * (cmin + BudgetFrac * (cmax - cmin));

            Ledger.Cap = Uncapped ? 1e30f : Cap;
            Ledger.Refresh(N);
            for (int i = 0; i < N; i++)
            {
                // behaviour is simulated once, so its salience is the nearest viewer's
                float sig = V > 1 ? Mathf.Min(Sig[i], Sig2[i]) : Sig[i];
                float s = 1f / (1f + sig / 20f);
                salience[i] = s;
                headroom[i] = Ledger.Headroom[i];
                if (factored) { stateSalM[i] = s; headroomM[i] = headroom[i]; }
            }
            ALap(7);
            if (factored)
                for (int k = 0; k < V; k++)
                {
                    var seen = SeenBy(k);
                    var sigK = k == 0 ? Sig : Sig2;
                    var vs = viewSalV[k];
                    if (smoothed[k] == null || smoothed[k].Length != N) smoothed[k] = new float[N];
                    var sm = smoothed[k];
                    Occluded[k] = 0;
                    if (Occlusion && hasView[k])
                    {
                        ALap(6);
                        float refPx = Coverage.Compute(Pos, N, viewProj[k], proj[k].x, proj[k].y, ViewD0, VisiblePx[k]);
                        ALap(8);
                        for (int i = 0; i < N; i++)
                        {
                            vs[i] = Mathf.Min(SalienceCap, VisiblePx[k][i] / refPx);
                            if (seen[i])
                            {
                                float r = ViewD0 / Mathf.Max(sigK[i], 1e-3f);
                                if (VisiblePx[k][i] < 0.25f * Mathf.Min(1f, r * r) * refPx) Occluded[k]++;
                            }
                        }
                    }
                    else
                        for (int i = 0; i < N; i++)
                        {
                            // in view, Sig is the camera distance
                            float r = ViewD0 / Mathf.Max(sigK[i], 1e-3f);
                            vs[i] = seen[i] ? Mathf.Min(SalienceCap, r * r) : 0f;
                        }
                    for (int i = 0; i < N; i++)
                    {
                        sm[i] = frame == 0 ? vs[i] : Mathf.Lerp(sm[i], vs[i], ViewSmoothing);
                        vs[i] = sm[i];
                    }
                    for (int i = 0; i < N; i++)
                        prevV[k][i] = frame == 0 ? -1 : Factored.ViewPairOfKey(AnimV[k][i] * ParityTable.NTiers + GeoV[k][i]);
                    if (k == 0)
                        for (int i = 0; i < N; i++)
                            prevS[i] = frame == 0 ? -1 : Factored.StatePairOfKey(Beh[i] * ParityTable.NTiers + Nav[i]);
                    if (PopLedger)
                    {
                        var h = Pops[k].Holds(seen, N);
                        for (int i = 0; i < N; i++) holdV[k][i] = h[i] >= 0 ? Factored.ViewPairOfKey(h[i]) : -1;
                    }
                }

            ALap(6);
            if (factored)
            {
                Factored.SetCosts(rowCost);
                var sal = V == 1 ? new[] { viewSalV[0] } : viewSalV;
                var holds = PopLedger ? (V == 1 ? new[] { holdV[0] } : holdV) : null;
                var asg = V == 1 ? new[] { assignV[0] } : assignV;
                var prevs = SwitchCost > 0f ? (V == 1 ? new[] { prevV[0] } : prevV) : null;
                if (Mode == Policy.Baseline)
                {
                    // the render knapsack takes MassLOD's simulation level as given
                    if (stateLock == null || stateLock.Length != N) stateLock = new int[N];
                    for (int i = 0; i < N; i++)
                    {
                        int t = TimeSlice > 0 ? SliceTier : SimTier[i];
                        stateLock[i] = Factored.StatePairOfKey(t * ParityTable.NTiers + t);
                    }
                }
                var lockS = Mode == Policy.Baseline ? stateLock : null;
                var sprev = StateSwitchCost > 0f ? prevS : null;
                if (!dual && Options != null)
                {
                    // B: the budget the controller leaves covers row costs plus the applied
                    // option's fixed cost, so each option gets it less its own fixed cost, is
                    // solved with its own geometry costs and quality, and is scored net of what
                    // it loses on the scene. The winner (with a little hysteresis, a global
                    // switch being a pop of the whole frame) is solved last so asg holds it.
                    var seen0 = SeenBy(0);
                    int drawn = 0;
                    for (int i = 0; i < N; i++) if (seen0[i]) drawn++;
                    float basis = BudgetMs;
                    if (optCost == null || optCost.Length != rowCost.Length) optCost = new float[rowCost.Length];
                    AllocResult SolveOption(int o)
                    {
                        var op = Options[o];
                        System.Array.Copy(thetaAxis, optTheta, thetaAxis.Length);
                        for (int t = 0; t < ParityTable.NTiers; t++) optTheta[3, t] = t < op.GeoTheta.Length ? op.GeoTheta[t] : 0.0;
                        Cost.RowCosts(Table, optTheta, optCost);
                        Factored.SetCosts(optCost);
                        Factored.SetGeometryQuality(Table, op.GeoQuality);
                        float fixedMs = op.FixedMs + drawn * op.DrawnMs;
                        BudgetMs = Mathf.Max(basis - fixedMs, 0f);
                        var r = Factored.Solve(stateSalM, sal, holds, headroomM, N, BudgetMs, asg, prevs, SwitchCost,
                                               sprev, StateSwitchCost, lockS);
                        r.Cost += fixedMs;
                        return r;
                    }
                    int cur = Mathf.Clamp(Option, 0, Options.Length - 1), best = cur, solved = -1;
                    double bestJ = double.NegativeInfinity, curJ = 0.0;
                    for (int o = 0; o < Options.Length; o++)
                    {
                        if (OptionExternal && o != cur) continue;
                        Last = SolveOption(o); solved = o;
                        double j = Last.Utility - Options[o].SceneLoss;
                        if (o == cur) curJ = j;
                        if (j > bestJ) { bestJ = j; best = o; }
                    }
                    if (best != cur && bestJ <= curJ + OptionHysteresis * System.Math.Abs(curJ)) best = cur;
                    if (best != solved) Last = SolveOption(best);
                    BudgetMs = basis;
                    decidedOption = best;
                }
                else if (!dual)
                    Last = Factored.Solve(stateSalM, sal, holds, headroomM, N, BudgetMs, asg, prevs, SwitchCost,
                                          sprev, StateSwitchCost, lockS);
                else
                {
                    // Rho: 0 when the GPU has slack at Rho = 0, else the ratio at which the GPU
                    // budget is met exactly (the CPU one then is too: the combined constraint is
                    // tight). Bracketed from last frame's value, a few solves per frame; the
                    // search continues next frame from where it stopped.
                    float lo = 0f, hi = float.PositiveInfinity, rho = Rho;
                    for (int it = 0; it < RhoSolves; it++)
                    {
                        for (int r = 0; r < rowCost.Length; r++) rowEff[r] = rowCpu[r] + rho * rowGpu[r];
                        Factored.SetCosts(rowEff);
                        BudgetMs = BudgetCpuMs + rho * BudgetGpuMs;
                        Last = Factored.Solve(stateSalM, sal, holds, headroomM, N, BudgetMs, asg, prevs, SwitchCost,
                                              sprev, StateSwitchCost, lockS);
                        double pc = 0, pg = 0;
                        for (int k = 0; k < V; k++)
                            for (int i = 0; i < N; i++) { pc += rowCpu[asg[k][i]]; pg += rowGpu[asg[k][i]]; }
                        SpendCpuMs = (float)pc; SpendGpuMs = (float)pg;
                        Rho = rho;
                        bool gpuOver = pg > BudgetGpuMs * (1f + RhoTol), cpuOver = pc > BudgetCpuMs * (1f + RhoTol);
                        if (gpuOver) lo = rho;
                        else if (cpuOver && rho > 0f) hi = rho;
                        else break;
                        rho = float.IsInfinity(hi) ? Mathf.Max(2f * rho, 0.05f)
                            : lo <= 0f ? 0.5f * hi : Mathf.Sqrt(lo * hi);
                        if (!gpuOver && lo <= 0f && hi < 1e-3f) rho = 0f;
                    }
                }
                Prof[9] += Factored.TotalTicks * TickMs; Factored.TotalTicks = 0;
                Evals += Last.Evals;
                HoldsReleased += Factored.Released;
            }
            else
            {
                Alloc.SetCosts(Table, rowCost);
                Last = Alloc.Solve(salience, headroom, N, BudgetMs, assign);
            }
            AllocatableMs = Last.Cost;
            if (!dual) { SpendCpuMs = AllocatableMs; SpendGpuMs = 0f; }
            long before = atick;
            ALap(10);
            AllocMs = (float)((atick - before) * TickMs);
            decidedFactored = factored;
            decidedViewers = V;
        }

        /// <summary>Applies the last decision: tiers, and reconciliation of any surrogate the
        /// decision promoted. Runs on the main thread.</summary>
        void ApplyDecision()
        {
            bool factored = decidedFactored;
            if (Options != null)
            {
                Option = decidedOption;
                if (OptionFrames == null || OptionFrames.Length != Options.Length) OptionFrames = new long[Options.Length];
                OptionFrames[Option]++;
            }
            int V = decidedViewers;
            Promotes = 0; Demotes = 0;
            for (int i = 0; i < N; i++)
            {
                int row = factored ? assignV[0][i] : assign[i];
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
                    if (i == Tracked) TrackedRestores++;
                }
                else if (Beh[i] < 3 && nb == 3) Demotes++;
                Beh[i] = nb;
                Nav[i] = (sbyte)Table.TierOf(row, 1);
                Anim[i] = (sbyte)Table.TierOf(row, 2);
                Geo[i] = (sbyte)Table.TierOf(row, 3);
                AnimV[0][i] = Anim[i];
                for (int k = 1; k < V; k++)
                {
                    int rk = assignV[k][i];
                    AnimV[k][i] = (sbyte)Table.TierOf(rk, 2);
                    GeoV[k][i] = (sbyte)Table.TierOf(rk, 3);
                    // the gait is evaluated once, at the most detailed tier any viewer draws
                    if (AnimV[k][i] < Anim[i]) Anim[i] = AnimV[k][i];
                }
            }
        }

        /// <summary>Pop-ledger holds the allocator released to meet the frame budget, cumulative.</summary>
        public int HoldsReleased;
        public long Evals;
        float[] popArea;

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

        /// <summary>Every agent's gait decoded at one animation tier (0: the reference pose an
        /// image judge compares a policy's frame against).</summary>
        public void Pose(int tier)
        {
            for (int i = 0; i < N; i++) Joints(i, JointCount[tier]);
        }

        /// <summary>Each agent's gait decoded at its own tier.</summary>
        public void Pose(sbyte[] tiers)
        {
            for (int i = 0; i < N; i++) Joints(i, JointCount[tiers[i]]);
        }
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
            System.Array.Copy(Pos, PrevPos, N);
            grid.Build(PrevPos, N);
            // Invariant 5: agents are visited in (cell, tier) order, the tier being the one the
            // allocator chose -- neighbours are adjacent in memory and one tier's body runs in
            // a stretch. Every neighbour read is from the start-of-frame snapshot (Jacobi), so the
            // order changes only speed, never the result, and the loop is free to go wide.
            if (visit == null || visit.Length != N) visit = new int[N];
            if (keyStart == null || keyStart.Length != grid.Cells * 4 + 1) keyStart = new int[grid.Cells * 4 + 1];
            var cellOf = grid.CellOf;
            System.Array.Clear(keyStart, 0, keyStart.Length);
            for (int i = 0; i < N; i++) keyStart[cellOf[i] * 4 + Beh[i] + 1]++;
            for (int k = 0; k + 1 < keyStart.Length; k++) keyStart[k + 1] += keyStart[k];
            for (int i = 0; i < N; i++) visit[keyStart[cellOf[i] * 4 + Beh[i]]++] = i;
            bool congested = Spec.Kind != SceneKind.Plaza;
            // Jacobi reads make every agent's update independent, so the loop is split across
            // cores in contiguous stretches of the visit order (each a run of nearby cells) --
            // for both policies alike
            int chunks = Parallelism > 1 && N >= 1024 ? Mathf.Min(Parallelism * 4, N / 256) : 1;
            if (chunks > 1)
                System.Threading.Tasks.Parallel.For(0, chunks, new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Parallelism },
                    c => FineRange((int)((long)c * N / chunks), (int)((long)(c + 1) * N / chunks), congested));
            else FineRange(0, N, congested);
            Scene.Constrain(this);
            Scene.AfterStep(this, frame, ref rng);
            Presentation();
        }

        /// <summary>Worker threads for the fine step (1 = serial).</summary>
        public int Parallelism = System.Environment.ProcessorCount;

        void FineRange(int lo, int hi, bool congested)
        {
            for (int k = lo; k < hi; k++)
            {
                int i = visit[k];
                int beh = Beh[i];
                // context is read from the core's neighbourhood, so it is available at every
                // tier including the surrogate, which never runs a neighbour query of its own.
                // At run time only PARITY's ledger needs it, and only for a surrogate; the
                // baseline's measured divergence is an instrument (see Instrument()).
                if (Mode == Policy.Parity && beh == 3) Ctx[i] = CtxOf(grid.Count(PrevPos, i, 2.2f, 8));
                if (beh == 3)
                {
                    // surrogate: ride the core, no local behaviour at all
                    Pos[i] = CoreP[i];
                }
                // staggered by agent, as UE5 spreads variable tick rates: ticking every tier-1 agent
                // on the same frame would put the whole saving and the whole cost in alternate frames
                else if (Mode == Policy.Baseline ? (frame + i) % BehStride(beh) == 0 : true)
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
                    // someone standing in their queue slot keeps 0.8 m to the next person
                    // rather than being pushed out to the 1.6 m separation radius
                    bool inLine = Slot[i] >= 0 && d < 1.2f;
                    if (budget > 0 && !inLine) sep = grid.Separation(PrevPos, i, 1.6f, budget);
                    dir = new Vector2(dir.x - dir.y * LatBias[i], dir.y + dir.x * LatBias[i]);
                    float adv = Mode == Policy.Baseline ? BehStride(beh) : 1;
                    Vector2 v;
                    if (!congested)
                        v = (dir + sep * 1.4f).normalized * Speed[i] * SpeedScale[i] * adv * Dt;
                    else
                    {
                        // Queues and doorways: the pull and the push are summed, not
                        // normalised, so a pressed crowd slows instead of vibrating at full
                        // walking speed, and an agent eases into its slot instead of
                        // overshooting it every frame.
                        Vector2 u = Vector2.ClampMagnitude(dir * Mathf.Clamp01(d / 0.6f) + sep * 1.4f, 1f);
                        v = u * Speed[i] * SpeedScale[i] * adv * Dt;
                    }
                    Pos[i] += v;
                    Pos[i] = new Vector2(Mathf.Clamp(Pos[i].x, 0f, size.x), Mathf.Clamp(Pos[i].y, 0f, size.y));
                }
                int anim = Anim[i];
                int astride = Mode == Policy.Baseline ? Stride[anim] : 1;
                if (anim < 3 && (frame + i) % astride == 0)
                {
                    Phase[i] += Speed[i] * Dt * astride * 6f * Mathf.Clamp01(Walk[i]);
                    Joints(i, JointCount[anim]);
                }
            }
        }

        void Presentation()
        {
            for (int i = 0; i < N; i++)
            {
                Vector2 m = Pos[i] - PrevPos[i];
                float step = m.magnitude;
                if (step > 2f) continue;    // a respawn or a reconciliation snap, not a stride
                // the baseline moves in k-frame jumps, so walking is judged over time: the
                // average is taken before the clamp, and a 10-frame jump still reads as a walk
                Walk[i] = Mathf.Lerp(Walk[i], step / (Speed[i] * Dt * 0.8f), 0.08f);
                if (step > 1e-4f)
                    Heading[i] = Mathf.LerpAngle(Heading[i] * Mathf.Rad2Deg,
                                                 Mathf.Atan2(m.x, m.y) * Mathf.Rad2Deg, 0.2f) * Mathf.Deg2Rad;
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
            else if (Instrumented) Instrument();
        }

        /// <summary>When false, Step leaves the baseline's measured divergence to a separate
        /// Instrument() call, so a benchmark can keep a measuring device out of the frame it
        /// times: the baseline needs no context at run time, only the figures do.</summary>
        public bool Instrumented = true;

        /// <summary>The baseline's measured divergence for this frame, from the same snapshot
        /// the step read -- identical whether called inside Step or after it.</summary>
        public void Instrument()
        {
            if (Mode != Policy.Baseline) return;
            // Every baseline tier above 0 misses part of the reference process, not just
            // the lowest one -- that is what ticking at 1/3/10 frames costs. Nothing ever
            // resets it: MassLOD has no ledger, so the total only goes up.
            for (int i = 0; i < N; i++)
            {
                float miss = TimeSlice > 0 ? 1f - 1f / TimeSlice : BaselineMiss[Beh[i]];
                if (miss <= 0f) continue;
                Ctx[i] = CtxOf(grid.Count(PrevPos, i, 2.2f, 8));
                Dmeas[i] += ERate[Ctx[i]] * miss;
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

        /// <summary>What this frame's tier mix costs under this world's calibrated cost model.</summary>
        public float PredictedSpendMs() => (float)Cost.Predict(Counts, N);

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

        public void Dispose() { Join(); Alloc?.Dispose(); Ledger?.Dispose(); DisposeNative(); }
    }

    /// <summary>Uniform grid for neighbour separation; the cost of a full query is what the
    /// navigation axis is buying.</summary>
    sealed class Grid
    {
        readonly int nx, ny;
        readonly float cell;
        readonly int[] start;        // agents of cell c are items[start[c] .. start[c + 1])
        int[] items, cellOf;
        public int[] CellOf => cellOf;
        public int Cells => nx * ny;

        public Grid(Vector2 size, float cellSize)
        {
            cell = cellSize;
            nx = Mathf.Max(1, Mathf.CeilToInt(size.x / cell));
            ny = Mathf.Max(1, Mathf.CeilToInt(size.y / cell));
            start = new int[nx * ny + 1];
        }

        int Idx(Vector2 p)
        {
            int cx = Mathf.Clamp((int)(p.x / cell), 0, nx - 1);
            int cy = Mathf.Clamp((int)(p.y / cell), 0, ny - 1);
            return cy * nx + cx;
        }

        // a counting sort by cell: each cell's agents are contiguous, so a query walks memory
        // in order instead of chasing a linked list across it
        public void Build(Vector2[] pos, int n)
        {
            if (items == null || items.Length < n) { items = new int[n]; cellOf = new int[n]; }
            System.Array.Clear(start, 0, start.Length);
            for (int i = 0; i < n; i++) { int c = Idx(pos[i]); cellOf[i] = c; start[c + 1]++; }
            for (int c = 0; c < nx * ny; c++) start[c + 1] += start[c];
            // backwards, so each cell keeps its agents in index order; the scatter leaves
            // start[c + 1] at the BEGINNING of cell c, so shift down one
            for (int i = n - 1; i >= 0; i--) items[--start[cellOf[i] + 1]] = i;
            for (int c = 0; c < nx * ny; c++) start[c] = start[c + 1];
            start[nx * ny] = n;
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
                    int c = y * nx + x, end = start[c + 1];
                    for (int k = start[c]; k < end && seen < cap; k++)
                    {
                        int j = items[k];
                        if (j != i && (pos[i] - pos[j]).sqrMagnitude <= r2) seen++;
                    }
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
                    int c = y * nx + x, end = start[c + 1];
                    for (int k = start[c]; k < end && seen < budget; k++)
                    {
                        int j = items[k];
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
