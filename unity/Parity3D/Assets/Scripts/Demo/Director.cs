// Builds the whole demo at runtime, so there is no scene to author and nothing to wire in
// the Inspector: open the project, press Play. Two crowds from one seed under one camera,
// left MassLOD baseline, right PARITY, both rendered by the same code in the same set, so the
// only difference on screen is the fidelity policy.
//
// Three sets, one per scene of sim/scenes.py: the open plaza, the station concourse with its
// ticket queue, and the evacuation corridor with its one door. Each has its own measured
// rates, its own cap and its own calibrated costs, so switching set re-derives all of them.
using UnityEngine;
using UnityEngine.Rendering;

namespace Parity
{
    /// <summary>Matched: PARITY is held to exactly what MassLOD spends this frame, each priced
    /// by its own measured costs -- the like-for-like comparison, and the default. Fraction and
    /// absolute are sim/tiered.py's two budget forms, kept for exploring.</summary>
    public enum BudgetMode { Matched, Fraction, Absolute }

    public sealed class Director : MonoBehaviour
    {
        public static Director Instance;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Boot()
        {
            if (Instance != null) return;
            var go = new GameObject("PARITY Director");
            DontDestroyOnLoad(go);
            Instance = go.AddComponent<Director>();
            go.AddComponent<Hud>();
            go.AddComponent<BenchmarkRunner>();
        }

        public SceneKind Kind = SceneKind.Plaza;
        public SceneSpec Spec => SceneSpec.Get(Kind);

        public int Agents = 1200;
        public BudgetMode Budget = BudgetMode.Matched;
        public float BudgetFrac = 0.25f;   // sim/tiered.py's budget_frac
        public float TargetMs = 16.7f;
        public float Cap = (float)(300.0 * ParityTable.PlazaESur);   // 3.969, as sim/tiered.py builds it
        public bool Orbit = true;
        public float OrbitSeconds = 40f;
        public Look Look = Look.Natural;
        public bool Shadows = true;
        /// <summary>Two saliences (alloc/factored.py): behaviour by significance, meshes and
        /// gait by projected size. Off falls back to the single-salience allocator.</summary>
        public bool ViewAware = true;
        /// <summary>Both panels show the ONE PARITY crowd, from two cameras: one behaviour per
        /// agent, each viewer's own mesh detail, one budget (alloc/factored.py, V = 2).</summary>
        public bool CoOp;
        public bool PopLedger = true;
        public bool SwitchCost = true;
        public bool Occlusion = true;
        public bool Post = true;
        public bool Paused;

        // OnGUI runs more than once per frame, so the HUD only ever *requests* a change and
        // the rebuild happens once, here, between frames.
        public int PendingAgents = -1;
        public int PendingScene = -1;
        public bool PendingRestage, PendingCoOp;

        public ParityTable Table;
        public CrowdWorld Base, Par;
        public Camera CamL, CamR;
        public GameObject SetRoot;
        public Light Sun;
        CrowdRenderer rendL, rendR;
        float orbitT;
        float lastBaseMs, lastParMs;
        public float RenderBaseMs, RenderParMs;

        public float[] TraceBase = new float[600];
        public float[] TracePar = new float[600];
        public float[] TraceOne = new float[600];   // one followed PARITY agent
        public int TraceHead;

        public ParityTable AllocTable;
        public double CalibFloor, BaseFloor;
        /// <summary>MassLOD's own measured per-axis costs: its tiers are frame strides, not
        /// latent widths, so the same tier number costs it something different.</summary>
        public double[,] BaseTheta;
        /// <summary>The baseline's predicted spend this frame, which is PARITY's budget when matched.</summary>
        public float BaseSpendMs;
        public int PrunedRows;
        /// <summary>Measured ms per agent to draw each geometry tier, for the HUD.</summary>
        public double[] GeoMs;
        /// <summary>Measured image quality of each geometry tier (the pixel judge), replacing the
        /// table's placeholder column.</summary>
        public double[] GeoQ;

        // Recompiling while in Play mode triggers a domain reload. UnityEngine.Object refs (the
        // cameras, the set, the sun) are re-serialised and survive; plain C# objects -- the
        // table, both crowds, the renderers -- and every static field do not. Everything below
        // is therefore written to be re-entrant, and Update calls it, so a hot reload heals
        // instead of throwing a NullReferenceException every frame.
        void OnEnable() { Instance = this; }

        void Awake()
        {
            Instance = this;
            Agents = Spec.DefaultAgents;
            Cap = Spec.Cap;
            EnsureBuilt();
        }

        void EnsureBuilt()
        {
            ApplyQuality(Shadows);
            if (SetRoot == null) SetRoot = Stage.Build(Spec, out Sun);
            if (CamL == null || CamR == null)
            {
                CamL = MakeCamera("Cam Baseline", new Rect(0f, 0f, 0.5f, 1f));
                CamR = MakeCamera("Cam PARITY", new Rect(0.5f, 0f, 0.5f, 1f));
            }
            if (rendL == null || rendR == null) MakeRenderers();
            if (Table == null || AllocTable == null || CrowdWorld.CalibratedTheta == null || BaseTheta == null)
                Calibrate();
            if (Base == null || Par == null) Rebuild(Agents);
        }

        /// <summary>Measured once per set, not authored (invariant 1): the sim axes by timing the
        /// crowd step, geometry by timing the renderer. Then the rows nothing would ever pick
        /// are dropped so the allocator's inner loop is 5-6x shorter.</summary>
        void Calibrate()
        {
            Table = ParityTable.Build(Spec.EMax);
            CrowdWorld.CalibrateAxes(Spec, Table, 400, out CalibFloor);
            BaseTheta = CrowdWorld.MeasureAxes(Spec, Table, 400, Policy.Baseline, out BaseFloor);
            GeoMs = null; GeoQ = null;
            if (SystemInfo.graphicsDeviceType != GraphicsDeviceType.Null && rendL != null) CalibrateGeometry();
            // the table the allocator scores with carries the measured geometry column; the
            // timings above do not depend on quality, so they stay valid
            if (GeoQ != null) Table = ParityTable.Build(Spec.EMax, GeoQ);
            // the two-salience allocator needs the full product table; pruning on the combined
            // quality would drop rows it uses, and its hulls prune each half themselves
            var pruned = CrowdWorld.PricedAndPruned(Table, CalibFloor, CrowdWorld.CalibratedTheta, out PrunedRows);
            AllocTable = ViewAware ? Table : pruned;
        }

        void CalibrateGeometry()
        {
            // Without this, the sim-only calibration prices geometry at zero -- the step never
            // reads it -- every coarser mesh is dominated, and PARITY is handed full-detail
            // figures for free while the baseline pays for its LOD by distance. That is not a
            // comparison. So the renderer is timed, GPU included, at each tier.
            var cam = MakeCamera("Calibration", new Rect(0f, 0f, 1f, 1f));
            cam.enabled = false;
            DestroyImmediate(cam.GetComponent<PostFx>());
            var rt = new RenderTexture(Mathf.Max(Screen.width / 2, 480), Mathf.Max(Screen.height, 360), 24,
                                       RenderTextureFormat.DefaultHDR);
            rt.antiAliasing = Mathf.Max(QualitySettings.antiAliasing, 1);
            cam.targetTexture = rt;
            Place(cam, 0f, out _, out _);
            var w = new CrowdWorld(Policy.Parity, 400, Spec, 99u, Table);
            rendL.Shadows = Shadows;
            GeoMs = rendL.MeasureGeometry(w, cam);
            // one renderer draws both crowds, so geometry costs both policies the same
            for (int t = 0; t < ParityTable.NTiers - 1; t++)
            {
                double g = System.Math.Max(GeoMs[t] - GeoMs[ParityTable.NTiers - 1], 0.0);
                CrowdWorld.CalibratedTheta[ParityTable.NAxes - 1, t] = g;
                BaseTheta[ParityTable.NAxes - 1, t] = g;
            }
            w.Dispose();

            // and what each tier costs in image quality, judged at the view-salience distance
            var jw = new CrowdWorld(Policy.Parity, 12, Spec, 5u, Table);
            GeoQ = MeasureQuality(rendL, jw, cam, Spec);
            jw.Dispose();

            cam.targetTexture = null;
            Destroy(rt);
            Destroy(cam.gameObject);
        }

        /// <summary>The pixel judge, from a standing eye height at the view-salience distance.</summary>
        public static double[] MeasureQuality(CrowdRenderer r, CrowdWorld jw, Camera cam, SceneSpec s)
        {
            CrowdRenderer.JudgeLayout(jw, s.JudgeAt, s.JudgeDir);
            var eye = s.JudgeAt - s.JudgeDir * jw.ViewD0;
            cam.transform.position = new Vector3(eye.x, 1.6f, eye.y);
            cam.transform.LookAt(new Vector3(s.JudgeAt.x, 0.9f, s.JudgeAt.y) + new Vector3(s.JudgeDir.x, 0f, s.JudgeDir.y));
            return r.MeasureGeometryQuality(jw, cam);
        }

        public void Rebuild(int n)
        {
            Agents = Mathf.Clamp(n, 50, 30000);
            Base?.Dispose(); Par?.Dispose();
            Base = new CrowdWorld(Policy.Baseline, Agents, Spec, 1u, AllocTable, TargetMs, Cap);
            Par = new CrowdWorld(Policy.Parity, Agents, Spec, 1u, AllocTable, TargetMs, Cap);
            Base.ApplyCalibration(BaseFloor, BaseTheta);
            Par.ApplyCalibration(CalibFloor, CrowdWorld.CalibratedTheta);
            Par.ViewAware = ViewAware;
            Par.Viewers = CoOp ? 2 : 1;
            System.Array.Clear(TraceBase, 0, TraceBase.Length);
            System.Array.Clear(TracePar, 0, TracePar.Length);
            System.Array.Clear(TraceOne, 0, TraceOne.Length);
            Par.Tracked = Agents / 2;
            TraceHead = 0;
        }

        void SwitchScene(SceneKind k)
        {
            Kind = k;
            if (SetRoot != null) Destroy(SetRoot);
            SetRoot = null;
            Table = null; AllocTable = null;
            Base?.Dispose(); Par?.Dispose();
            Base = null; Par = null;
            Agents = Spec.DefaultAgents;
            Cap = Spec.Cap;
            orbitT = 0f;
            EnsureBuilt();
        }

        void MakeRenderers()
        {
            var lit = Shader.Find("Parity/CrowdInstanced");
            var imp = Shader.Find("Parity/CrowdImpostor");
            var dec = Shader.Find("Parity/CrowdDecal");
            if (lit == null || imp == null || dec == null)
            {
                Debug.LogError("Parity crowd shaders not found -- is Assets/Shaders in the project?");
                return;
            }
            rendL = new CrowdRenderer(lit, imp, dec);
            rendR = new CrowdRenderer(lit, imp, dec);
        }

        public static void ApplyQuality(bool shadows)
        {
            QualitySettings.shadows = shadows ? ShadowQuality.All : ShadowQuality.Disable;
            QualitySettings.shadowResolution = ShadowResolution.VeryHigh;
            QualitySettings.shadowCascades = 4;
            QualitySettings.shadowDistance = 110f;
            QualitySettings.shadowProjection = ShadowProjection.StableFit;
            QualitySettings.antiAliasing = 4;
            QualitySettings.pixelLightCount = 4;
            QualitySettings.anisotropicFiltering = AnisotropicFiltering.ForceEnable;
        }

        public static Camera MakeCamera(string name, Rect rect)
        {
            var go = new GameObject(name);
            var c = go.AddComponent<Camera>();
            c.rect = rect;
            c.fieldOfView = 55f;
            c.nearClipPlane = 0.3f;
            c.farClipPlane = 900f;
            c.clearFlags = CameraClearFlags.Skybox;
            c.allowHDR = true;
            c.allowMSAA = true;
            c.depth = 0;
            go.AddComponent<PostFx>();
            return c;
        }

        /// <summary>bench/camerapaths.py "orbit": radius 0.45 * min(w, h) about the focus point.
        /// Height is presentation only; the LOD camera is (x, y, yaw), exactly as in Python.</summary>
        void Place(Camera c, float t, out Vector2 camXZ, out float yaw)
        {
            var s = Spec;
            float r = 0.45f * Mathf.Min(s.Size.x, s.Size.y);
            float ang = t * Mathf.PI * 2f;
            camXZ = new Vector2(s.Focus.x + r * Mathf.Cos(ang), s.Focus.y + r * Mathf.Sin(ang));
            var look = new Vector3(s.Focus.x, 1.2f, s.Focus.y);
            c.transform.position = new Vector3(camXZ.x, s.CamHeight, camXZ.y);
            c.transform.LookAt(look);
            yaw = Mathf.Atan2(look.z - camXZ.y, look.x - camXZ.x);
        }

        void Update()
        {
            if (PendingScene >= 0) { var k = (SceneKind)PendingScene; PendingScene = -1; SwitchScene(k); }
            if (PendingCoOp)
            {
                PendingCoOp = false;
                CoOp = !CoOp;
                Rebuild(Agents);
            }
            if (PendingRestage)
            {
                // shadows change what geometry costs, so it is measured again
                PendingRestage = false;
                Table = null; AllocTable = null;
                Base?.Dispose(); Par?.Dispose(); Base = null; Par = null;
                if (Sun != null) Sun.shadows = Shadows ? LightShadows.Soft : LightShadows.None;
            }
            EnsureBuilt();
            if (PendingAgents > 0 && PendingAgents != Agents) { Rebuild(PendingAgents); PendingAgents = -1; }
            PostFx.On = Post;
            if (Paused) { Render(); return; }
            float dt = Mathf.Min(Time.deltaTime, 0.05f);
            if (Orbit) orbitT += dt / Mathf.Max(OrbitSeconds, 1f);

            Place(CamL, orbitT, out var camXZ, out float yaw);
            // co-op: the second viewer walks the same orbit half a turn behind
            Place(CamR, CoOp ? orbitT + 0.5f : orbitT, out var camXZ2, out float yaw2);

            foreach (var w in new[] { Base, Par })
            {
                w.BudgetFrac = BudgetFrac; w.AbsoluteBudget = Budget == BudgetMode.Absolute;
                w.TargetMs = TargetMs; w.Cap = Cap;
            }
            Par.PopLedger = PopLedger;
            Par.SwitchCost = SwitchCost ? 0.12f : 0f;
            Par.Occlusion = Occlusion;
            Par.Viewers = CoOp ? 2 : 1;
            Par.Cam2 = camXZ2; Par.Yaw2 = yaw2;
            SetView(Par, 0, CoOp ? CamL : CamR);
            if (CoOp) SetView(Par, 1, CamR);

            // the LOD camera is the render camera: what drives the tiers is what you see
            if (!CoOp)
            {
                Base.Step(camXZ, yaw, lastBaseMs);
                BaseSpendMs = Base.PredictedSpendMs();
            }
            // two viewers have no MassLOD counterpart on screen to match, so co-op runs on the
            // fraction budget sized for two views
            Par.MatchBudgetMs = Budget == BudgetMode.Matched && !CoOp ? BaseSpendMs : -1f;
            Par.Step(camXZ, yaw, lastParMs);
            lastBaseMs = Base.StepMs + RenderBaseMs;
            lastParMs = Par.StepMs + RenderParMs;

            TraceBase[TraceHead] = Base.MaxDivergence();
            TracePar[TraceHead] = Par.MaxDivergence();
            TraceOne[TraceHead] = Par.TrackedD;
            TraceHead = (TraceHead + 1) % TraceBase.Length;

            Render();
        }

        static void SetView(CrowdWorld w, int k, Camera c)
        {
            var p = c.projectionMatrix;
            w.SetView(k, p * c.worldToCameraMatrix, p.m00, p.m11);
        }

        void Render()
        {
            if (rendL == null || rendR == null || Base == null || Par == null) return;
            foreach (var r in new[] { rendL, rendR }) { r.Look = Look; r.Cap = Cap; r.Shadows = Shadows; }
            float t = Time.realtimeSinceStartup;
            if (CoOp) rendL.Draw(Par, CamL, 0); else rendL.Draw(Base, CamL);
            RenderBaseMs = (Time.realtimeSinceStartup - t) * 1000f;
            t = Time.realtimeSinceStartup;
            rendR.Draw(Par, CamR, CoOp ? 1 : 0);
            RenderParMs = (Time.realtimeSinceStartup - t) * 1000f;
        }

        public float TraceCap() { return Cap; }

        void OnDestroy() { Base?.Dispose(); Par?.Dispose(); }
    }
}
