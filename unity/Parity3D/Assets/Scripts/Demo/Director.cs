// Builds the whole demo at runtime, so there is no scene to author and nothing to wire in
// the Inspector: open the project, press Play. Two crowds from one seed under one camera,
// left MassLOD baseline, right PARITY, both rendered by the same code so the only difference
// on screen is the fidelity policy.
using UnityEngine;

namespace Parity
{
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

        public static readonly Vector2 SceneSize = new Vector2(120f, 120f);   // sim/scenes.py Plaza
        static readonly Vector2 Focus = new Vector2(60f, 60f);                // bench/camerapaths.py FOCUS

        public int Agents = 1200;
        public float BudgetMs = 6.0f;
        public float Cap = 4.0f;
        public bool Orbit = true;
        public float OrbitSeconds = 40f;
        public float CamHeight = 14f;
        public bool ColourByDivergence = true;
        public bool Paused;

        // OnGUI runs more than once per frame, so the HUD only ever *requests* a size and
        // the rebuild happens once, here, between frames.
        public int PendingAgents = -1;

        public ParityTable Table;
        public CrowdWorld Base, Par;
        public Camera CamL, CamR;
        CrowdRenderer rendL, rendR;
        float orbitT;
        float lastBaseMs, lastParMs;

        public float[] TraceBase = new float[600];
        public float[] TracePar = new float[600];
        public int TraceHead;

        void Awake()
        {
            Table = ParityTable.Build();
            BuildStage();
            Rebuild(Agents);
        }

        public void Rebuild(int n)
        {
            Agents = Mathf.Clamp(n, 50, 30000);
            Base?.Dispose(); Par?.Dispose();
            Base = new CrowdWorld(Policy.Baseline, Agents, SceneSize, 1u, Table, BudgetMs, Cap);
            Par = new CrowdWorld(Policy.Parity, Agents, SceneSize, 1u, Table, BudgetMs, Cap);
            System.Array.Clear(TraceBase, 0, TraceBase.Length);
            System.Array.Clear(TracePar, 0, TracePar.Length);
            TraceHead = 0;
        }

        void BuildStage()
        {
            var sun = new GameObject("Sun").AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.transform.rotation = Quaternion.Euler(52f, 40f, 0f);
            sun.intensity = 1.05f;
            sun.shadows = LightShadows.None;
            RenderSettings.ambientLight = new Color(0.36f, 0.38f, 0.44f);

            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.position = new Vector3(SceneSize.x * 0.5f, 0f, SceneSize.y * 0.5f);
            ground.transform.localScale = new Vector3(SceneSize.x / 10f, 1f, SceneSize.y / 10f);
            Destroy(ground.GetComponent<Collider>());
            var gm = ground.GetComponent<MeshRenderer>().material;
            gm.color = new Color(0.16f, 0.17f, 0.20f);

            CamL = MakeCamera("Cam Baseline", new Rect(0f, 0f, 0.5f, 1f));
            CamR = MakeCamera("Cam PARITY", new Rect(0.5f, 0f, 0.5f, 1f));

            var shader = Shader.Find("Parity/CrowdInstanced");
            if (shader == null) Debug.LogError("Parity/CrowdInstanced shader not found -- is Assets/Shaders in the project?");
            rendL = new CrowdRenderer(CamL, shader);
            rendR = new CrowdRenderer(CamR, shader);
        }

        static Camera MakeCamera(string name, Rect rect)
        {
            var go = new GameObject(name);
            var c = go.AddComponent<Camera>();
            c.rect = rect;
            c.fieldOfView = 60f;
            c.farClipPlane = 400f;
            c.clearFlags = CameraClearFlags.SolidColor;
            c.backgroundColor = new Color(0.07f, 0.08f, 0.10f);
            c.depth = 0;
            return c;
        }

        void Update()
        {
            if (PendingAgents > 0 && PendingAgents != Agents) { Rebuild(PendingAgents); PendingAgents = -1; }
            if (Paused) { Render(); return; }
            float dt = Mathf.Min(Time.deltaTime, 0.05f);
            if (Orbit) orbitT += dt / Mathf.Max(OrbitSeconds, 1f);

            // bench/camerapaths.py "orbit": radius 0.45 * min(w, h) about the focus point
            float r = 0.45f * Mathf.Min(SceneSize.x, SceneSize.y);
            float ang = orbitT * Mathf.PI * 2f;
            var camXZ = new Vector2(Focus.x + r * Mathf.Cos(ang), Focus.y + r * Mathf.Sin(ang));
            var eye = new Vector3(camXZ.x, CamHeight, camXZ.y);
            var look = new Vector3(Focus.x, 1.2f, Focus.y);
            CamL.transform.position = eye; CamL.transform.LookAt(look);
            CamR.transform.position = eye; CamR.transform.LookAt(look);

            // the LOD camera is the render camera: what drives the tiers is what you see
            float yaw = Mathf.Atan2(look.z - camXZ.y, look.x - camXZ.x);

            Base.BudgetMs = BudgetMs; Base.Cap = Cap;
            Par.BudgetMs = BudgetMs; Par.Cap = Cap;

            Base.Step(camXZ, yaw, lastBaseMs);
            Par.Step(camXZ, yaw, lastParMs);
            lastBaseMs = Base.StepMs;
            lastParMs = Par.StepMs;

            TraceBase[TraceHead] = Base.MaxDivergence();
            TracePar[TraceHead] = Par.MaxDivergence();
            TraceHead = (TraceHead + 1) % TraceBase.Length;

            Render();
        }

        void Render()
        {
            rendL.ColourByDivergence = ColourByDivergence; rendL.Cap = Cap;
            rendR.ColourByDivergence = ColourByDivergence; rendR.Cap = Cap;
            rendL.Draw(Base, Vector3.zero);
            rendR.Draw(Par, Vector3.zero);
        }

        public float TraceCap() { return Cap; }

        void OnDestroy() { Base?.Dispose(); Par?.Dispose(); }
    }
}
