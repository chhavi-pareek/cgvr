// The three sets, built from primitives at runtime so there is still no scene to author and
// nothing to import. Surfaces are shaded by Parity/Environment's world-space patterns.
//
// One rule governs every placement: nothing solid stands inside the walkable rectangle unless
// the sim's constrain() also knows about it. The corridor's partition and doorway are real
// walls in both. Everything else -- buildings, trees, lamps, the station's roof and its
// gantry -- sits outside the rectangle or overhead, and what is inside it is flush with the
// floor (paving, lane markings, hazard stripes). A crowd that walks through a prop is a
// visual lie about the simulation; this layout makes one impossible.
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Parity
{
    public static class Stage
    {
        static Shader env, sign;
        static Font font;
        static readonly List<Material> signMats = new List<Material>();
        static readonly Dictionary<string, Material> cache = new Dictionary<string, Material>();
        static Transform root;

        public static GameObject Build(SceneSpec spec, out Light sun)
        {
            env = Shader.Find("Parity/Environment");
            sign = Shader.Find("Parity/Sign");
            font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (env == null || sign == null)
                Debug.LogError("Parity/Environment or Parity/Sign not found -- is Assets/Shaders in the project?");
            cache.Clear();
            signMats.Clear();
            Font.textureRebuilt -= Rebind;
            Font.textureRebuilt += Rebind;

            var go = new GameObject("Set: " + spec.Title);
            root = go.transform;
            sun = new GameObject("Sun").AddComponent<Light>();
            sun.transform.SetParent(root, false);
            sun.type = LightType.Directional;
            sun.shadows = LightShadows.Soft;
            sun.shadowStrength = 0.82f;
            sun.shadowBias = 0.04f;
            sun.shadowNormalBias = 0.35f;
            switch (spec.Kind)
            {
                case SceneKind.Hub: Hub(sun); break;
                case SceneKind.Corridor: Corridor(sun); break;
                default: Plaza(sun); break;
            }
            RenderSettings.sun = sun;
            DynamicGI.UpdateEnvironment();
            return go;
        }

        static void Rebind(Font f)
        {
            if (f != font) return;
            foreach (var m in signMats) if (m != null) m.mainTexture = f.material.mainTexture;
        }

        // -- materials -------------------------------------------------------------------

        enum P { Plain = 0, Tiles = 1, Paving = 2, Facade = 3, Hazard = 4, Planting = 5, Asphalt = 6, WallTile = 7, Timber = 8, Glazing = 9 }

        static Vector4 Lin(Color c, float k = 1f) { var l = c.linear; return new Vector4(l.r * k, l.g * k, l.b * k, 1f); }

        static Material Mat(string key, P p, Color a, Color b = default, Color c = default,
                            Vector4 scale = default, float gloss = 0.2f, float metal = 0f,
                            Color emit = default, float emitK = 0f, Color glow = default, float glowK = 0f,
                            Vector4 centre = default, float seed = 0f)
        {
            if (cache.TryGetValue(key, out var hit)) return hit;
            var m = new Material(env) { enableInstancing = true, name = key };
            m.SetFloat("_Pattern", (float)p);
            // SetVector, not SetColor: the values are already linear, and HDR intensities must
            // not go through a second gamma conversion
            m.SetVector("_ColorA", Lin(a));
            m.SetVector("_ColorB", Lin(b == default ? a : b));
            m.SetVector("_ColorC", Lin(c == default ? a * 0.5f : c));
            m.SetVector("_Scale", scale == default ? new Vector4(1f, 1f, 0.03f, 0f) : scale);
            m.SetFloat("_Gloss", gloss);
            m.SetFloat("_Metal", metal);
            m.SetVector("_Emission", Lin(emit, emitK));
            m.SetVector("_Glow", Lin(glow, glowK));
            m.SetVector("_Center", centre);
            m.SetFloat("_Seed", seed);
            cache[key] = m;
            return m;
        }

        static Material Glow(string key, Color c, float k) =>
            Mat(key, P.Plain, c * 0.2f, emit: c, emitK: k, gloss: 0.4f);

        // -- primitives ------------------------------------------------------------------

        static GameObject Prim(PrimitiveType t, Vector3 pos, Vector3 scale, Material m, bool shadows, Quaternion rot)
        {
            var go = GameObject.CreatePrimitive(t);
            Object.DestroyImmediate(go.GetComponent<Collider>());
            go.transform.SetParent(root, false);
            go.transform.localPosition = pos;
            go.transform.localRotation = rot;
            go.transform.localScale = scale;
            var r = go.GetComponent<MeshRenderer>();
            r.sharedMaterial = m;
            r.shadowCastingMode = shadows ? ShadowCastingMode.On : ShadowCastingMode.Off;
            r.receiveShadows = true;
            r.lightProbeUsage = LightProbeUsage.Off;
            r.reflectionProbeUsage = ReflectionProbeUsage.Simple;
            return go;
        }

        /// <summary>Axis-aligned box by min / max corner. Sim (x, y) is Unity (x, z).</summary>
        static GameObject Box(float x0, float y0, float z0, float x1, float y1, float z1, Material m, bool shadows = true)
        {
            return Prim(PrimitiveType.Cube, new Vector3((x0 + x1) * 0.5f, (y0 + y1) * 0.5f, (z0 + z1) * 0.5f),
                        new Vector3(Mathf.Abs(x1 - x0), Mathf.Abs(y1 - y0), Mathf.Abs(z1 - z0)), m, shadows, Quaternion.identity);
        }

        static GameObject Cyl(float x, float y0, float z, float r, float h, Material m, bool shadows = true)
        {
            return Prim(PrimitiveType.Cylinder, new Vector3(x, y0 + h * 0.5f, z), new Vector3(2f * r, h * 0.5f, 2f * r),
                        m, shadows, Quaternion.identity);
        }

        static GameObject Ball(Vector3 c, Vector3 size, Material m, bool shadows = true)
        {
            return Prim(PrimitiveType.Sphere, c, size, m, shadows, Quaternion.identity);
        }

        /// <summary>A square-section member from a to b.</summary>
        static GameObject Beam(Vector3 a, Vector3 b, float t, Material m, bool shadows = true)
        {
            Vector3 d = b - a;
            return Prim(PrimitiveType.Cube, (a + b) * 0.5f, new Vector3(t, t, d.magnitude), m, shadows,
                        Quaternion.LookRotation(d.normalized, Mathf.Abs(d.normalized.y) > 0.99f ? Vector3.forward : Vector3.up));
        }

        /// <summary>3-D text; <paramref name="yaw"/> is the direction the reader looks, so a
        /// sign read by someone facing +x takes 90.</summary>
        static TextMesh Text(string s, Vector3 pos, float yaw, float height, Color col, float glow,
                             TextAnchor anchor = TextAnchor.MiddleCenter, float pitch = 0f)
        {
            var go = new GameObject("Text " + s.Split('\n')[0]);
            go.transform.SetParent(root, false);
            go.transform.localPosition = pos;
            go.transform.localRotation = Quaternion.Euler(pitch, yaw, 0f);
            var tm = go.AddComponent<TextMesh>();
            tm.font = font;
            tm.fontSize = 96;
            tm.characterSize = height * 10f / 96f;   // one line is ~fontSize * characterSize / 10 units
            tm.anchor = anchor;
            tm.alignment = TextAlignment.Left;
            tm.color = Color.white;
            tm.text = s;
            var mr = go.GetComponent<MeshRenderer>();
            var m = new Material(sign) { mainTexture = font.material.mainTexture };
            m.SetVector("_Color", Lin(col));
            m.SetFloat("_Glow", glow);
            signMats.Add(m);
            mr.sharedMaterial = m;
            mr.shadowCastingMode = ShadowCastingMode.Off;
            mr.receiveShadows = false;
            return tm;
        }

        static void Sky(Color tint, float exposure, float atmosphere, Color ground, float sunSize = 0.045f)
        {
            var sh = Shader.Find("Skybox/Procedural");
            if (sh == null) return;
            var m = new Material(sh);
            m.SetColor("_SkyTint", tint);
            m.SetColor("_GroundColor", ground);
            m.SetFloat("_Exposure", exposure);
            m.SetFloat("_AtmosphereThickness", atmosphere);
            m.SetFloat("_SunSize", sunSize);
            m.SetFloat("_SunSizeConvergence", 6f);
            RenderSettings.skybox = m;
            RenderSettings.defaultReflectionMode = DefaultReflectionMode.Skybox;
            RenderSettings.reflectionIntensity = 0.85f;
        }

        static void Ambient(Color sky, Color equator, Color ground)
        {
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = sky;
            RenderSettings.ambientEquatorColor = equator;
            RenderSettings.ambientGroundColor = ground;
        }

        static void Fog(Color c, float density)
        {
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogColor = c;
            RenderSettings.fogDensity = density;
        }

        // -- plaza: 120 x 120 open square in a city block --------------------------------

        static void Plaza(Light sun)
        {
            sun.transform.rotation = Quaternion.Euler(33f, 212f, 0f);
            sun.color = new Color(1.0f, 0.86f, 0.70f);
            sun.intensity = 1.45f;
            Sky(new Color(0.50f, 0.58f, 0.72f), 1.15f, 1.25f, new Color(0.38f, 0.36f, 0.34f));
            Ambient(new Color(0.46f, 0.52f, 0.64f), new Color(0.42f, 0.40f, 0.38f), new Color(0.22f, 0.20f, 0.18f));
            Fog(new Color(0.64f, 0.66f, 0.70f), 0.0042f);

            var paving = Mat("paving", P.Paving, new Color(0.74f, 0.72f, 0.68f), new Color(0.64f, 0.63f, 0.61f),
                             new Color(0.30f, 0.30f, 0.31f), new Vector4(0.9f, 0.45f, 0.035f, 0f), 0.28f,
                             centre: new Vector4(60f, 60f, 0f, 0f));
            var slabs = Mat("slabs", P.Tiles, new Color(0.62f, 0.62f, 0.60f), new Color(0.57f, 0.57f, 0.56f),
                            new Color(0.34f, 0.34f, 0.34f), new Vector4(1.5f, 1.5f, 0.02f, 0f), 0.14f);
            var asphalt = Mat("asphalt", P.Asphalt, new Color(0.16f, 0.16f, 0.17f), gloss: 0.15f);
            var paint = Mat("paint", P.Plain, new Color(0.86f, 0.86f, 0.82f), gloss: 0.3f);
            var yellow = Mat("paintY", P.Plain, new Color(0.92f, 0.72f, 0.12f), gloss: 0.3f);

            Box(0f, -0.2f, 0f, 120f, 0f, 120f, paving, false);
            // inner pavement ring, kerb, road ring, outer pavement
            Ring(-7f, 127f, 0f, 120f, slabs, 0f);
            Ring(-19f, 139f, -7f, 127f, asphalt, -0.15f);
            Ring(-24f, 144f, -19f, 139f, slabs, 0f);
            Box(-500f, -0.6f, -500f, 620f, -0.4f, 620f, asphalt, false);

            // road markings: dashed centre lines and a zebra crossing on each side
            for (float x = -16f; x < 136f; x += 6f)
            {
                Box(x, -0.149f, -13.08f, x + 3f, -0.145f, -12.92f, paint, false);
                Box(x, -0.149f, 132.92f, x + 3f, -0.145f, 133.08f, paint, false);
                Box(-13.08f, -0.149f, x, -12.92f, -0.145f, x + 3f, paint, false);
                Box(132.92f, -0.149f, x, 133.08f, -0.145f, x + 3f, paint, false);
            }
            for (float t = -18.5f; t < -7.5f; t += 1f)
            {
                Box(57f, -0.149f, t, 63f, -0.145f, t + 0.5f, paint, false);
                Box(57f, -0.149f, 120f - t - 0.5f, 63f, -0.145f, 120f - t, paint, false);
                Box(t, -0.149f, 57f, t + 0.5f, -0.145f, 63f, paint, false);
                Box(120f - t - 0.5f, -0.149f, 57f, 120f - t, -0.145f, 63f, paint, false);
            }
            Box(-7.2f, -0.149f, -7.2f, 127.2f, -0.145f, -7.05f, yellow, false);

            // trees and lamps alternate along the inner pavement, benches face the square
            var bark = Mat("bark", P.Plain, new Color(0.25f, 0.18f, 0.13f), gloss: 0.1f);
            var leaf = Mat("leaf", P.Planting, new Color(0.18f, 0.33f, 0.12f), new Color(0.34f, 0.47f, 0.17f), gloss: 0.05f);
            var leaf2 = Mat("leaf2", P.Planting, new Color(0.22f, 0.30f, 0.10f), new Color(0.42f, 0.44f, 0.16f), gloss: 0.05f);
            var grate = Mat("grate", P.Plain, new Color(0.14f, 0.14f, 0.15f), gloss: 0.4f, metal: 0.6f);
            var pole = Mat("pole", P.Plain, new Color(0.10f, 0.16f, 0.13f), gloss: 0.5f, metal: 0.6f);
            var lampHead = Glow("lampHead", new Color(1.0f, 0.80f, 0.52f), 7f);
            var wood = Mat("wood", P.Timber, new Color(0.46f, 0.30f, 0.18f));
            var iron = Mat("iron", P.Plain, new Color(0.12f, 0.12f, 0.13f), gloss: 0.45f, metal: 0.7f);
            var rng = new Rng(4242u);
            for (int side = 0; side < 4; side++)
                for (int k = 0; k < 10; k++)
                {
                    float along = 6f + 12f * k;
                    Vector2 tree = Edge(side, along, -3.5f);
                    Box(tree.x - 0.75f, -0.02f, tree.y - 0.75f, tree.x + 0.75f, 0.005f, tree.y + 0.75f, grate, false);
                    float th = rng.Range(3.0f, 4.2f);
                    Cyl(tree.x, 0f, tree.y, 0.2f, th + 1f, bark);
                    var lm = rng.NextFloat() < 0.5f ? leaf : leaf2;
                    for (int c = 0; c < 3; c++)
                    {
                        float r = rng.Range(1.7f, 2.5f);
                        var off = new Vector3(rng.Range(-1f, 1f), rng.Range(0f, 1.2f), rng.Range(-1f, 1f));
                        Ball(new Vector3(tree.x, th + r * 0.7f, tree.y) + off, new Vector3(2f * r, 1.7f * r, 2f * r), lm);
                    }
                    Vector2 lamp = Edge(side, along + 6f, -3.2f);
                    if (along + 6f < 120f)
                    {
                        Cyl(lamp.x, 0f, lamp.y, 0.07f, 5.2f, pole);
                        Box(lamp.x - 0.22f, 5.1f, lamp.y - 0.22f, lamp.x + 0.22f, 5.45f, lamp.y + 0.22f, lampHead, false);
                        Box(lamp.x - 0.3f, 5.45f, lamp.y - 0.3f, lamp.x + 0.3f, 5.52f, lamp.y + 0.3f, pole);
                    }
                    if (k % 3 == 1) Bench(side, along + 3f, wood, iron);
                }

            // the city block: rows of buildings behind the outer pavement, two cross streets
            Color[] facade =
            {
                new Color(0.80f, 0.72f, 0.58f), new Color(0.70f, 0.46f, 0.36f), new Color(0.86f, 0.83f, 0.76f),
                new Color(0.55f, 0.60f, 0.64f), new Color(0.62f, 0.34f, 0.26f), new Color(0.74f, 0.66f, 0.54f),
                new Color(0.48f, 0.50f, 0.52f), new Color(0.90f, 0.86f, 0.70f),
            };
            var glass = new Color(0.10f, 0.13f, 0.17f);
            var warm = new Color(1.0f, 0.78f, 0.50f);
            var roofTop = new Color(0.30f, 0.29f, 0.28f);
            var kit = Mat("roofkit", P.Plain, new Color(0.55f, 0.56f, 0.57f), gloss: 0.35f, metal: 0.3f);
            int bi = 0;
            for (int side = 0; side < 4; side++)
            {
                float a = -30f;
                while (a < 150f)
                {
                    float w = rng.Range(11f, 24f);
                    if (a + w > 150f) w = 150f - a;
                    bool street = (a < 34f && a + w > 30f) || (a < 94f && a + w > 90f)
                                  || (side == 2 && a < 70f && a + w > 50f);   // the clock tower's plot
                    if (street) { a += w; continue; }
                    float h = rng.Range(11f, 36f);
                    float d = rng.Range(14f, 22f);
                    var col = facade[bi % facade.Length];
                    var m = Mat("facade" + bi, P.Facade, col, col * 0.78f, glass, new Vector4(2.8f, 3.4f, 4.6f, 0f),
                                0.2f, glow: warm, glowK: 1.7f, seed: bi * 7.3f);
                    Building(side, a, a + w, 24.5f, 24.5f + d, h, m);
                    if (rng.NextFloat() < 0.6f)
                    {
                        Vector2 c0 = Edge(side, a + w * 0.3f, -(24.5f + d * 0.5f));
                        Box(c0.x - 1.5f, h, c0.y - 1.5f, c0.x + 1.5f, h + 1.8f, c0.y + 1.5f, kit);
                        Vector2 c1 = Edge(side, a + w * 0.7f, -(24.5f + d * 0.4f));
                        Cyl(c1.x, h, c1.y, 1.1f, 2.4f, kit);
                    }
                    a += w + 0.4f;
                    bi++;
                }
            }
            // landmarks: a clock tower mid-north and a glass tower on the far corner
            var stone = Mat("tower", P.Facade, new Color(0.78f, 0.70f, 0.58f), new Color(0.62f, 0.55f, 0.45f), glass,
                            new Vector4(3.6f, 5.2f, 6f, 0f), 0.2f, glow: warm, glowK: 1.2f, seed: 99f);
            Box(55f, 0f, 146f, 65f, 38f, 156f, stone);
            Box(56f, 38f, 147f, 64f, 47f, 155f, Mat("towerTop", P.Plain, new Color(0.74f, 0.66f, 0.55f)));
            Box(57.5f, 47f, 148.5f, 62.5f, 50f, 153.5f, Mat("towerTop", P.Plain, new Color(0.74f, 0.66f, 0.55f)));
            Cyl(60f, 50f, 151f, 0.35f, 9f, iron);
            var face = Glow("clockFace", new Color(1.0f, 0.95f, 0.82f), 2.2f);
            Prim(PrimitiveType.Cylinder, new Vector3(60f, 42.5f, 146.9f), new Vector3(5f, 0.06f, 5f), face, false,
                 Quaternion.Euler(90f, 0f, 0f));
            Beam(new Vector3(60f, 42.5f, 146.8f), new Vector3(60f, 44.4f, 146.8f), 0.18f, iron, false);
            Beam(new Vector3(60f, 42.5f, 146.78f), new Vector3(61.3f, 42.0f, 146.78f), 0.14f, iron, false);
            var curtain = Mat("curtainTower", P.Glazing, new Color(0.18f, 0.24f, 0.30f), c: new Color(0.20f, 0.21f, 0.22f),
                              scale: new Vector4(1.6f, 3.6f, 0.05f, 0f), glow: new Color(0.55f, 0.65f, 0.75f), glowK: 0.25f);
            Box(146f, 0f, 146f, 168f, 78f, 168f, curtain);
            Box(-50f, 0f, 146f, -26f, 54f, 170f, curtain);
        }

        /// <summary>A rectangular ring [o0, o1]^2 minus [i0, i1]^2 with its top at y.</summary>
        static void Ring(float o0, float o1, float i0, float i1, Material m, float y)
        {
            Box(o0, y - 0.2f, o0, o1, y, i0, m, false);
            Box(o0, y - 0.2f, i1, o1, y, o1, m, false);
            Box(o0, y - 0.2f, i0, i0, y, i1, m, false);
            Box(i1, y - 0.2f, i0, o1, y, i1, m, false);
        }

        /// <summary>Point on side s (0 south, 1 east, 2 north, 3 west) of the square, `along`
        /// metres from its start and `out` metres outward (negative is away from the square).</summary>
        static Vector2 Edge(int side, float along, float outward)
        {
            switch (side)
            {
                case 0: return new Vector2(along, outward);
                case 1: return new Vector2(120f - outward, along);
                case 2: return new Vector2(120f - along, 120f - outward);
                default: return new Vector2(outward, 120f - along);
            }
        }

        static void Building(int side, float a0, float a1, float d0, float d1, float h, Material m)
        {
            Vector2 p = Edge(side, a0, -d0), q = Edge(side, a1, -d1);
            Box(Mathf.Min(p.x, q.x), 0f, Mathf.Min(p.y, q.y), Mathf.Max(p.x, q.x), h, Mathf.Max(p.y, q.y), m);
        }

        static void Bench(int side, float along, Material wood, Material iron)
        {
            Vector2 c = Edge(side, along, -5.6f);
            float yaw = side * -90f;
            var rot = Quaternion.Euler(0f, yaw, 0f);
            var p = new Vector3(c.x, 0f, c.y);
            Prim(PrimitiveType.Cube, p + rot * new Vector3(0f, 0.45f, 0f), new Vector3(1.8f, 0.07f, 0.46f), wood, true, rot);
            Prim(PrimitiveType.Cube, p + rot * new Vector3(0f, 0.72f, -0.22f), new Vector3(1.8f, 0.38f, 0.05f), wood, true, rot);
            Prim(PrimitiveType.Cube, p + rot * new Vector3(-0.8f, 0.22f, 0f), new Vector3(0.06f, 0.44f, 0.44f), iron, true, rot);
            Prim(PrimitiveType.Cube, p + rot * new Vector3(0.8f, 0.22f, 0f), new Vector3(0.06f, 0.44f, 0.44f), iron, true, rot);
        }

        // -- hub: 100 x 40 station concourse, one ticket window at (60, 20) ---------------

        static float Arch(float z) => 12f + 5f * Mathf.Sin(Mathf.PI * Mathf.Clamp01(z / 40f));

        static void Hub(Light sun)
        {
            sun.transform.rotation = Quaternion.Euler(57f, 158f, 0f);
            sun.color = new Color(1.0f, 0.94f, 0.86f);
            sun.intensity = 1.35f;
            Sky(new Color(0.48f, 0.56f, 0.70f), 1.2f, 1.0f, new Color(0.40f, 0.38f, 0.36f));
            Ambient(new Color(0.44f, 0.46f, 0.52f), new Color(0.40f, 0.37f, 0.33f), new Color(0.24f, 0.21f, 0.18f));
            Fog(new Color(0.60f, 0.61f, 0.64f), 0.0065f);

            var floor = Mat("hubFloor", P.Tiles, new Color(0.80f, 0.76f, 0.68f), new Color(0.71f, 0.67f, 0.60f),
                            new Color(0.30f, 0.27f, 0.24f), new Vector4(1.2f, 1.2f, 0.012f, 0f), 0.74f);
            var inlay = Mat("inlay", P.Plain, new Color(0.24f, 0.22f, 0.21f), gloss: 0.8f);
            var plaster = Mat("plaster", P.Plain, new Color(0.83f, 0.79f, 0.71f), gloss: 0.12f);
            var stone = Mat("hubStone", P.Plain, new Color(0.64f, 0.60f, 0.53f), gloss: 0.2f);
            var steel = Mat("steel", P.Plain, new Color(0.30f, 0.33f, 0.36f), gloss: 0.45f, metal: 0.75f);
            var paintY = Mat("laneY", P.Plain, new Color(0.95f, 0.74f, 0.10f), gloss: 0.5f);
            var clerestory = Mat("clerestory", P.Glazing, new Color(0.30f, 0.36f, 0.42f), c: new Color(0.22f, 0.23f, 0.24f),
                                 scale: new Vector4(1.5f, 1.5f, 0.05f, 0f), glow: new Color(0.86f, 0.92f, 1.0f), glowK: 1.6f);
            var exitGlass = Mat("exitGlass", P.Glazing, new Color(0.22f, 0.28f, 0.32f), c: new Color(0.16f, 0.17f, 0.18f),
                                scale: new Vector4(2.0f, 2.6f, 0.035f, 0f), glow: new Color(0.88f, 0.94f, 1.0f), glowK: 1.35f);

            Box(-2f, -0.2f, -2f, 102f, 0f, 42f, floor, false);
            Box(0f, 0f, 0.5f, 100f, 0.003f, 1.1f, inlay, false);
            Box(0f, 0f, 38.9f, 100f, 0.003f, 39.5f, inlay, false);

            // long walls: recessed panels between pilasters, a clerestory in every bay
            foreach (float zSide in new[] { 0f, 40f })
            {
                float s = zSide == 0f ? -1f : 1f;
                float zw0 = zSide + s * 0.4f, zw1 = zSide + s * 1.0f;
                Box(-0.6f, 0f, Mathf.Min(zw0, zw1), 100.6f, 12.4f, Mathf.Max(zw0, zw1), plaster);
                Box(-0.6f, 0f, Mathf.Min(zw0, zw1) - 0.01f, 100.6f, 1.3f, Mathf.Max(zw0, zw1) + 0.01f, stone);
                for (float x = 0f; x <= 100f; x += 10f)
                    Box(x - 0.6f, 0f, Mathf.Min(zSide, zw0), x + 0.6f, 12.4f, Mathf.Max(zSide, zw0), stone);
                for (float x = 0f; x < 100f; x += 10f)
                    Box(x + 2.2f, 4.4f, Mathf.Min(zw0 - s * 0.02f, zw0 + s * 0.06f), x + 7.8f, 10.6f,
                        Mathf.Max(zw0 - s * 0.02f, zw0 + s * 0.06f), clerestory, false);
                Box(-0.8f, 12.4f, Mathf.Min(zSide - s * 0.3f, zw1), 100.8f, 13.0f, Mathf.Max(zSide - s * 0.3f, zw1), stone);
            }

            // entrance arcade at x = 0 (agents re-enter here) and the street seen through it
            float[] piers = { -1f, 3f, 9f, 13f, 19f, 21f, 27f, 31f, 37f, 41f };
            for (int k = 0; k < piers.Length; k += 2)
                Box(-1.0f, 0f, piers[k], -0.3f, 12.4f, piers[k + 1], stone);
            Box(-1.0f, 7.2f, -1f, -0.3f, 12.4f, 41f, plaster);
            Box(-1.1f, 7.0f, -1f, -0.2f, 7.3f, 41f, stone);
            var street = Mat("hubStreet", P.Paving, new Color(0.62f, 0.60f, 0.56f), new Color(0.55f, 0.54f, 0.52f),
                             new Color(0.3f, 0.3f, 0.3f), new Vector4(0.9f, 0.45f, 0.05f, 0f), 0.2f,
                             centre: new Vector4(-500f, -500f, 0f, 0f));
            Box(-40f, -0.2f, -20f, -1f, 0f, 60f, street, false);
            var opp = Mat("hubOpposite", P.Facade, new Color(0.82f, 0.76f, 0.64f), new Color(0.66f, 0.60f, 0.50f),
                          new Color(0.10f, 0.13f, 0.17f), new Vector4(2.8f, 3.4f, 4.6f, 0f), 0.2f,
                          glow: new Color(1.0f, 0.8f, 0.55f), glowK: 1.4f, seed: 5f);
            Box(-34f, 0f, -20f, -22f, 26f, 60f, opp);

            // exit wall: glazed, daylight beyond, doors along its whole length
            Box(100f, 0f, -1f, 100.3f, 12.4f, 41f, exitGlass, false);
            var exitGreen = Glow("exitGreen", new Color(0.12f, 0.85f, 0.35f), 2.4f);
            for (float z = 4f; z < 40f; z += 8f)
            {
                Box(99.85f, 2.9f, z - 0.55f, 99.95f, 3.3f, z + 0.55f, exitGreen, false);
                Text("EXIT", new Vector3(99.8f, 3.1f, z), 90f, 0.28f, Color.white, 1.6f);
            }

            // roof: arched trusses on the pilaster line, purlins between them, open to the sky
            for (float x = 10f; x <= 90f; x += 10f) Truss(x, steel);
            Truss(0.2f, steel); Truss(99.8f, steel);
            for (float z = 0f; z <= 40f; z += 5f)
                Box(0f, Arch(z) + 0.15f, z - 0.1f, 100f, Arch(z) + 0.4f, z + 0.1f, steel);

            // pendant lamps in two rows
            var lampShade = Mat("shade", P.Plain, new Color(0.14f, 0.15f, 0.16f), gloss: 0.5f, metal: 0.6f);
            var bulb = Glow("bulb", new Color(1.0f, 0.84f, 0.58f), 6f);
            foreach (float z in new[] { 12f, 28f })
                for (float x = 15f; x < 100f; x += 10f)
                {
                    Cyl(x, 10.9f, z, 0.02f, Arch(z) - 10.9f, steel, false);
                    Cyl(x, 10.55f, z, 0.45f, 0.35f, lampShade);
                    Ball(new Vector3(x, 10.5f, z), new Vector3(0.5f, 0.35f, 0.5f), bulb, false);
                }

            // queue lane to the ticket window, painted on the floor
            Box(11.6f, 0f, 19.31f, 59.6f, 0.004f, 19.39f, paintY, false);
            Box(11.6f, 0f, 20.61f, 59.6f, 0.004f, 20.69f, paintY, false);
            Box(59.5f, 0f, 19.31f, 59.62f, 0.004f, 20.69f, paintY, false);
            Text("PLEASE QUEUE HERE", new Vector3(54f, 0.006f, 21.4f), 0f, 0.42f, new Color(0.95f, 0.74f, 0.10f), 0.7f,
                 TextAnchor.MiddleCenter, 90f);

            // the gantry over the window: tickets, departure boards and a clock, all overhead
            const float gx = 61.5f;
            Box(gx - 0.2f, 0f, -0.4f, gx + 0.2f, 6.2f, 0f, steel);
            Box(gx - 0.2f, 0f, 40f, gx + 0.2f, 6.2f, 40.4f, steel);
            Box(gx - 0.25f, 5.6f, -0.4f, gx + 0.25f, 6.2f, 40.4f, steel);
            var blue = Glow("ticketBlue", new Color(0.08f, 0.28f, 0.72f), 1.3f);
            Hang(gx, 20f, 5.0f, 1.1f, 3.4f, blue, steel, 5.6f);
            foreach (float side in new[] { -1f, 1f })
                Text("TICKETS", new Vector3(gx + side * 0.1f, 5.0f, 20f), side < 0f ? 90f : 270f, 0.62f, Color.white, 1.8f);
            var board = Mat("board", P.Plain, new Color(0.03f, 0.03f, 0.035f), gloss: 0.6f);
            const string departures =
                "DEPARTURES\n" +
                "10:42  MYSURU         4\n" +
                "10:55  CHENNAI        2\n" +
                "11:05  HUBBALLI       6\n" +
                "11:20  MANGALURU      1\n" +
                "11:35  TIRUPATI       3";
            foreach (float bz in new[] { 8.5f, 31.5f })
            {
                Hang(gx, bz, 4.4f, 2.2f, 7.0f, board, steel, 5.6f);
                foreach (float side in new[] { -1f, 1f })
                {
                    // read facing +x the text runs toward -z, so its left edge is at +z
                    float left = side < 0f ? bz + 3.2f : bz - 3.2f;
                    Text(departures, new Vector3(gx + side * 0.1f, 5.35f, left), side < 0f ? 90f : 270f, 0.28f,
                         new Color(1.0f, 0.66f, 0.12f), 2.2f, TextAnchor.UpperLeft);
                }
            }
            var dial = Glow("dial", new Color(1.0f, 0.97f, 0.9f), 1.6f);
            Prim(PrimitiveType.Cylinder, new Vector3(gx, 7.0f, 20f), new Vector3(1.5f, 0.08f, 1.5f), dial, true,
                 Quaternion.Euler(0f, 0f, 90f));
            Box(gx - 0.05f, 6.2f, 19.97f, gx + 0.05f, 6.3f, 20.03f, steel);
            foreach (float s in new[] { -1f, 1f })
            {
                Beam(new Vector3(gx + s * 0.1f, 7.0f, 20f), new Vector3(gx + s * 0.1f, 7.55f, 20f), 0.05f, steel, false);
                Beam(new Vector3(gx + s * 0.1f, 7.0f, 20f), new Vector3(gx + s * 0.1f, 7.25f, 20.35f), 0.05f, steel, false);
            }

            var green = Glow("platGreen", new Color(0.05f, 0.45f, 0.25f), 1.1f);
            Hang(92f, 20f, 5.2f, 1.0f, 6f, green, steel, Arch(20f));
            Text("PLATFORMS 1-8  >", new Vector3(91.88f, 5.2f, 20f), 90f, 0.42f, Color.white, 1.6f);
            Text("CENTRAL STATION", new Vector3(-0.25f, 9.8f, 20f), 270f, 1.1f, new Color(0.95f, 0.86f, 0.62f), 1.2f);
        }

        /// <summary>A board of height h and width w hung on two rods from rodTop.</summary>
        static void Hang(float x, float z, float yMid, float h, float w, Material face, Material rod, float rodTop)
        {
            Box(x - 0.08f, yMid - h * 0.5f, z - w * 0.5f, x + 0.08f, yMid + h * 0.5f, z + w * 0.5f, face);
            float top = yMid + h * 0.5f;
            foreach (float s in new[] { -0.35f, 0.35f })
                Box(x - 0.02f, top, z + s * w - 0.02f, x + 0.02f, rodTop, z + s * w + 0.02f, rod, false);
        }

        static void Truss(float x, Material steel)
        {
            const int seg = 16;
            for (int k = 0; k < seg; k++)
            {
                float z0 = 40f * k / seg, z1 = 40f * (k + 1) / seg;
                var t0 = new Vector3(x, Arch(z0), z0);
                var t1 = new Vector3(x, Arch(z1), z1);
                var b0 = t0 - new Vector3(0f, 1.3f, 0f);
                var b1 = t1 - new Vector3(0f, 1.3f, 0f);
                Beam(t0, t1, 0.32f, steel);
                Beam(b0, b1, 0.22f, steel);
                Beam(b0, t0, 0.12f, steel);
                Beam(k % 2 == 0 ? b0 : t0, k % 2 == 0 ? t1 : b1, 0.1f, steel);
            }
        }

        // -- corridor: 60 x 10 with a partition at x = 30 and a 1.6 m door --------------

        static void Corridor(Light sun)
        {
            sun.transform.rotation = Quaternion.Euler(64f, 200f, 0f);
            sun.color = new Color(0.90f, 0.93f, 1.0f);
            sun.intensity = 1.2f;
            Sky(new Color(0.40f, 0.46f, 0.62f), 0.85f, 1.5f, new Color(0.22f, 0.22f, 0.24f), 0.03f);
            Ambient(new Color(0.50f, 0.55f, 0.64f), new Color(0.42f, 0.44f, 0.48f), new Color(0.20f, 0.20f, 0.22f));
            Fog(new Color(0.30f, 0.32f, 0.38f), 0.008f);

            var vinyl = Mat("vinyl", P.Tiles, new Color(0.58f, 0.61f, 0.64f), new Color(0.53f, 0.56f, 0.60f),
                            new Color(0.31f, 0.33f, 0.36f), new Vector4(0.6f, 0.6f, 0.02f, 0f), 0.48f);
            var tiles = Mat("wallTile", P.WallTile, new Color(0.90f, 0.90f, 0.87f), new Color(0.16f, 0.34f, 0.37f),
                            new Color(0.52f, 0.53f, 0.52f), new Vector4(0.3f, 0.15f, 0.05f, 0f), 0.6f,
                            centre: new Vector4(0f, 0f, 0f, 1.1f));
            var cap = Mat("wallCap", P.Plain, new Color(0.46f, 0.48f, 0.50f), gloss: 0.3f);
            var metal = Mat("doorMetal", P.Plain, new Color(0.22f, 0.24f, 0.26f), gloss: 0.55f, metal: 0.7f);
            var hazard = Mat("hazard", P.Hazard, new Color(0.95f, 0.74f, 0.08f), c: new Color(0.06f, 0.06f, 0.06f),
                             scale: new Vector4(0.7f, 1f, 0f, 0f), gloss: 0.4f);
            var lumi = Glow("lumi", new Color(0.35f, 1.0f, 0.55f), 1.3f);
            var cove = Glow("cove", new Color(1.0f, 0.95f, 0.85f), 2.4f);
            var exitGreen = Glow("exitGreenC", new Color(0.10f, 0.80f, 0.32f), 2.6f);
            var beacon = Glow("beacon", new Color(1.0f, 0.22f, 0.10f), 5f);
            var daylight = Glow("daylight", new Color(0.92f, 0.96f, 1.0f), 2.2f);
            var roof = Mat("roofSlab", P.Asphalt, new Color(0.40f, 0.40f, 0.41f), gloss: 0.1f);
            var paint = Mat("arrowPaint", P.Plain, new Color(0.84f, 0.84f, 0.80f), gloss: 0.4f);
            var red = Mat("extRed", P.Plain, new Color(0.75f, 0.08f, 0.06f), gloss: 0.6f);
            const float H = 3.2f;

            Box(-1f, -0.2f, -0.5f, 67f, 0f, 10.5f, vinyl, false);
            Box(0f, 0f, 0.1f, 60f, 0.004f, 0.22f, lumi, false);
            Box(0f, 0f, 9.78f, 60f, 0.004f, 9.9f, lumi, false);
            Box(27.9f, 0f, 3.1f, 29.7f, 0.004f, 6.9f, hazard, false);
            Box(30.3f, 0f, 3.6f, 31.4f, 0.004f, 6.4f, hazard, false);
            foreach (float x in new[] { 6f, 14f, 22f, 38f, 46f, 54f })
            {
                Prim(PrimitiveType.Cube, new Vector3(x, 0.003f, 5.35f), new Vector3(1.1f, 0.004f, 0.22f), paint, false,
                     Quaternion.Euler(0f, 35f, 0f));
                Prim(PrimitiveType.Cube, new Vector3(x, 0.003f, 4.65f), new Vector3(1.1f, 0.004f, 0.22f), paint, false,
                     Quaternion.Euler(0f, -35f, 0f));
            }

            // long walls with a lit cove along their tops, and the building's roof beyond
            Box(-0.4f, 0f, -0.4f, 60.4f, H, 0f, tiles);
            Box(-0.4f, 0f, 10f, 60.4f, H, 10.4f, tiles);
            Box(-0.5f, H, -0.5f, 60.5f, H + 0.12f, 0.02f, cap);
            Box(-0.5f, H, 9.98f, 60.5f, H + 0.12f, 10.5f, cap);
            Box(-0.4f, H + 0.12f, -0.05f, 60.4f, H + 0.16f, 0.02f, cove, false);
            Box(-0.4f, H + 0.12f, 9.98f, 60.4f, H + 0.16f, 10.05f, cove, false);
            Box(-30f, H - 0.3f, -30f, 90f, H + 0.05f, -0.5f, roof);
            Box(-30f, H - 0.3f, 10.5f, 90f, H + 0.05f, 40f, roof);
            Box(-30f, H - 0.3f, -0.5f, -0.5f, H + 0.05f, 10.5f, roof);
            Box(66f, H - 0.3f, -0.5f, 90f, H + 0.05f, 10.5f, roof);
            var kit = Mat("hvac", P.Plain, new Color(0.56f, 0.58f, 0.60f), gloss: 0.35f, metal: 0.4f);
            var rng = new Rng(77u);
            for (int k = 0; k < 14; k++)
            {
                float x = rng.Range(-20f, 80f);
                float z = rng.NextFloat() < 0.5f ? rng.Range(-18f, -3f) : rng.Range(13f, 28f);
                float w = rng.Range(1.5f, 4f), d = rng.Range(1.5f, 3f);
                Box(x, H + 0.05f, z, x + w, H + rng.Range(0.8f, 2.2f), z + d, kit);
            }
            Box(-500f, -0.6f, -500f, 560f, -0.4f, 560f, roof, false);

            // entry end: the stair door the crowd arrives through
            Box(-0.4f, 0f, -0.4f, 0f, H, 10.4f, tiles);
            Box(-0.02f, 0f, 3.4f, 0.04f, 2.3f, 6.6f, metal);
            Box(0.04f, 0.9f, 4.95f, 0.07f, 1.0f, 5.05f, cap, false);

            // the partition and its one door: a real wall in the sim as well as on screen
            Box(CorridorScene.DoorX0, 0f, -0.4f, CorridorScene.DoorX1, H, CorridorScene.Gap0, tiles);
            Box(CorridorScene.DoorX0, 0f, CorridorScene.Gap1, CorridorScene.DoorX1, H, 10.4f, tiles);
            Box(CorridorScene.DoorX0, 2.2f, CorridorScene.Gap0, CorridorScene.DoorX1, H, CorridorScene.Gap1, tiles);
            Box(CorridorScene.DoorX0 - 0.06f, 0f, CorridorScene.Gap0 - 0.08f, CorridorScene.DoorX1 + 0.06f, 2.26f,
                CorridorScene.Gap0, metal);
            Box(CorridorScene.DoorX0 - 0.06f, 0f, CorridorScene.Gap1, CorridorScene.DoorX1 + 0.06f, 2.26f,
                CorridorScene.Gap1 + 0.08f, metal);
            Box(CorridorScene.DoorX0 - 0.06f, 2.2f, CorridorScene.Gap0 - 0.08f, CorridorScene.DoorX1 + 0.06f, 2.3f,
                CorridorScene.Gap1 + 0.08f, metal);
            Box(CorridorScene.DoorX0 - 0.08f, 2.9f, 4.95f, CorridorScene.DoorX1 + 0.08f, 3.05f, 5.05f, cap, false);
            Ball(new Vector3(30f, 3.28f, 5f), new Vector3(0.28f, 0.2f, 0.28f), beacon, false);
            foreach (float side in new[] { -1f, 1f })
            {
                // one sign on each face of the partition, read from the queue and from beyond
                float sx = side < 0f ? CorridorScene.DoorX0 - 0.04f : CorridorScene.DoorX1;
                float fx = side < 0f ? CorridorScene.DoorX0 - 0.06f : CorridorScene.DoorX1 + 0.06f;
                Box(sx, 2.42f, 4.35f, sx + 0.04f, 2.82f, 5.65f, exitGreen, false);
                Text("EXIT", new Vector3(fx, 2.62f, 5f), side < 0f ? 90f : 270f, 0.26f, Color.white, 1.8f);
            }

            // exit end: the opening to daylight
            Box(60f, 0f, -0.4f, 60.4f, H, 2.5f, tiles);
            Box(60f, 0f, 7.5f, 60.4f, H, 10.4f, tiles);
            Box(60f, 2.6f, 2.5f, 60.4f, H, 7.5f, tiles);
            Box(66f, -0.5f, -6f, 66.4f, 7f, 16f, daylight, false);
            Box(59.94f, 2.66f, 4.2f, 59.98f, 3.08f, 5.8f, exitGreen, false);
            Text("EXIT", new Vector3(59.9f, 2.87f, 5f), 90f, 0.26f, Color.white, 1.8f);

            // wall furniture: extinguishers and way-finding signs
            foreach (float x in new[] { 9f, 24f, 36f, 51f })
            {
                Cyl(x, 0.55f, 0.05f, 0.09f, 0.52f, red);
                Box(x - 0.06f, 1.4f, 0.0f, x + 0.06f, 1.52f, 0.05f, red, false);
                Box(x + 1.5f, 1.9f, 9.96f, x + 3.3f, 2.25f, 10f, exitGreen, false);
                Text("FIRE EXIT  >", new Vector3(x + 2.4f, 2.075f, 9.94f), 0f, 0.2f, Color.white, 1.6f);
            }
        }
    }
}
