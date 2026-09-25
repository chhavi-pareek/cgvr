// Draws one crowd. The point of this file is that the fidelity tiers are *visible*, and that
// what each one costs is real:
//
//   geometry 0  articulated figure from smooth capsules, hair, some with a backpack  ~2.4k tris
//   geometry 1  the same figure from coarse capsules, no hair or bag                  ~0.6k tris
//   geometry 2  one body and a head, no limbs                                           ~100 tris
//   geometry 3  a camera-facing card with the figure cut out in the shader                 2 tris
//
// The animation axis freezes the limbs, and the behaviour axis is what stops the agent
// steering at all. Divergence shows as a glowing ring on the ground (natural look) or as the
// body colour (divergence look), so the baseline's crowd lights up and stays lit while
// PARITY's flares and is cleared by restoration.
//
// Director.CalibrateGeometry times Draw + render + GPU sync per geometry tier, so the
// allocator pays for mesh detail in measured milliseconds rather than getting it for free.
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Parity
{
    public enum Look { Natural, Divergence, Tiers }

    public sealed class CrowdRenderer
    {
        // Unity's instanced property arrays are capped; 511 is the safe batch everywhere.
        const int Batch = 511;

        enum S { TorsoHi, LegHi, ArmHi, HeadHi, Box, TorsoLo, LegLo, ArmLo, HeadLo, BodyMin, HeadMin, Impostor, Decal }

        sealed class Stream
        {
            public readonly Mesh Mesh;
            public readonly Material Mat;
            public readonly bool Casts, Impostor;
            public readonly Matrix4x4[] M = new Matrix4x4[Batch];
            public readonly Vector4[] C = new Vector4[Batch], C2, C3;
            public readonly float[] H = new float[Batch];
            public readonly MaterialPropertyBlock Mpb = new MaterialPropertyBlock();
            public int N;

            public Stream(Mesh mesh, Material mat, bool casts, bool impostor)
            {
                Mesh = mesh; Mat = mat; Casts = casts; Impostor = impostor;
                if (impostor) { C2 = new Vector4[Batch]; C3 = new Vector4[Batch]; }
            }
        }

        // figure layout, metres, agent space (y up, facing +z)
        const float HipY = 0.90f, HipX = 0.095f, LegLen = 0.88f;
        const float ShoulderY = 1.45f, ShoulderX = 0.215f, ArmLen = 0.64f;
        static readonly Vector3 TorsoC = new Vector3(0f, 1.20f, 0f), TorsoS = new Vector3(1f, 1f, 0.68f);
        static readonly Vector3 HeadC = new Vector3(0f, 1.635f, 0.01f);
        static readonly Vector3 HairC = new Vector3(0f, 1.66f, -0.018f), HairS = new Vector3(1.07f, 0.82f, 1.06f);
        static readonly Vector3 BagC = new Vector3(0f, 1.22f, -0.17f), BagS = new Vector3(0.30f, 0.40f, 0.15f);
        static readonly Vector3 BodyC = new Vector3(0f, 0.80f, 0f), BodyS = new Vector3(1f, 1f, 0.72f);
        const float CardW = 0.64f, CardH = 1.78f;

        readonly Stream[] streams;
        public bool Shadows = true;
        public Look Look = Look.Natural;
        public float Cap = 4f;

        // appearance, cached per agent index so agent i is the same person in both crowds
        Vector4[] shirt, trousers, skin, hair;
        float[] height;
        bool[] bag;

        public CrowdRenderer(Shader lit, Shader impostor, Shader decal)
        {
            var mLit = new Material(lit) { enableInstancing = true };
            var mImp = new Material(impostor) { enableInstancing = true };
            var mDec = new Material(decal) { enableInstancing = true };
            streams = new[]
            {
                new Stream(Meshes.Capsule("TorsoHi", 0.17f, 0.66f, 16, 4), mLit, true, false),
                new Stream(Meshes.Capsule("LegHi", 0.072f, LegLen, 12, 3), mLit, true, false),
                new Stream(Meshes.Capsule("ArmHi", 0.052f, ArmLen, 10, 3), mLit, true, false),
                new Stream(Meshes.Capsule("HeadHi", 0.115f, 0.23f, 16, 6), mLit, true, false),
                new Stream(Meshes.Box(), mLit, true, false),
                new Stream(Meshes.Capsule("TorsoLo", 0.17f, 0.66f, 8, 2), mLit, true, false),
                new Stream(Meshes.Capsule("LegLo", 0.072f, LegLen, 6, 1), mLit, true, false),
                new Stream(Meshes.Capsule("ArmLo", 0.052f, ArmLen, 5, 1), mLit, true, false),
                new Stream(Meshes.Capsule("HeadLo", 0.115f, 0.23f, 8, 3), mLit, true, false),
                new Stream(Meshes.Capsule("BodyMin", 0.21f, 1.56f, 6, 1), mLit, true, false),
                new Stream(Meshes.Capsule("HeadMin", 0.12f, 0.24f, 6, 2), mLit, true, false),
                new Stream(Meshes.Card(CardW, CardH), mImp, true, true),
                new Stream(Meshes.Decal(), mDec, false, false),
            };
        }

        // -- appearance ----------------------------------------------------------------

        static readonly Color[] Shirts =
        {
            new Color(0.13f, 0.18f, 0.33f), new Color(0.46f, 0.12f, 0.14f), new Color(0.36f, 0.38f, 0.20f),
            new Color(0.80f, 0.60f, 0.20f), new Color(0.86f, 0.86f, 0.83f), new Color(0.55f, 0.68f, 0.82f),
            new Color(0.45f, 0.45f, 0.47f), new Color(0.10f, 0.10f, 0.11f), new Color(0.12f, 0.42f, 0.45f),
            new Color(0.74f, 0.16f, 0.14f), new Color(0.20f, 0.45f, 0.28f), new Color(0.84f, 0.52f, 0.56f),
            new Color(0.87f, 0.43f, 0.15f), new Color(0.73f, 0.63f, 0.49f), new Color(0.36f, 0.25f, 0.50f),
            new Color(0.30f, 0.55f, 0.80f),
        };
        static readonly Color[] Trousers =
        {
            new Color(0.20f, 0.27f, 0.40f), new Color(0.12f, 0.15f, 0.24f), new Color(0.08f, 0.08f, 0.09f),
            new Color(0.60f, 0.53f, 0.38f), new Color(0.35f, 0.35f, 0.37f), new Color(0.35f, 0.25f, 0.18f),
            new Color(0.10f, 0.12f, 0.22f), new Color(0.30f, 0.32f, 0.20f),
        };
        static readonly Color[] Skins =
        {
            new Color(0.34f, 0.21f, 0.14f), new Color(0.52f, 0.34f, 0.23f),
            new Color(0.70f, 0.50f, 0.37f), new Color(0.88f, 0.70f, 0.56f),
        };
        static readonly Color[] Hairs =
        {
            new Color(0.05f, 0.04f, 0.035f), new Color(0.05f, 0.04f, 0.035f), new Color(0.16f, 0.10f, 0.06f),
            new Color(0.30f, 0.20f, 0.12f), new Color(0.55f, 0.55f, 0.55f), new Color(0.70f, 0.58f, 0.36f),
        };

        static uint Mix(uint x)
        {
            x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
            return x;
        }

        void EnsureAppearance(int n)
        {
            if (shirt != null && shirt.Length >= n) return;
            shirt = new Vector4[n]; trousers = new Vector4[n]; skin = new Vector4[n]; hair = new Vector4[n];
            height = new float[n]; bag = new bool[n];
            for (int i = 0; i < n; i++)
            {
                uint h = Mix((uint)i * 2654435761u + 17u);
                float jitter = 0.9f + 0.2f * ((Mix(h) & 0xffff) / 65535f);
                shirt[i] = (Vector4)(Shirts[h % (uint)Shirts.Length] * jitter).linear;
                trousers[i] = (Vector4)Trousers[(h >> 5) % (uint)Trousers.Length].linear;
                float t = ((h >> 9) & 0xff) / 255f * (Skins.Length - 1);
                int k = Mathf.Min((int)t, Skins.Length - 2);
                skin[i] = (Vector4)Color.Lerp(Skins[k], Skins[k + 1], t - k).linear;
                hair[i] = (Vector4)Hairs[(h >> 17) % (uint)Hairs.Length].linear;
                height[i] = 0.92f + 0.16f * (((h >> 21) & 0xff) / 255f);
                bag[i] = ((h >> 29) & 3u) == 0u;
            }
        }

        static readonly Color[] TierTint =
        {
            new Color(0.42f, 0.74f, 1.00f),   // tier 0
            new Color(0.38f, 0.85f, 0.62f),   // tier 1
            new Color(0.95f, 0.82f, 0.35f),   // tier 2
            new Color(0.70f, 0.70f, 0.78f),   // tier 3 surrogate
        };
        static readonly Color Hot = new Color(1f, 0.16f, 0.08f);

        /// <summary>0..1 on a log scale against the cap: the whole story is how far up this goes
        /// and whether it ever comes back down.</summary>
        float Heat(CrowdWorld w, int i)
        {
            float d = Mathf.Max(w.Dmeas[i], 1e-3f);
            float t = Mathf.Clamp01(Mathf.Log10(d / (Cap * 0.05f)) / Mathf.Log10(40f));
            return t * t;
        }

        // -- drawing -------------------------------------------------------------------

        /// <summary>Draws the crowd as viewer `viewer` of the world sees it: its own mesh and gait
        /// tiers (several viewers share one behaviour but not one LOD).</summary>
        public void Draw(CrowdWorld w, Camera cam, int viewer = 0)
        {
            EnsureAppearance(w.N);
            var geoT = w.GeoV[viewer];
            var animT = w.AnimV[viewer];
            Vector3 eye = cam.transform.position;
            var sTorso = streams[(int)S.TorsoHi];
            for (int i = 0; i < w.N; i++)
            {
                Vector3 p = new Vector3(w.Pos[i].x, 0f, w.Pos[i].y);
                float heat = Heat(w, i);
                Vector4 cShirt = shirt[i], cLegs = trousers[i], cSkin = skin[i], cHair = hair[i];
                float glow = 0f;
                if (Look == Look.Natural) glow = 0.12f * heat;
                else
                {
                    Color tint = TierTint[w.Beh[i]];
                    if (Look == Look.Divergence) tint = Color.Lerp(tint, Hot, heat);
                    cShirt = cLegs = (Vector4)tint.linear;
                }

                int g = geoT[i];
                float k = height[i];
                if (g == 3)
                {
                    float yaw = Mathf.Atan2(eye.x - p.x, eye.z - p.z);
                    var s = streams[(int)S.Impostor];
                    ref var m = ref s.M[s.N];
                    Fill(ref m, p, Mathf.Cos(yaw), Mathf.Sin(yaw), k, Vector3.zero,
                         Vector3.right, Vector3.up, Vector3.forward);
                    s.C[s.N] = cShirt; s.C2[s.N] = cLegs; s.C3[s.N] = cSkin; s.H[s.N] = glow;
                    if (++s.N == Batch) Flush(s, cam);
                }
                else
                {
                    float walk = Mathf.Clamp01(w.Walk[i]);
                    bool animated = animT[i] < 3;
                    // the decoded gait, not a raw sine: a lower animation tier blends fewer
                    // joints, so the swing coarsens before it freezes at tier 3
                    float swing = animated ? w.JointBlend[i] * walk : 0f;
                    float bob = animated ? Mathf.Abs(Mathf.Cos(w.Phase[i])) * 0.035f * walk : 0f;
                    Vector3 pb = p + new Vector3(0f, bob, 0f);
                    float cy = Mathf.Cos(w.Heading[i]), sy = Mathf.Sin(w.Heading[i]);

                    if (g == 2)
                    {
                        Part(S.BodyMin, pb, cy, sy, k, BodyC, BodyS, cShirt, glow, cam);
                        Part(S.HeadMin, pb, cy, sy, k, HeadC, Vector3.one, cSkin, glow, cam);
                    }
                    else
                    {
                        bool hi = g == 0;
                        Part(hi ? S.TorsoHi : S.TorsoLo, pb, cy, sy, k, TorsoC, TorsoS, cShirt, glow, cam);
                        Part(hi ? S.HeadHi : S.HeadLo, pb, cy, sy, k, HeadC, Vector3.one, cSkin, glow, cam);
                        Limb(hi ? S.LegHi : S.LegLo, pb, cy, sy, k, -HipX, HipY, LegLen, 30f * swing, cLegs, glow, cam);
                        Limb(hi ? S.LegHi : S.LegLo, pb, cy, sy, k, HipX, HipY, LegLen, -30f * swing, cLegs, glow, cam);
                        Limb(hi ? S.ArmHi : S.ArmLo, pb, cy, sy, k, -ShoulderX, ShoulderY, ArmLen, -22f * swing, cShirt, glow, cam);
                        Limb(hi ? S.ArmHi : S.ArmLo, pb, cy, sy, k, ShoulderX, ShoulderY, ArmLen, 22f * swing, cShirt, glow, cam);
                        if (hi)
                        {
                            Part(S.HeadHi, pb, cy, sy, k, HairC, HairS, cHair, glow, cam);
                            if (bag[i]) Part(S.Box, pb, cy, sy, k, BagC, BagS, cHair * 0.6f + cLegs * 0.4f, glow, cam);
                        }
                    }
                }

                // ground decal: contact shadow always, divergence ring in the natural look
                var dcl = streams[(int)S.Decal];
                ref var dm = ref dcl.M[dcl.N];
                Fill(ref dm, p + new Vector3(0f, 0.012f, 0f), 1f, 0f, 1.3f, Vector3.zero,
                     Vector3.right, Vector3.up, Vector3.forward);
                dcl.C[dcl.N] = (Vector4)Color.Lerp(new Color(1f, 0.62f, 0.12f), Hot, heat).linear;
                dcl.H[dcl.N] = Look == Look.Natural ? Mathf.Clamp01(heat * 1.3f) : 0f;
                if (++dcl.N == Batch) Flush(dcl, cam);
            }
            for (int s = 0; s < streams.Length; s++) Flush(streams[s], cam);
        }

        void Part(S which, Vector3 p, float c, float s, float k, Vector3 centre, Vector3 scale,
                  Vector4 col, float glow, Camera cam)
        {
            var st = streams[(int)which];
            Fill(ref st.M[st.N], p, c, s, k, centre,
                 new Vector3(scale.x, 0f, 0f), new Vector3(0f, scale.y, 0f), new Vector3(0f, 0f, scale.z));
            st.C[st.N] = col; st.H[st.N] = glow;
            if (++st.N == Batch) Flush(st, cam);
        }

        /// <summary>A limb swung by <paramref name="deg"/> about its hip or shoulder pivot.</summary>
        void Limb(S which, Vector3 p, float c, float s, float k, float px, float py, float len,
                  float deg, Vector4 col, float glow, Camera cam)
        {
            float a = deg * Mathf.Deg2Rad, ca = Mathf.Cos(a), sa = Mathf.Sin(a);
            var centre = new Vector3(px, py - 0.5f * len * ca, -0.5f * len * sa);
            var st = streams[(int)which];
            Fill(ref st.M[st.N], p, c, s, k, centre,
                 new Vector3(1f, 0f, 0f), new Vector3(0f, ca, sa), new Vector3(0f, -sa, ca));
            st.C[st.N] = col; st.H[st.N] = glow;
            if (++st.N == Batch) Flush(st, cam);
        }

        /// <summary>world = T(p) * Ry(yaw) * k * [c0 c1 c2 | centre], written in place. Doing this
        /// by hand rather than through Matrix4x4.TRS keeps the per-part cost to a few dozen
        /// multiplies, which matters: it is the CPU half of what geometry costs.</summary>
        static void Fill(ref Matrix4x4 m, Vector3 p, float c, float s, float k, Vector3 centre,
                         Vector3 c0, Vector3 c1, Vector3 c2)
        {
            m.m00 = k * (c * c0.x + s * c0.z); m.m10 = k * c0.y; m.m20 = k * (c * c0.z - s * c0.x); m.m30 = 0f;
            m.m01 = k * (c * c1.x + s * c1.z); m.m11 = k * c1.y; m.m21 = k * (c * c1.z - s * c1.x); m.m31 = 0f;
            m.m02 = k * (c * c2.x + s * c2.z); m.m12 = k * c2.y; m.m22 = k * (c * c2.z - s * c2.x); m.m32 = 0f;
            m.m03 = p.x + k * (c * centre.x + s * centre.z);
            m.m13 = p.y + k * centre.y;
            m.m23 = p.z + k * (c * centre.z - s * centre.x);
            m.m33 = 1f;
        }

        void Flush(Stream s, Camera cam)
        {
            if (s.N == 0) return;
            s.Mpb.Clear();
            s.Mpb.SetVectorArray("_BaseColor", s.C);
            s.Mpb.SetFloatArray("_Heat", s.H);
            if (s.Impostor)
            {
                s.Mpb.SetVectorArray("_Color2", s.C2);
                s.Mpb.SetVectorArray("_Color3", s.C3);
            }
            var rp = new RenderParams(s.Mat)
            {
                camera = cam,
                matProps = s.Mpb,
                receiveShadows = true,
                shadowCastingMode = s.Casts && Shadows ? ShadowCastingMode.On : ShadowCastingMode.Off,
                worldBounds = new Bounds(Vector3.zero, Vector3.one * 5000f),
            };
            Graphics.RenderMeshInstanced(rp, s.Mesh, 0, s.M, s.N);
            s.N = 0;
        }

        // -- measuring what geometry costs -------------------------------------------------

        /// <summary>Wall time per agent to pack, submit, render and wait for the GPU, with every
        /// agent pinned at one geometry tier. Everything that is not geometry -- the set, the
        /// decals, the post pass -- is identical across tiers and cancels in the difference the
        /// caller takes against tier 3. Measured, never authored: invariant 1.</summary>
        public double[] MeasureGeometry(CrowdWorld w, Camera cam, int rounds = 10)
        {
            var best = new double[ParityTable.NTiers];
            for (int g = 0; g < best.Length; g++) best[g] = double.MaxValue;
            var probe = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            var keepGeo = (sbyte[])w.Geo.Clone();
            var keepAnim = (sbyte[])w.AnimV[0].Clone();
            for (int i = 0; i < w.N; i++) w.AnimV[0][i] = 0;
            // Round-robin over the tiers rather than one tier at a time: the GPU clocks up
            // while this runs, and measuring tier by tier hands whichever tier went first the
            // slow clock. Interleaved, any drift lands on every tier alike, and the minimum per
            // tier rejects the scheduler. The first rounds compile shader variants and warm up.
            const int warm = 3;
            for (int r = 0; r < warm + rounds; r++)
                for (int g = 0; g < best.Length; g++)
                {
                    for (int i = 0; i < w.N; i++) w.Geo[i] = (sbyte)g;
                    float t0 = Time.realtimeSinceStartup;
                    RenderOnce(w, cam, probe);
                    double dt = (Time.realtimeSinceStartup - t0) * 1000.0;
                    if (r >= warm && dt < best[g]) best[g] = dt;
                }
            for (int g = 0; g < best.Length; g++) best[g] /= w.N;
            System.Array.Copy(keepGeo, w.Geo, w.N);
            System.Array.Copy(keepAnim, w.AnimV[0], w.N);
            Object.DestroyImmediate(probe);
            return best;
        }

        /// <summary>The pixel judge: what each geometry tier costs in image quality, measured
        /// rather than authored. A row of figures stands `dist` metres in front of `cam`, at
        /// varied headings and mid-stride; each tier is rendered and compared pixel by pixel with
        /// the full-detail render, and the difference is normalised by the difference that
        /// removing the figures altogether makes. So quality 1 is indistinguishable from tier 0
        /// and 0 is as wrong as drawing nobody. The set, the sky and the contact shadows are the
        /// same in every render and cancel.</summary>
        public double[] MeasureGeometryQuality(CrowdWorld w, Camera cam)
        {
            var rt = cam.targetTexture;
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
            var keepLook = Look;
            Look = Look.Natural;
            for (int i = 0; i < w.N; i++)
            {
                w.AnimV[0][i] = 0; w.Walk[i] = 1f; w.Dmeas[i] = 0f;
                w.JointBlend[i] = Mathf.Sin(1.7f * i);
                w.Phase[i] = 0.9f * i;
            }
            Color32[] Grab(bool draw)
            {
                if (draw) Draw(w, cam);
                cam.Render();
                var prev = RenderTexture.active;
                RenderTexture.active = rt;
                tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0, false);
                tex.Apply(false);
                RenderTexture.active = prev;
                return tex.GetPixels32();
            }
            Grab(true);                                   // warm: shaders, shadow maps
            var empty = Grab(false);
            var img = new Color32[ParityTable.NTiers][];
            for (int g = 0; g < img.Length; g++)
            {
                for (int i = 0; i < w.N; i++) w.Geo[i] = (sbyte)g;
                img[g] = Grab(true);
            }
            double absent = Diff(empty, img[0]);
            var q = new double[img.Length];
            for (int g = 0; g < img.Length; g++)
                q[g] = absent > 0.0 ? System.Math.Max(0.0, 1.0 - Diff(img[g], img[0]) / absent) : 1.0;
            Look = keepLook;
            Object.DestroyImmediate(tex);
            return q;
        }

        static double Diff(Color32[] a, Color32[] b)
        {
            long s = 0;
            for (int i = 0; i < a.Length; i++)
                s += System.Math.Abs(a[i].r - b[i].r) + System.Math.Abs(a[i].g - b[i].g) + System.Math.Abs(a[i].b - b[i].b);
            return s;
        }

        /// <summary>Lays a calibration world out as the judge's row of figures.</summary>
        public static void JudgeLayout(CrowdWorld w, Vector2 at, Vector2 dir)
        {
            var side = new Vector2(dir.y, -dir.x);
            for (int i = 0; i < w.N; i++)
            {
                int row = i / 6, col = i % 6;
                w.Pos[i] = at + dir * (2.2f * row) + side * (1.3f * (col - 2.5f) + 0.6f * row);
                w.Heading[i] = i * 0.9f;
            }
        }

        void RenderOnce(CrowdWorld w, Camera cam, Texture2D probe)
        {
            Draw(w, cam);
            cam.Render();
            var prev = RenderTexture.active;
            RenderTexture.active = cam.targetTexture;
            probe.ReadPixels(new Rect(0, 0, 1, 1), 0, 0, false);   // blocks until the GPU is done
            RenderTexture.active = prev;
        }
    }

    static class Meshes
    {
        /// <summary>Capsule along y, centred, radius r, total height h (h = 2r is a sphere).</summary>
        public static Mesh Capsule(string name, float r, float h, int seg, int ring)
        {
            var v = new List<Vector3>();
            var n = new List<Vector3>();
            var t = new List<int>();
            float c = Mathf.Max(h * 0.5f - r, 0f);
            int rows = 0;
            for (int half = 0; half < 2; half++)
                for (int j = 0; j <= ring; j++)
                {
                    float phi = half == 0 ? -Mathf.PI * 0.5f * (1f - j / (float)ring) : Mathf.PI * 0.5f * j / ring;
                    float y = (half == 0 ? -c : c) + r * Mathf.Sin(phi);
                    float rr = r * Mathf.Cos(phi);
                    for (int k = 0; k <= seg; k++)
                    {
                        float th = 2f * Mathf.PI * k / seg;
                        float cx = Mathf.Cos(th), cz = Mathf.Sin(th);
                        v.Add(new Vector3(cx * rr, y, cz * rr));
                        n.Add(new Vector3(cx * Mathf.Cos(phi), Mathf.Sin(phi), cz * Mathf.Cos(phi)));
                    }
                    rows++;
                }
            int cols = seg + 1;
            for (int j = 0; j < rows - 1; j++)
                for (int k = 0; k < seg; k++)
                {
                    int a = j * cols + k, b = a + 1, d = a + cols, e = d + 1;
                    t.Add(a); t.Add(d); t.Add(b);
                    t.Add(b); t.Add(d); t.Add(e);
                }
            var m = new Mesh { name = name };
            m.SetVertices(v); m.SetNormals(n); m.SetTriangles(t, 0);
            m.RecalculateBounds();
            return m;
        }

        public static Mesh Box()
        {
            var m = new Mesh { name = "ParityBox" };
            Vector3[] v =
            {
                new Vector3(-.5f,-.5f,-.5f), new Vector3(.5f,-.5f,-.5f), new Vector3(.5f,.5f,-.5f), new Vector3(-.5f,.5f,-.5f),
                new Vector3(-.5f,-.5f, .5f), new Vector3(.5f,-.5f, .5f), new Vector3(.5f,.5f, .5f), new Vector3(-.5f,.5f, .5f),
            };
            int[][] faces =
            {
                new[]{0,3,2,1}, new[]{4,5,6,7}, new[]{0,1,5,4},
                new[]{2,3,7,6}, new[]{1,2,6,5}, new[]{0,4,7,3},
            };
            Vector3[] fn = { Vector3.back, Vector3.forward, Vector3.down, Vector3.up, Vector3.right, Vector3.left };
            var verts = new List<Vector3>(24);
            var norms = new List<Vector3>(24);
            var tris = new List<int>(36);
            for (int f = 0; f < 6; f++)
            {
                int b = verts.Count;
                for (int k = 0; k < 4; k++) { verts.Add(v[faces[f][k]]); norms.Add(fn[f]); }
                tris.Add(b); tris.Add(b + 1); tris.Add(b + 2);
                tris.Add(b); tris.Add(b + 2); tris.Add(b + 3);
            }
            m.SetVertices(verts); m.SetNormals(norms); m.SetTriangles(tris, 0);
            m.RecalculateBounds();
            return m;
        }

        /// <summary>Upright card, object xy in metres (the impostor shader reads them as uv).</summary>
        public static Mesh Card(float w, float h)
        {
            var m = new Mesh { name = "ImpostorCard" };
            m.SetVertices(new List<Vector3>
            {
                new Vector3(-w * 0.5f, 0f, 0f), new Vector3(w * 0.5f, 0f, 0f),
                new Vector3(w * 0.5f, h, 0f), new Vector3(-w * 0.5f, h, 0f),
            });
            m.SetNormals(new List<Vector3> { Vector3.forward, Vector3.forward, Vector3.forward, Vector3.forward });
            m.SetTriangles(new[] { 0, 2, 1, 0, 3, 2 }, 0);
            m.RecalculateBounds();
            return m;
        }

        /// <summary>Unit ground quad in xz with 0..1 uv.</summary>
        public static Mesh Decal()
        {
            var m = new Mesh { name = "Decal" };
            m.SetVertices(new List<Vector3>
            {
                new Vector3(-0.5f, 0f, -0.5f), new Vector3(0.5f, 0f, -0.5f),
                new Vector3(0.5f, 0f, 0.5f), new Vector3(-0.5f, 0f, 0.5f),
            });
            m.SetUVs(0, new List<Vector2> { new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1), new Vector2(0, 1) });
            m.SetNormals(new List<Vector3> { Vector3.up, Vector3.up, Vector3.up, Vector3.up });
            m.SetTriangles(new[] { 0, 2, 1, 0, 3, 2 }, 0);
            m.RecalculateBounds();
            return m;
        }
    }
}
