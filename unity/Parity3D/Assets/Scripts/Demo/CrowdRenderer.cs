// Draws one crowd. The point of this file is that the fidelity tiers are *visible*: the
// geometry axis removes body parts, the animation axis freezes the gait, and the behaviour
// axis is what stops the agent steering at all. Colour carries accumulated divergence, so
// the baseline reddens and stays red while PARITY reddens and snaps back.
using System.Collections.Generic;
using UnityEngine;

namespace Parity
{
    public sealed class CrowdRenderer
    {
        // Unity's instanced property arrays are capped; 511 is the safe batch everywhere.
        const int Batch = 511;

        // one entry per (geometry tier, body part)
        struct Part { public Vector3 Offset, Scale; public int Kind; }  // Kind: 0 torso 1 head 2 leg 3 arm
        static readonly Part[][] Parts =
        {
            new[] {  // geo 0 high: torso, head, two legs, two arms
                new Part { Offset = new Vector3(0, 1.05f, 0), Scale = new Vector3(0.42f, 0.55f, 0.26f), Kind = 0 },
                new Part { Offset = new Vector3(0, 1.58f, 0), Scale = new Vector3(0.25f, 0.28f, 0.25f), Kind = 1 },
                new Part { Offset = new Vector3(-0.13f, 0.40f, 0), Scale = new Vector3(0.16f, 0.45f, 0.16f), Kind = 2 },
                new Part { Offset = new Vector3( 0.13f, 0.40f, 0), Scale = new Vector3(0.16f, 0.45f, 0.16f), Kind = 2 },
                new Part { Offset = new Vector3(-0.30f, 1.08f, 0), Scale = new Vector3(0.12f, 0.42f, 0.12f), Kind = 3 },
                new Part { Offset = new Vector3( 0.30f, 1.08f, 0), Scale = new Vector3(0.12f, 0.42f, 0.12f), Kind = 3 },
            },
            new[] {  // geo 1 mid: torso, head, two legs
                new Part { Offset = new Vector3(0, 1.05f, 0), Scale = new Vector3(0.44f, 0.58f, 0.28f), Kind = 0 },
                new Part { Offset = new Vector3(0, 1.58f, 0), Scale = new Vector3(0.26f, 0.28f, 0.26f), Kind = 1 },
                new Part { Offset = new Vector3(-0.13f, 0.40f, 0), Scale = new Vector3(0.17f, 0.45f, 0.17f), Kind = 2 },
                new Part { Offset = new Vector3( 0.13f, 0.40f, 0), Scale = new Vector3(0.17f, 0.45f, 0.17f), Kind = 2 },
            },
            new[] {  // geo 2 low: one blocky body, no limbs
                new Part { Offset = new Vector3(0, 0.85f, 0), Scale = new Vector3(0.46f, 1.7f, 0.30f), Kind = 0 },
            },
            new[] {  // geo 3 impostor: a flat card
                new Part { Offset = new Vector3(0, 0.85f, 0), Scale = new Vector3(0.50f, 1.7f, 0.04f), Kind = 0 },
            },
        };

        readonly Mesh box;
        readonly Material mat;
        readonly Camera cam;
        readonly MaterialPropertyBlock mpb = new MaterialPropertyBlock();
        readonly Matrix4x4[] mats = new Matrix4x4[Batch];
        readonly Vector4[] cols = new Vector4[Batch];
        int pending;

        public bool ColourByDivergence = true;
        public float Cap = 4f;

        static readonly Color[] TierTint =
        {
            new Color(0.42f, 0.74f, 1.00f),   // tier 0
            new Color(0.38f, 0.85f, 0.62f),   // tier 1
            new Color(0.95f, 0.82f, 0.35f),   // tier 2
            new Color(0.70f, 0.70f, 0.78f),   // tier 3 surrogate
        };

        public CrowdRenderer(Camera camera, Shader shader)
        {
            cam = camera;
            box = BuildBox();
            mat = new Material(shader) { enableInstancing = true };
        }

        static Mesh BuildBox()
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

        Color Tint(CrowdWorld w, int i)
        {
            Color baseCol = TierTint[w.Beh[i]];
            if (!ColourByDivergence) return baseCol;
            // log scale against the cap: the whole story is how far up this goes and whether
            // it ever comes back down
            float d = Mathf.Max(w.Dmeas[i], 1e-3f);
            float t = Mathf.Clamp01(Mathf.Log10(d / (Cap * 0.05f)) / Mathf.Log10(40f));
            return Color.Lerp(baseCol, new Color(1f, 0.16f, 0.08f), t * t);
        }

        public void Draw(CrowdWorld w, Vector3 origin)
        {
            pending = 0;
            for (int i = 0; i < w.N; i++)
            {
                var parts = Parts[w.Geo[i]];
                Vector3 p = origin + new Vector3(w.Pos[i].x, 0f, w.Pos[i].y);
                Vector2 fwd2 = w.Goal[i] - w.Pos[i];
                float yaw = fwd2.sqrMagnitude > 1e-6f ? Mathf.Atan2(fwd2.x, fwd2.y) * Mathf.Rad2Deg : 0f;
                Quaternion rot = Quaternion.Euler(0f, yaw, 0f);
                Color col = Tint(w, i);
                // a frozen gait is what animation tier 3 looks like; Phase simply stops moving
                float swing = w.Anim[i] < 3 ? Mathf.Sin(w.Phase[i]) : 0f;
                float bob = w.Anim[i] < 3 ? Mathf.Abs(Mathf.Cos(w.Phase[i])) * 0.045f : 0f;

                for (int k = 0; k < parts.Length; k++)
                {
                    var part = parts[k];
                    Vector3 off = part.Offset;
                    Quaternion lr = Quaternion.identity;
                    if (part.Kind == 2 || part.Kind == 3)
                    {
                        float s = (k % 2 == 0) ? swing : -swing;
                        lr = Quaternion.Euler(s * (part.Kind == 2 ? 32f : 20f), 0f, 0f);
                        off += new Vector3(0f, part.Kind == 2 ? 0.16f : 0.12f, 0f);
                    }
                    off.y += bob;
                    Push(Matrix4x4.TRS(p + rot * off, rot * lr, part.Scale), col);
                }
            }
            Flush();
        }

        void Push(Matrix4x4 m, Color c)
        {
            mats[pending] = m; cols[pending] = c; pending++;
            if (pending == Batch) Flush();
        }

        void Flush()
        {
            if (pending == 0) return;
            mpb.Clear();
            mpb.SetVectorArray("_BaseColor", cols);
            var rp = new RenderParams(mat)
            {
                camera = cam,
                matProps = mpb,
                receiveShadows = false,
                shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off,
                worldBounds = new Bounds(Vector3.zero, Vector3.one * 5000f),
            };
            Graphics.RenderMeshInstanced(rp, box, 0, mats, pending);
            pending = 0;
        }
    }
}
