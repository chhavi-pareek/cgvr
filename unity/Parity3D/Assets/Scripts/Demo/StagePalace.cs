// The palace set: every OpenStreetMap footprint of the compound (PalaceSite) extruded to its
// mapped height -- the palace's towers and wings come from its building:part outlines, up to
// 40 m -- with domes on the tallest parts, the compound wall with an arch at each gate, the
// mapped inner walls, lawns and paving. Afternoon light: the Jamboo Savari leaves around 4 pm.
//
// The Stage rule holds: everything solid stands on cells the sim's grid marks unwalkable
// (buildings, walls, the compound outline), and the lawns and paving are flush with the ground.
// Unmapped heights are an assumption (7 m: the one-storey fort ranges along the walls).
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Parity
{
    public static partial class Stage
    {
        const float UnmappedHeight = 7f, WallHeight = 5f, WallT = 0.5f, InnerWallHeight = 2.4f;

        static void Palace(Light sun)
        {
            var site = PalaceSite.Load();
            sun.transform.rotation = Quaternion.Euler(28f, 70f, 0f);   // low, from the west-south-west
            sun.color = new Color(1.0f, 0.83f, 0.64f);
            sun.intensity = 1.5f;
            Sky(new Color(0.46f, 0.56f, 0.74f), 1.05f, 0.8f, new Color(0.40f, 0.38f, 0.36f), 0.03f);
            Ambient(new Color(0.48f, 0.54f, 0.66f), new Color(0.46f, 0.42f, 0.38f), new Color(0.24f, 0.21f, 0.18f));
            Fog(new Color(0.74f, 0.76f, 0.80f), 0.0009f);

            var ground = Mat("palaceGround", P.Paving, new Color(0.78f, 0.72f, 0.62f), new Color(0.70f, 0.65f, 0.56f),
                             new Color(0.36f, 0.33f, 0.30f), new Vector4(1.2f, 0.6f, 0.03f, 0f), 0.18f);
            var city = Mat("palaceCity", P.Asphalt, new Color(0.20f, 0.20f, 0.21f), gloss: 0.12f);
            var lawn = Mat("palaceLawn", P.Planting, new Color(0.20f, 0.36f, 0.13f), new Color(0.36f, 0.48f, 0.18f), gloss: 0.05f);
            var granite = Mat("palaceGranite", P.Facade, new Color(0.88f, 0.84f, 0.80f), new Color(0.76f, 0.72f, 0.68f),
                              new Color(0.20f, 0.18f, 0.20f), new Vector4(3.2f, 4.2f, 5.0f, 0f), 0.22f, seed: 5f);
            var plaster = Mat("palacePlaster", P.Facade, new Color(0.90f, 0.84f, 0.70f), new Color(0.78f, 0.72f, 0.60f),
                              new Color(0.30f, 0.26f, 0.22f), new Vector4(3.0f, 3.6f, 4.0f, 0f), 0.18f, seed: 9f);
            var roof = Mat("palaceRoof", P.Plain, new Color(0.64f, 0.62f, 0.60f), gloss: 0.2f);
            var dome = Mat("palaceDome", P.Plain, new Color(0.86f, 0.66f, 0.30f), gloss: 0.7f, metal: 0.6f);
            var wall = Mat("palaceWall", P.Plain, new Color(0.84f, 0.80f, 0.72f), gloss: 0.15f);

            Box(-4000f, -0.5f, -4000f, site.Size.x + 4000f, -0.05f, site.Size.y + 4000f, city, false);
            Flat(site.Compound, 0f, ground, "compound ground");
            foreach (var g in site.Gardens) Flat(g, 0.01f, lawn, "lawn");

            // whole buildings first; parts carry the palace's own heights and stand slightly
            // inset, so their faces never coincide with the palace mass they sit in
            foreach (var b in site.Buildings)
            {
                float h = b.Height > 0f ? b.Height : UnmappedHeight;
                var ring = b.Part ? Inset(b.Ring, 0.998f) : b.Ring;
                var m = h >= 15f || b.Part ? granite : plaster;
                Prism(ring, 0f, h + (b.Part ? 0.03f : 0f), m, roof, b.Part ? "part" : "building");
                if (b.Part && h >= 32f) Dome(ring, h, dome);
            }

            // the compound wall, open at every gate, with an arch over each
            var gaps = new List<(Vector2 at, float half)>();
            foreach (var g in site.Gates) gaps.Add((g.At, g.Width * 0.5f + 0.6f));
            var c = site.Compound;
            for (int k = 0; k + 1 < c.Length; k++) WallRun(c[k], c[k + 1], gaps, WallHeight, wall);
            foreach (var g in site.Gates) Arch(g, granite, dome);
            foreach (var w in site.Walls)
                for (int k = 0; k + 1 < w.Length; k++) WallRun(w[k], w[k + 1], gaps, InnerWallHeight, wall);
        }

        // -- geometry ----------------------------------------------------------------------

        static GameObject MeshObj(string name, Mesh mesh, Material m, bool shadows)
        {
            var go = new GameObject(name);
            go.transform.SetParent(root, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var r = go.AddComponent<MeshRenderer>();
            r.sharedMaterial = m;
            r.shadowCastingMode = shadows ? ShadowCastingMode.On : ShadowCastingMode.Off;
            r.receiveShadows = true;
            r.lightProbeUsage = LightProbeUsage.Off;
            r.reflectionProbeUsage = ReflectionProbeUsage.Simple;
            return go;
        }

        /// <summary>The ring without its closing duplicate, counter-clockwise in (x, y).</summary>
        static List<Vector2> Ccw(Vector2[] ring)
        {
            var p = new List<Vector2>(ring);
            if (p.Count > 1 && (p[0] - p[p.Count - 1]).sqrMagnitude < 1e-6f) p.RemoveAt(p.Count - 1);
            for (int k = p.Count - 1; k > 0; k--) if ((p[k] - p[k - 1]).sqrMagnitude < 1e-6f) p.RemoveAt(k);
            float a = 0f;
            for (int k = 0; k < p.Count; k++) { var u = p[k]; var v = p[(k + 1) % p.Count]; a += u.x * v.y - v.x * u.y; }
            if (a < 0f) p.Reverse();
            return p;
        }

        static float Cross(Vector2 a, Vector2 b) => a.x * b.y - a.y * b.x;

        static bool InTri(Vector2 p, Vector2 a, Vector2 b, Vector2 c) =>
            Cross(b - a, p - a) >= 0f && Cross(c - b, p - b) >= 0f && Cross(a - c, p - c) >= 0f;

        /// <summary>Ear clipping of a simple counter-clockwise polygon; counter-clockwise triangles.</summary>
        static List<int> EarClip(List<Vector2> p)
        {
            var idx = new List<int>();
            for (int k = 0; k < p.Count; k++) idx.Add(k);
            var tris = new List<int>();
            int guard = 0;
            while (idx.Count > 3 && guard++ < 100000)
            {
                bool cut = false;
                for (int k = 0; k < idx.Count && !cut; k++)
                {
                    int a = idx[(k + idx.Count - 1) % idx.Count], b = idx[k], c = idx[(k + 1) % idx.Count];
                    if (Cross(p[b] - p[a], p[c] - p[b]) <= 1e-7f) continue;
                    bool hit = false;
                    foreach (int q in idx)
                        if (q != a && q != b && q != c && InTri(p[q], p[a], p[b], p[c])) { hit = true; break; }
                    if (hit) continue;
                    tris.Add(a); tris.Add(b); tris.Add(c);
                    idx.RemoveAt(k);
                    cut = true;
                }
                if (!cut) { tris.Add(idx[idx.Count - 1]); tris.Add(idx[0]); tris.Add(idx[1]); idx.RemoveAt(0); }
            }
            if (idx.Count == 3) tris.AddRange(idx);
            return tris;
        }

        /// <summary>A flat polygon at height y, facing up.</summary>
        static void Flat(Vector2[] ring, float y, Material m, string name)
        {
            var p = Ccw(ring);
            if (p.Count < 3) return;
            var v = new Vector3[p.Count];
            for (int k = 0; k < p.Count; k++) v[k] = new Vector3(p[k].x, y, p[k].y);
            var t = EarClip(p);
            for (int k = 0; k < t.Count; k += 3) (t[k + 1], t[k + 2]) = (t[k + 2], t[k + 1]);   // up-facing in Unity
            var mesh = new Mesh { indexFormat = IndexFormat.UInt32, vertices = v, triangles = t.ToArray() };
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            MeshObj(name, mesh, m, false);
        }

        /// <summary>Extruded footprint: walls in <paramref name="side"/>, flat top in <paramref name="top"/>.</summary>
        static void Prism(Vector2[] ring, float y0, float y1, Material side, Material top, string name)
        {
            var p = Ccw(ring);
            if (p.Count < 3 || y1 <= y0) return;
            var v = new List<Vector3>();
            var t = new List<int>();
            for (int k = 0; k < p.Count; k++)
            {
                Vector2 a = p[k], b = p[(k + 1) % p.Count];
                int o = v.Count;
                v.Add(new Vector3(a.x, y0, a.y)); v.Add(new Vector3(a.x, y1, a.y));
                v.Add(new Vector3(b.x, y1, b.y)); v.Add(new Vector3(b.x, y0, b.y));
                t.Add(o); t.Add(o + 1); t.Add(o + 2); t.Add(o); t.Add(o + 2); t.Add(o + 3);   // outward for a CCW ring
            }
            var walls = new Mesh { indexFormat = IndexFormat.UInt32, vertices = v.ToArray(), triangles = t.ToArray() };
            walls.RecalculateNormals();
            MeshObj(name, walls, side, true);
            Flat(ring, y1, top, name + " roof");
        }

        static Vector2[] Inset(Vector2[] ring, float k)
        {
            var c = Vector2.zero;
            foreach (var p in ring) c += p;
            c /= ring.Length;
            var o = new Vector2[ring.Length];
            for (int i = 0; i < ring.Length; i++) o[i] = c + (ring[i] - c) * k;
            return o;
        }

        static void Dome(Vector2[] ring, float h, Material m)
        {
            Vector2 lo = ring[0], hi = ring[0], c = Vector2.zero;
            foreach (var p in ring) { lo = Vector2.Min(lo, p); hi = Vector2.Max(hi, p); c += p; }
            c /= ring.Length;
            float r = 0.42f * Mathf.Min(hi.x - lo.x, hi.y - lo.y);
            Ball(new Vector3(c.x, h + r * 0.55f, c.y), new Vector3(2f * r, 2.6f * r, 2f * r), m);
            Cyl(c.x, h + r * 1.6f, c.y, 0.08f * r + 0.05f, 0.9f * r, m);
        }

        /// <summary>A wall along a-b, broken wherever it passes a gate.</summary>
        static void WallRun(Vector2 a, Vector2 b, List<(Vector2 at, float half)> gaps, float h, Material m)
        {
            float len = (b - a).magnitude;
            if (len < 0.05f) return;
            var d = (b - a) / len;
            var cuts = new List<(float s0, float s1)>();
            foreach (var g in gaps)
            {
                float s = Vector2.Dot(g.at - a, d);
                float off = Mathf.Abs(Cross(d, g.at - a));
                if (off < 2f && s > -g.half && s < len + g.half) cuts.Add((s - g.half, s + g.half));
            }
            cuts.Sort((x, y) => x.s0.CompareTo(y.s0));
            float from = 0f;
            foreach (var cut in cuts)
            {
                if (cut.s0 > from) Slab(a + d * from, a + d * Mathf.Min(cut.s0, len), h, m);
                from = Mathf.Max(from, cut.s1);
            }
            if (from < len) Slab(a + d * from, b, h, m);
        }

        static void Slab(Vector2 a, Vector2 b, float h, Material m)
        {
            var d = b - a;
            if (d.magnitude < 0.05f) return;
            var mid = (a + b) * 0.5f;
            Prim(PrimitiveType.Cube, new Vector3(mid.x, h * 0.5f, mid.y), new Vector3(WallT, h, d.magnitude), m, true,
                 Quaternion.LookRotation(new Vector3(d.x, 0f, d.y).normalized, Vector3.up));
        }

        static void Arch(PalaceSite.Gate g, Material stone, Material cap)
        {
            var along = new Vector2(-g.Out.y, g.Out.x);
            if (along.sqrMagnitude < 0.5f) along = Vector2.right;   // a closed gate carries no outward direction
            float half = g.Width * 0.5f + 0.35f, h = 8.5f;
            var p0 = g.At - along * half;
            var p1 = g.At + along * half;
            Cyl(p0.x, 0f, p0.y, 0.55f, h, stone);
            Cyl(p1.x, 0f, p1.y, 0.55f, h, stone);
            var mid = g.At;
            Prim(PrimitiveType.Cube, new Vector3(mid.x, h + 0.6f, mid.y), new Vector3(1.4f, 1.2f, 2f * half + 1.2f), stone, true,
                 Quaternion.LookRotation(new Vector3(along.x, 0f, along.y).normalized, Vector3.up));
            Ball(new Vector3(mid.x, h + 1.9f, mid.y), new Vector3(1.6f, 2.0f, 1.6f), cap);
        }
    }
}
