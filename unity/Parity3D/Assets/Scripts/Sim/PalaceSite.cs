// The Mysore Palace site for the engine. tools/export_site_unity.py writes the grids that
// sim/site.py rasterised from OpenStreetMap (walkable, clear of walls, building, one sink per
// gate), the gates, the seat-block candidates and the building outlines into
// StreamingAssets/palace. The distance fields and routes are recomputed here exactly as
// sim/site.py builds them -- same 8-neighbour graph, same descent order, same string-pulling --
// and ParitySmoke checks both against the Python values in oracle.txt.
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Parity
{
    public sealed class PalaceSite
    {
        public struct Gate { public string Key; public bool Open; public Vector2 At, Out; public float Width; }
        public struct Footprint { public float Height; public bool Part; public Vector2[] Ring; }

        public int Nx, Ny;
        public float Res;
        public Vector2 Shift, Size;
        public bool[] Walk, Clear, Building, SinkAny;
        public bool[][] Sink;
        public Gate[] Gates;
        public int[] Open;
        public readonly List<Vector2> Blocks = new List<Vector2>();
        public readonly List<Footprint> Buildings = new List<Footprint>();
        public readonly List<Vector2[]> Walls = new List<Vector2[]>(), Gardens = new List<Vector2[]>();
        public Vector2[] Compound;
        public readonly string Dir;

        static readonly int[] DI = { 1, -1, 0, 0, 1, 1, -1, -1 };
        static readonly int[] DJ = { 0, 0, 1, -1, 1, -1, 1, -1 };
        static readonly double[] DC = { 1, 1, 1, 1, 1.4142135623730951, 1.4142135623730951, 1.4142135623730951, 1.4142135623730951 };

        double[][] fields;
        static PalaceSite cached;

        public static PalaceSite Load()
        {
            if (cached == null) cached = new PalaceSite(Path.Combine(Application.streamingAssetsPath, "palace"));
            return cached;
        }

        PalaceSite(string dir)
        {
            Dir = dir;
            using (var br = new BinaryReader(File.OpenRead(Path.Combine(dir, "grid.bytes"))))
            {
                if (new string(br.ReadChars(4)) != "PSG1") throw new IOException("palace grid.bytes: bad magic");
                Nx = br.ReadInt32(); Ny = br.ReadInt32();
                Res = br.ReadSingle();
                Shift = new Vector2(br.ReadSingle(), br.ReadSingle());
                int gates = br.ReadInt32();
                int n = Nx * Ny, bytes = (n + 7) / 8;
                Walk = Unpack(br.ReadBytes(bytes), n);
                Clear = Unpack(br.ReadBytes(bytes), n);
                Building = Unpack(br.ReadBytes(bytes), n);
                Sink = new bool[gates][];
                for (int g = 0; g < gates; g++) Sink[g] = Unpack(br.ReadBytes(bytes), n);
                Gates = new Gate[gates];
            }
            var open = new List<int>();
            foreach (var line in File.ReadAllLines(Path.Combine(dir, "site.txt")))
            {
                var t = line.Split(' ');
                switch (t[0])
                {
                    case "size": Size = new Vector2(F(t[1]), F(t[2])); break;
                    case "gate":
                        int k = int.Parse(t[1]);
                        Gates[k] = new Gate { Key = t[2], Open = t[3] == "1", At = new Vector2(F(t[4]), F(t[5])),
                                              Width = F(t[6]), Out = new Vector2(F(t[7]), F(t[8])) };
                        if (Gates[k].Open) open.Add(k);
                        break;
                    case "block": Blocks.Add(new Vector2(F(t[1]), F(t[2]))); break;
                    case "building": Buildings.Add(new Footprint { Height = F(t[1]), Ring = Ring(t, 2) }); break;
                    case "part": Buildings.Add(new Footprint { Height = F(t[1]), Part = true, Ring = Ring(t, 2) }); break;
                    case "compound": Compound = Ring(t, 1); break;
                    case "wall": Walls.Add(Ring(t, 1)); break;
                    case "garden": Gardens.Add(Ring(t, 1)); break;
                }
            }
            Open = open.ToArray();
            SinkAny = new bool[Nx * Ny];
            foreach (int g in Open)
                for (int c = 0; c < SinkAny.Length; c++) SinkAny[c] |= Sink[g][c];
            fields = new double[Gates.Length][];
        }

        static float F(string s) => float.Parse(s, System.Globalization.CultureInfo.InvariantCulture);

        static Vector2[] Ring(string[] t, int at)
        {
            int n = int.Parse(t[at]);
            var r = new Vector2[n];
            for (int k = 0; k < n; k++) r[k] = new Vector2(F(t[at + 1 + 2 * k]), F(t[at + 2 + 2 * k]));
            return r;
        }

        static bool[] Unpack(byte[] b, int n)
        {
            var m = new bool[n];
            for (int c = 0; c < n; c++) m[c] = ((b[c >> 3] >> (7 - (c & 7))) & 1) != 0;   // numpy packbits order
            return m;
        }

        // -- cells -----------------------------------------------------------------------------
        public int CellI(double x) => System.Math.Min(System.Math.Max((int)System.Math.Floor(x / Res), 0), Nx - 1);
        public int CellJ(double y) => System.Math.Min(System.Math.Max((int)System.Math.Floor(y / Res), 0), Ny - 1);

        public bool Walkable(double x, double y, bool[] grid)
        {
            if (x < 0 || y < 0 || x >= Nx * (double)Res || y >= Ny * (double)Res) return false;
            return grid[CellI(x) * Ny + CellJ(y)];
        }

        public bool Walkable(Vector2 p) => Walkable(p.x, p.y, Walk);
        public bool InSink(Vector2 p) => Walkable(p.x, p.y, SinkAny);

        // -- distance fields -------------------------------------------------------------------
        /// <summary>Metres from each cell to gate g's sink along clear cells, 8-connected without
        /// corner cutting (sim/site.py Site.field); +inf where unreachable.</summary>
        public double[] Field(int g)
        {
            if (fields[g] != null) return fields[g];
            int n = Nx * Ny;
            var d = new double[n];
            for (int c = 0; c < n; c++) d[c] = double.PositiveInfinity;
            var heap = new MinHeap(1 << 16);
            for (int c = 0; c < n; c++)
                if (Sink[g][c] && Clear[c]) { d[c] = 0; heap.Push(0, c); }
            while (heap.Count > 0)
            {
                heap.Pop(out double du, out int u);
                if (du > d[u]) continue;
                int ui = u / Ny, uj = u - ui * Ny;
                for (int k = 0; k < 8; k++)
                {
                    int vi = ui + DI[k], vj = uj + DJ[k];
                    if (vi < 0 || vj < 0 || vi >= Nx || vj >= Ny) continue;
                    int v = vi * Ny + vj;
                    if (!Clear[v]) continue;
                    if (DI[k] != 0 && DJ[k] != 0 && !(Clear[(ui + DI[k]) * Ny + uj] && Clear[ui * Ny + uj + DJ[k]])) continue;
                    double nd = du + DC[k] * Res;
                    if (nd < d[v]) { d[v] = nd; heap.Push(nd, v); }
                }
            }
            return fields[g] = d;
        }

        public double Distance(Vector2 p, int g) => Field(g)[CellI(p.x) * Ny + CellJ(p.y)];

        public int NearestGate(Vector2 p)
        {
            int best = -1;
            double bd = double.PositiveInfinity;
            foreach (int g in Open)
            {
                double d = Distance(p, g);
                if (d < bd) { bd = d; best = g; }
            }
            return best;
        }

        // -- routes ----------------------------------------------------------------------------
        public bool Visible(double ax, double ay, double bx, double by)
        {
            double len = System.Math.Sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
            int n = System.Math.Max((int)System.Math.Ceiling(len / (Res * 0.5)), 1);
            for (int s = 0; s <= n; s++)
            {
                double t = (double)s / n;
                if (!Walkable(ax + t * (bx - ax), ay + t * (by - ay), Clear)) return false;
            }
            return true;
        }

        /// <summary>Polyline from start to gate g's sink: down the field, string-pulled on clear
        /// cells (sim/site.py Site.route). Null when start cannot reach the gate.</summary>
        public Vector2[] Route(Vector2 start, int g)
        {
            var f = Field(g);
            int c = CellI(start.x) * Ny + CellJ(start.y);
            if (double.IsInfinity(f[c])) return null;
            var path = new List<int> { c };
            while (!Sink[g][c])
            {
                int ci = c / Ny, cj = c - ci * Ny, next = -1;
                double best = f[c];
                for (int k = 0; k < 8; k++)
                {
                    int vi = ci + DI[k], vj = cj + DJ[k];
                    if (vi < 0 || vj < 0 || vi >= Nx || vj >= Ny) continue;
                    int v = vi * Ny + vj;
                    if (f[v] < best) { best = f[v]; next = v; }
                }
                if (next < 0) return null;
                path.Add(next);
                c = next;
            }
            var px = new double[path.Count];
            var py = new double[path.Count];
            for (int k = 0; k < path.Count; k++)
            {
                int pi = path[k] / Ny;
                px[k] = (pi + 0.5) * Res;
                py[k] = (path[k] - pi * Ny + 0.5) * Res;
            }
            var ox = new List<double> { start.x };
            var oy = new List<double> { start.y };
            int i = 0;
            while (i < path.Count - 1)
            {
                int j = path.Count - 1;
                while (j > i + 1 && !Visible(ox[ox.Count - 1], oy[oy.Count - 1], px[j], py[j])) j--;
                ox.Add(px[j]); oy.Add(py[j]);
                i = j;
            }
            var r = new Vector2[ox.Count];
            for (int k = 0; k < r.Length; k++) r[k] = new Vector2((float)ox[k], (float)oy[k]);
            return r;
        }

        /// <summary>Unit step toward gate g from p: down the field on a clear cell, and toward the
        /// nearest clear cell from a walkable one beside a wall (sim/site.py Site.flow).</summary>
        public Vector2 Steer(Vector2 p, int g)
        {
            var f = Field(g);
            int ci = CellI(p.x), cj = CellJ(p.y), c = ci * Ny + cj;
            if (Clear[c])
            {
                double best = f[c];
                Vector2 step = Vector2.zero;
                for (int k = 0; k < 8; k++)
                {
                    int vi = ci + DI[k], vj = cj + DJ[k];
                    if (vi < 0 || vj < 0 || vi >= Nx || vj >= Ny) continue;
                    if (f[vi * Ny + vj] < best) { best = f[vi * Ny + vj]; step = new Vector2(DI[k], DJ[k]) / (float)DC[k]; }
                }
                return step;
            }
            Vector2 to = Vector2.zero;
            float bd = float.PositiveInfinity;
            for (int di = -3; di <= 3; di++)
                for (int dj = -3; dj <= 3; dj++)
                {
                    int vi = ci + di, vj = cj + dj;
                    if (vi < 0 || vj < 0 || vi >= Nx || vj >= Ny || !Clear[vi * Ny + vj]) continue;
                    float d2 = di * di + dj * dj;
                    if (d2 < bd) { bd = d2; to = new Vector2(di, dj); }
                }
            return bd < float.PositiveInfinity ? to.normalized : Vector2.zero;
        }

        /// <summary>Moves that end off walkable ground slide along one axis, or are undone
        /// (sim/site.py Site.constrain).</summary>
        public Vector2 Constrain(Vector2 p, Vector2 prev)
        {
            if (Walkable(p)) return p;
            var sx = new Vector2(p.x, prev.y);
            if (Walkable(sx)) return sx;
            var sy = new Vector2(prev.x, p.y);
            return Walkable(sy) ? sy : prev;
        }

        public static bool Inside(Vector2[] ring, Vector2 p)
        {
            bool c = false;
            for (int a = 0, b = ring.Length - 1; a < ring.Length; b = a++)
                if ((ring[a].y > p.y) != (ring[b].y > p.y) &&
                    p.x < (ring[b].x - ring[a].x) * (p.y - ring[a].y) / (ring[b].y - ring[a].y) + ring[a].x) c = !c;
            return c;
        }

        // -- oracle ----------------------------------------------------------------------------
        /// <summary>Compares the recomputed fields and routes with oracle.txt from Python. Returns
        /// the number of disagreements and the worst field and route-length differences.</summary>
        public int CheckOracle(out double worstField, out double worstRoute, out int routes)
        {
            int bad = 0;
            worstField = worstRoute = 0;
            routes = 0;
            foreach (var line in File.ReadAllLines(Path.Combine(Dir, "oracle.txt")))
            {
                var t = line.Split(' ');
                int g = int.Parse(t[1]);
                if (t[0] == "field")
                {
                    double want = double.Parse(t[4], System.Globalization.CultureInfo.InvariantCulture);
                    double got = Field(g)[int.Parse(t[2]) * Ny + int.Parse(t[3])];
                    double e = System.Math.Abs(got - want);
                    worstField = System.Math.Max(worstField, e);
                    if (e > 1e-4) bad++;
                }
                else if (t[0] == "route")
                {
                    routes++;
                    var r = Route(new Vector2(F(t[2]), F(t[3])), g);
                    double want = double.Parse(t[4], System.Globalization.CultureInfo.InvariantCulture);
                    if (r == null) { bad++; continue; }
                    double len = 0;
                    for (int k = 1; k < r.Length; k++) len += (r[k] - r[k - 1]).magnitude;
                    double e = System.Math.Abs(len - want);
                    worstRoute = System.Math.Max(worstRoute, e);
                    if (e > 0.01 * want + 0.05) bad++;   // float32 start points can shift a pull by a cell
                }
            }
            return bad;
        }

        sealed class MinHeap
        {
            double[] key;
            int[] val;
            public int Count;
            public MinHeap(int cap) { key = new double[cap]; val = new int[cap]; }

            public void Push(double k, int v)
            {
                if (Count == key.Length) { System.Array.Resize(ref key, Count * 2); System.Array.Resize(ref val, Count * 2); }
                int i = Count++;
                while (i > 0)
                {
                    int p = (i - 1) >> 1;
                    if (key[p] <= k) break;
                    key[i] = key[p]; val[i] = val[p]; i = p;
                }
                key[i] = k; val[i] = v;
            }

            public void Pop(out double k, out int v)
            {
                k = key[0]; v = val[0];
                double lk = key[--Count];
                int lv = val[Count], i = 0;
                while (true)
                {
                    int a = 2 * i + 1;
                    if (a >= Count) break;
                    if (a + 1 < Count && key[a + 1] < key[a]) a++;
                    if (key[a] >= lk) break;
                    key[i] = key[a]; val[i] = val[a]; i = a;
                }
                key[i] = lk; val[i] = lv;
            }
        }
    }
}
