// Port of sim/assign_threshold.py: the UE5 MassLOD-style baseline this whole project is
// measured against. Distance bands with 10% edge hysteresis, a frustum test with a 15-frame
// visibility hold, an out-of-view significance penalty, and hard per-level caps that demote
// the least significant agents first. One tier per agent, applied to every axis at once --
// that coupling is exactly what PARITY drops.
using UnityEngine;

namespace Parity
{
    public sealed class MassLodAssigner
    {
        public float[] VisBands = { 15f, 40f, 90f };
        public float[] SimBands = { 20f, 60f, 140f };
        public int[] VisCaps = { 50, 250, 800 };
        public int[] SimCaps = { 80, 400, 1200 };
        public float HystDist = 0.10f;
        public int HystView = 15;
        public float OovPenalty = 2f;
        public float Fov = Mathf.PI / 2f;
        public float ViewMargin = 0.1745f;
        public float Far = 250f;
        public float CamH = 1.7f;

        sbyte[] vis, sim;
        short[] hold;
        int[] order;
        float[] dist;
        int n;

        public void Resize(int count)
        {
            n = count;
            vis = new sbyte[n]; sim = new sbyte[n]; hold = new short[n];
            order = new int[n]; dist = new float[n];
            cold = true;
        }

        bool cold = true;

        static void BandTier(float[] d, float[] bands, sbyte[] t, float h, int n)
        {
            int nb = bands.Length;
            for (int pass = 0; pass < nb; pass++)
                for (int i = 0; i < n; i++)
                {
                    int ti = t[i];
                    if (ti < nb && d[i] > bands[Mathf.Min(ti, nb - 1)] * (1f + h)) t[i] = (sbyte)(ti + 1);
                }
            for (int pass = 0; pass < nb; pass++)
                for (int i = 0; i < n; i++)
                {
                    int ti = t[i];
                    if (ti > 0 && d[i] < bands[Mathf.Max(ti - 1, 0)] * (1f - h)) t[i] = (sbyte)(ti - 1);
                }
        }

        /// <summary>Caps cascade: an agent pushed out of level L competes for level L+1.</summary>
        static void Cap(sbyte[] t, int[] order, int[] caps, int n)
        {
            for (int lvl = 0; lvl < caps.Length; lvl++)
            {
                int seen = 0;
                for (int k = 0; k < n; k++)
                {
                    int i = order[k];
                    if (t[i] != lvl) continue;
                    seen++;
                    if (seen > caps[lvl]) t[i] = (sbyte)(lvl + 1);
                }
            }
        }

        /// <param name="sig">out: significance, also the salience source for PARITY.</param>
        public void Step(Vector2[] pos, int count, Vector2 camPos, float camYaw,
                         float[] sig, bool[] inView, sbyte[] simTier, sbyte[] visTier)
        {
            if (n != count) Resize(count);
            float cosLimit = Mathf.Cos(Fov * 0.5f + ViewMargin);
            var fwd = new Vector2(Mathf.Cos(camYaw), Mathf.Sin(camYaw));
            for (int i = 0; i < n; i++)
            {
                float rx = pos[i].x - camPos.x, ry = pos[i].y - camPos.y;
                float d2 = rx * rx + ry * ry;
                float d = Mathf.Sqrt(d2 + CamH * CamH);
                dist[i] = d;
                float cosang = (rx * fwd.x + ry * fwd.y) / Mathf.Max(Mathf.Sqrt(d2), 1e-6f);
                bool seen = cosang >= cosLimit && d <= Far;
                hold[i] = seen ? (short)HystView : (short)Mathf.Max(hold[i] - 1, 0);
                inView[i] = hold[i] > 0;
                sig[i] = d * (inView[i] ? 1f : OovPenalty);
            }
            if (cold)
            {
                for (int i = 0; i < n; i++) { vis[i] = 0; sim[i] = 0; }
                BandTier(dist, VisBands, vis, 0f, n);
                cold = false;
            }
            // hysteresis state holds the pre-cap tier, so cap pressure never becomes sticky
            BandTier(dist, VisBands, vis, HystDist, n);
            BandTier(sig, SimBands, sim, HystDist, n);

            for (int i = 0; i < n; i++) order[i] = i;
            var keys = sig;
            System.Array.Sort(order, (a, b) => keys[a].CompareTo(keys[b]));  // stable enough: ties are equal-significance

            for (int i = 0; i < n; i++) visTier[i] = inView[i] ? vis[i] : (sbyte)3;
            for (int i = 0; i < n; i++) simTier[i] = sim[i];
            Cap(visTier, order, VisCaps, n);
            Cap(simTier, order, SimCaps, n);
        }
    }
}
