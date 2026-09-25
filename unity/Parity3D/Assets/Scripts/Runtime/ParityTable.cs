// Port of alloc/config.py. The table is *constructed* the same way rather than dumped as
// data, so a change to the Python constants shows up here as a diff and not as a silently
// stale blob. tools/gen_unity_table.py re-derives both and asserts they agree.
using System.Collections.Generic;

namespace Parity
{
    public sealed class ParityTable
    {
        public const int NAxes = 4;
        public const int NTiers = 4;

        public static readonly string[] AxisNames = { "behaviour", "navigation", "animation", "geometry" };

        // phase 2 nested truncation, held-out normalised MSE, k = 16 / 8 / 4. Behaviour and
        // animation quality are 1 - this, so those two axes share a unit through the latent.
        const double B0 = 0.132, B1 = 0.213, B2 = 0.449;
        const double A0 = 0.196, A1 = 0.435, A2 = 0.746;

        public static readonly double[,] AxisQuality =
        {
            { 1.0 - B0, 1.0 - B1, 1.0 - B2, 0.0 },
            { 1.0, 0.9, 0.6, 0.3 },       // placeholder: trajectory deviation vs full ORCA
            { 1.0 - A0, 1.0 - A1, 1.0 - A2, 0.0 },
            { 1.0, 0.75, 0.5, 0.25 },     // placeholder: screen-space geometric error
        };

        // per-frame state-divergence rate; the visual axes do not diverge simulation state
        public static readonly double[,] AxisErr =
        {
            { B0, B1, B2, 1.0 },
            { 0.0, 0.05, 0.2, 0.4 },      // placeholder: measured nav deviation
            { 0.0, 0.0, 0.0, 0.0 },
            { 0.0, 0.0, 0.0, 0.0 },
        };

        const int Impostor = 3;
        const int FullIk = 0;

        static bool Allowed(int b, int n, int a, int g)
        {
            if (g == Impostor && a == FullIk) return false;              // no IK under an impostor
            if ((n == 2 || n == 3) && (b == 0 || b == 1)) return false;  // field nav cannot bear gestures
            return true;
        }

        /// <summary>Row count after the coupling prune. Matches build_table(): 180.</summary>
        public int M { get; private set; }

        /// <summary>[M * NAxes] per-axis tier of each row, row-major.</summary>
        public byte[] Tiers { get; private set; }

        /// <summary>[M] mean quality across axes.</summary>
        public float[] Quality { get; private set; }

        /// <summary>[M] summed per-frame divergence rate.</summary>
        public float[] Err { get; private set; }

        public static ParityTable Build()
        {
            var tiers = new List<byte>();
            var quality = new List<float>();
            var err = new List<float>();
            // itertools.product(range(4), repeat=4) is lexicographic; keep that row order so a
            // row index means the same thing here as in the Python oracle vectors.
            for (int b = 0; b < NTiers; b++)
            for (int n = 0; n < NTiers; n++)
            for (int a = 0; a < NTiers; a++)
            for (int g = 0; g < NTiers; g++)
            {
                if (!Allowed(b, n, a, g)) continue;
                int[] t = { b, n, a, g };
                double q = 0.0, e = 0.0;
                for (int ax = 0; ax < NAxes; ax++)
                {
                    q += AxisQuality[ax, t[ax]];
                    e += AxisErr[ax, t[ax]];
                }
                for (int ax = 0; ax < NAxes; ax++) tiers.Add((byte)t[ax]);
                quality.Add((float)(q / NAxes));
                err.Add((float)e);
            }
            return new ParityTable
            {
                M = quality.Count,
                Tiers = tiers.ToArray(),
                Quality = quality.ToArray(),
                Err = err.ToArray(),
            };
        }

        public int TierOf(int row, int axis) => Tiers[row * NAxes + axis];

        /// <summary>Row index of the cheapest-fidelity configuration (all axes at tier 3).</summary>
        public int FloorRow()
        {
            for (int c = 0; c < M; c++)
            {
                bool all3 = true;
                for (int ax = 0; ax < NAxes; ax++) if (TierOf(c, ax) != 3) { all3 = false; break; }
                if (all3) return c;
            }
            return M - 1;
        }
    }
}
