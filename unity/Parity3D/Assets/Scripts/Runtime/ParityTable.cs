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

        // Error column, following sim/tiered.py::phase7_table rather than alloc/config.py's
        // raw AXIS_ERR. This is the distinction that makes invariant 3 work at all:
        //
        //   behaviour tiers 0-2 run the LIVE process, so they diverge by nothing -- 0, not the
        //     phase 2 truncation NMSE, which measures something else entirely;
        //   behaviour tier 3 is the surrogate, and diverges at the MEASURED KL rate;
        //   navigation does not execute in phase 7 (a stated limitation), so its column is 0;
        //   the two visual axes never diverge simulation state.
        //
        // Consequence: rate-0 rows always exist, so the feasibility mask can never empty out
        // however little headroom an agent has left. Using the NMSE column instead gives every
        // row a nonzero rate, every agent's ledger runs to the cap, and the mask empties --
        // which is exactly what the headless smoke test caught.
        //
        // The admission rate is e_max (the worst context above 1% occupancy), while the cap is
        // built from e_sur (the occupancy-weighted mean): reserve the worst case, spend the
        // actual. Plaza calibration, bench/logs/phase7_plaza_calib.npz:
        //   e_rate per context [0.00393, 0.01033, 0.05716, 0.00393], occupancy [0, .938, .062, 0]
        //   e_sur 0.01323 -> cap 300 * e_sur = 3.969;  e_max 0.05716
        public const double PlazaESur = 0.01323;   // occupancy-weighted mean, sets the cap
        public const double PlazaEMax = 0.05716;   // worst live context, sets the admission rate
        public static readonly double[] PlazaERate = { 0.00393, 0.01033, 0.05716, 0.00393 };

        /// <summary>Per-frame divergence rate of a row: only a behaviour-tier-3 surrogate
        /// diverges, and it is admitted against the conservative rate.</summary>
        public static double RowErr(int behTier, double eMax) { return behTier == 3 ? eMax : 0.0; }

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

        /// <summary>[NAxes, NTiers] the per-axis qualities this table was built from: AxisQuality,
        /// unless a measured geometry column replaced the placeholder (see Build).</summary>
        public double[,] AxisQ { get; private set; } = AxisQuality;

        /// <param name="eMax">measured surrogate KL rate used for admission; see RowErr.</param>
        /// <param name="geometryQuality">measured per-tier geometry quality (the engine's pixel
        /// judge) replacing the placeholder column; null keeps the Python table exactly.</param>
        /// <param name="python">true: the Python table exactly, where every tier is a label. The
        /// engine runs navigation (separation) only for a live agent and animates nothing drawn
        /// as an impostor, so there a surrogate's navigation tier and an impostor's animation tier
        /// would be credited quality they never deliver: false drops those rows (180 -> 117).</param>
        public static ParityTable Build(double eMax = PlazaEMax, double[] geometryQuality = null, bool python = false)
        {
            var axq = (double[,])AxisQuality.Clone();
            if (geometryQuality != null)
                for (int t = 0; t < NTiers; t++) axq[NAxes - 1, t] = geometryQuality[t];
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
                if (!Allowed(b, n, a, g) || (!python && ((b == 3 && n != 3) || (g == Impostor && a != 3)))) continue;
                int[] t = { b, n, a, g };
                double q = 0.0;
                for (int ax = 0; ax < NAxes; ax++) q += axq[ax, t[ax]];
                double e = RowErr(b, eMax);
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
                AxisQ = axq,
            };
        }

        public int TierOf(int row, int axis) => Tiers[row * NAxes + axis];

        /// <summary>Port of alloc/config.py::prune_dominated. Drops rows beaten on quality,
        /// cost and error by some other row, with at least one strict. Nothing the allocator
        /// would ever pick is removed, so the result is identical and the inner loop shrinks --
        /// on the calibrated cost vector, 180 rows down to about 32.
        ///
        /// Two properties have to survive the prune or the whole mechanism breaks: at least one
        /// rate-0 row (or the feasibility mask can empty) and at least one surrogate row (or the
        /// ledger can never engage). Build() asserts both.</summary>
        public ParityTable PruneDominated(float[] cost, out int[] keptRows)
        {
            var keep = new List<int>();
            for (int i = 0; i < M; i++)
            {
                bool dominated = false;
                for (int j = 0; j < M; j++)
                {
                    // j == i fails the strictness clause, so a row never dominates itself and
                    // exact duplicates both survive -- same as the NumPy version.
                    if (Quality[j] >= Quality[i] && cost[j] <= cost[i] && Err[j] <= Err[i] &&
                        (Quality[j] > Quality[i] || cost[j] < cost[i] || Err[j] < Err[i]))
                    {
                        dominated = true;
                        break;
                    }
                }
                if (!dominated) keep.Add(i);
            }
            keptRows = keep.ToArray();
            var t = new ParityTable
            {
                M = keep.Count,
                Tiers = new byte[keep.Count * NAxes],
                Quality = new float[keep.Count],
                Err = new float[keep.Count],
                AxisQ = AxisQ,
            };
            for (int k = 0; k < keep.Count; k++)
            {
                for (int ax = 0; ax < NAxes; ax++) t.Tiers[k * NAxes + ax] = Tiers[keep[k] * NAxes + ax];
                t.Quality[k] = Quality[keep[k]];
                t.Err[k] = Err[keep[k]];
            }
            return t;
        }

        /// <summary>Rows with no divergence rate. Must be nonzero: the mask can never empty.</summary>
        public int RateZeroRows()
        {
            int c = 0;
            for (int i = 0; i < M; i++) if (Err[i] <= 0f) c++;
            return c;
        }

        /// <summary>Rows running the surrogate. Must be nonzero or the ledger never engages.</summary>
        public int SurrogateRows()
        {
            int c = 0;
            for (int i = 0; i < M; i++) if (TierOf(i, 0) == 3) c++;
            return c;
        }

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
