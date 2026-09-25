// Port of alloc/costmodel.py. frame_ms = theta_core * N + sum_{axis, tier<3} theta[axis,tier]
// * n[axis,tier]; tier 3 of every axis is the zero-cost reference level, which makes the
// per-agent intercept identifiable. This is invariant 1: the cost of a tier is *measured*
// from frame time, never authored, so nothing here reads a hand-tuned constant.
namespace Parity
{
    public sealed class RlsCostModel
    {
        public const int NFeat = 1 + ParityTable.NAxes * (ParityTable.NTiers - 1);  // 13

        public readonly double[] Theta = new double[NFeat];
        readonly double[,] P = new double[NFeat, NFeat];
        readonly double[] x = new double[NFeat];
        readonly double[] Px = new double[NFeat];
        readonly double[] k = new double[NFeat];

        public double Forget = 0.98, PMax = 1e4, ResidVar;
        public int N;

        public RlsCostModel(double p0 = 1.0)
        {
            for (int i = 0; i < NFeat; i++) P[i, i] = p0;
        }

        /// <param name="counts">[NAxes, NTiers] agents per axis-tier.</param>
        static void Features(int[,] counts, int nAgents, double[] outX)
        {
            outX[0] = nAgents;
            int f = 1;
            for (int ax = 0; ax < ParityTable.NAxes; ax++)
                for (int t = 0; t < ParityTable.NTiers - 1; t++)
                    outX[f++] = counts[ax, t];
        }

        public double Predict(int[,] counts, int nAgents)
        {
            Features(counts, nAgents, x);
            double s = 0.0;
            for (int i = 0; i < NFeat; i++) s += x[i] * Theta[i];
            return s;
        }

        public double Update(int[,] counts, int nAgents, double frameMs)
        {
            Features(counts, nAgents, x);
            double pred = 0.0;
            for (int i = 0; i < NFeat; i++) pred += x[i] * Theta[i];
            double err = frameMs - pred;

            double denom = Forget;
            for (int i = 0; i < NFeat; i++)
            {
                double s = 0.0;
                for (int j = 0; j < NFeat; j++) s += P[i, j] * x[j];
                Px[i] = s;
                denom += x[i] * s;
            }
            for (int i = 0; i < NFeat; i++) k[i] = Px[i] / denom;
            // unclamped, as in Python: clamping theta here destabilises the recursion
            for (int i = 0; i < NFeat; i++) Theta[i] += k[i] * err;
            for (int i = 0; i < NFeat; i++)
                for (int j = 0; j < NFeat; j++)
                    P[i, j] = (P[i, j] - k[i] * Px[j]) / Forget;

            double tr = 0.0;
            for (int i = 0; i < NFeat; i++) tr += P[i, i];
            if (tr > PMax * NFeat)
            {
                double f = PMax * NFeat / tr;
                for (int i = 0; i < NFeat; i++) for (int j = 0; j < NFeat; j++) P[i, j] *= f;
            }
            N++;
            double a = N > 20 ? 0.05 : 1.0 / N;
            ResidVar += a * (err * err - ResidVar);
            return err;
        }

        public double ThetaCore => Theta[0] > 0.0 ? Theta[0] : 0.0;

        /// <summary>[NAxes, NTiers] incremental ms per agent, tier 3 = 0, clamped at 0.</summary>
        public void ThetaAxis(double[,] outT)
        {
            int f = 1;
            for (int ax = 0; ax < ParityTable.NAxes; ax++)
            {
                for (int t = 0; t < ParityTable.NTiers - 1; t++)
                {
                    double v = Theta[f++];
                    outT[ax, t] = v > 0.0 ? v : 0.0;
                }
                outT[ax, ParityTable.NTiers - 1] = 0.0;
            }
        }

        /// <summary>Per-row predicted ms per agent, the allocator's cost vector for this frame.</summary>
        public void RowCosts(ParityTable table, double[,] thetaAxis, float[] outCost)
        {
            double core = ThetaCore;
            for (int c = 0; c < table.M; c++)
            {
                double s = core;
                for (int ax = 0; ax < ParityTable.NAxes; ax++) s += thetaAxis[ax, table.TierOf(c, ax)];
                outCost[c] = (float)s;
            }
        }
    }
}
