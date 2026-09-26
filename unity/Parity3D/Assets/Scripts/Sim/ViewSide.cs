// What a viewer sees change, and what a viewer can see at all.
//
// PopTracker is alloc/pops.py: a token bucket per agent for one viewer. A change of the
// agent's (animation, geometry) tier while it is in view is a visible pop and spends a token;
// an agent in view with less than one token has its view pair held for the frame. Over any T
// frames an agent pops at most Capacity + Refill * T times, plus any hold the allocator had to
// release to meet the frame budget. With Hold off it only counts, which is how MassLOD's pops
// are measured on the same scale.
//
// CoverageBuffer is a software occlusion pass: each agent's upright bounding rectangle is
// projected into a small buffer and the agents are drawn nearest first, so what an agent
// covers is only the pixels nobody nearer has already taken. Its visible pixel count, over
// the pixel count of an unoccluded figure at the view-salience distance, is the agent's view
// salience -- the (d0 / d)^2 area model with occlusion put back in. Walls are not occluders
// here; the crowd occluding itself is the effect that grows with density.
using UnityEngine;

namespace Parity
{
    public sealed class PopTracker
    {
        public float Capacity = 2f, Refill = 1f / 300f;   // one token per 5 s: at most 12 pops a minute
        public int Window = 120;          // the "any 2 s" window the HUD reports
        const int Ring = 32;

        float[] tokens;
        int[] prev;
        int[] ring, ringHead;
        int[] holdBuf;
        public long Pops;
        /// <summary>Pops weighted by the agent's projected area at the time ((d0/d)^2, capped at 1):
        /// a distant impostor changing is a few pixels, a figure at arm's length is not.</summary>
        public double WeightedPops;
        public long AgentFrames;
        public int WorstWindow;
        int frame;

        public void Resize(int n)
        {
            tokens = new float[n];
            prev = new int[n];
            ring = new int[n * Ring];
            ringHead = new int[n];
            holdBuf = new int[n];
            for (int i = 0; i < n; i++) { tokens[i] = Capacity; prev[i] = -1; }
            for (int i = 0; i < ring.Length; i++) ring[i] = int.MinValue / 2;
            Pops = 0; WeightedPops = 0; AgentFrames = 0; WorstWindow = 0; frame = 0;
        }

        /// <summary>Key of the (animation, geometry) pair to hold, or -1 where it may change.</summary>
        public int[] Holds(bool[] seen, int n)
        {
            for (int i = 0; i < n; i++)
                holdBuf[i] = seen[i] && tokens[i] < 1f && prev[i] >= 0 ? prev[i] : -1;
            return holdBuf;
        }

        public void Update(sbyte[] anim, sbyte[] geo, bool[] seen, int n, float[] area = null)
        {
            frame++;
            for (int i = 0; i < n; i++)
            {
                int key = anim[i] * ParityTable.NTiers + geo[i];
                bool popped = prev[i] >= 0 && key != prev[i] && seen[i];
                if (popped)
                {
                    tokens[i] -= 1f;
                    Pops++;
                    WeightedPops += area != null ? area[i] : 1.0;
                    ring[i * Ring + ringHead[i]] = frame;
                    ringHead[i] = (ringHead[i] + 1) % Ring;
                    int recent = 0;
                    for (int k = 0; k < Ring; k++) if (frame - ring[i * Ring + k] < Window) recent++;
                    if (recent > WorstWindow) WorstWindow = recent;
                }
                tokens[i] = Mathf.Min(Capacity, tokens[i] + Refill);
                prev[i] = key;
            }
            AgentFrames += n;
        }

        /// <summary>Visible pops per agent per minute at 60 frames a second.</summary>
        public float PerAgentMinute => AgentFrames > 0 ? Pops * 3600f / AgentFrames : 0f;
        /// <summary>Area-weighted visible pops per agent per minute.</summary>
        public float WeightedPerAgentMinute => AgentFrames > 0 ? (float)(WeightedPops * 3600.0 / AgentFrames) : 0f;
    }

    public sealed class CoverageBuffer
    {
        public int W = 128, H = 128;
        public float Height = 1.8f, HalfWidth = 0.3f;
        ulong[] covered;
        int[] order, x0, x1, y0, y1;
        int filled;
        readonly RadixSorter radix = new RadixSorter();
        public static readonly long[] Ticks = new long[3];
        public static long Filled;

        static int Pop(ulong x)
        {
            x -= (x >> 1) & 0x5555555555555555UL;
            x = (x & 0x3333333333333333UL) + ((x >> 2) & 0x3333333333333333UL);
            x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FUL;
            return (int)((x * 0x0101010101010101UL) >> 56);
        }

        /// <summary>Visible pixels per agent through view-projection vp (projX, projY are the
        /// projection's x and y scale). Returns the pixel count of an unoccluded figure at d0.</summary>
        public float Compute(Vector2[] pos, int n, Matrix4x4 vp, float projX, float projY, float d0, float[] visible)
        {
            long tq0 = System.Diagnostics.Stopwatch.GetTimestamp();
            int words = (W + 63) >> 6;
            if (covered == null || covered.Length != words * H) covered = new ulong[words * H];
            System.Array.Clear(covered, 0, covered.Length);
            if (order == null || order.Length < n)
            {
                order = new int[n];
                x0 = new int[n]; x1 = new int[n]; y0 = new int[n]; y1 = new int[n];
            }
            var keys = radix.Keys(n);
            filled = 0;
            // rows x, y and w of vp at the foot (y = 0); the head adds Height times column 1
            float m00 = vp.m00, m02 = vp.m02, m03 = vp.m03, m10 = vp.m10, m12 = vp.m12, m13 = vp.m13;
            float m30 = vp.m30, m32 = vp.m32, m33 = vp.m33;
            float hX = vp.m01 * Height, hY = vp.m11 * Height, hW = vp.m31 * Height;
            for (int i = 0; i < n; i++)
            {
                visible[i] = 0f;
                float px = pos[i].x, pz = pos[i].y;
                float fxc = m00 * px + m02 * pz + m03, fyc = m10 * px + m12 * pz + m13, fw = m30 * px + m32 * pz + m33;
                float hw0 = fw + hW;
                if (fw <= 0.1f || hw0 <= 0.1f) continue;
                float fx = fxc / fw, fy = fyc / fw, hx = (fxc + hX) / hw0, hy = (fyc + hY) / hw0;
                float hw = HalfWidth * projX / (0.5f * (fw + hw0));
                float ax = Mathf.Min(fx, hx) - hw, bx = Mathf.Max(fx, hx) + hw;
                float ay = Mathf.Min(fy, hy), by = Mathf.Max(fy, hy);
                int px0 = Mathf.Clamp(Mathf.FloorToInt((ax + 1f) * 0.5f * W), 0, W);
                int px1 = Mathf.Clamp(Mathf.CeilToInt((bx + 1f) * 0.5f * W), 0, W);
                int py0 = Mathf.Clamp(Mathf.FloorToInt((ay + 1f) * 0.5f * H), 0, H);
                int py1 = Mathf.Clamp(Mathf.CeilToInt((by + 1f) * 0.5f * H), 0, H);
                if (px1 <= px0 || py1 <= py0) continue;
                x0[i] = px0; x1[i] = px1; y0[i] = py0; y1[i] = py1;
                keys[filled] = (uint)Mathf.Min(fw * 64f, 65535f);     // 1.6 cm depth steps: two radix passes
                order[filled] = i;
                filled++;
            }
            long tq = System.Diagnostics.Stopwatch.GetTimestamp(); Ticks[0] += tq - tq0; Filled += filled;
            radix.Sort(order, filled);                          // nearest first
            long tr = System.Diagnostics.Stopwatch.GetTimestamp(); Ticks[1] += tr - tq;
            for (int k = 0; k < filled; k++)
            {
                int i = order[k], c = 0;
                int w0 = x0[i] >> 6, w1 = (x1[i] - 1) >> 6;
                for (int w = w0; w <= w1; w++)
                {
                    int lo = Mathf.Max(x0[i] - (w << 6), 0), hi = Mathf.Min(x1[i] - (w << 6), 64);
                    ulong mask = (hi == 64 ? ~0UL : (1UL << hi) - 1UL) & ~((1UL << lo) - 1UL);
                    for (int y = y0[i]; y < y1[i]; y++)
                    {
                        int at = y * words + w;
                        ulong fresh = mask & ~covered[at];
                        if (fresh == 0UL) continue;
                        c += Pop(fresh);
                        covered[at] |= fresh;
                    }
                }
                visible[i] = c;
            }
            Ticks[2] += System.Diagnostics.Stopwatch.GetTimestamp() - tr;
            float refW = 2f * HalfWidth * projX / d0 * 0.5f * W;
            float refH = Height * projY / d0 * 0.5f * H;
            return Mathf.Max(refW * refH, 1f);
        }
    }
}
