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
        bool[] covered;
        float[] depth;
        int[] order, x0, x1, y0, y1;
        int filled;

        /// <summary>Visible pixels per agent through view-projection vp (projX, projY are the
        /// projection's x and y scale). Returns the pixel count of an unoccluded figure at d0.</summary>
        public float Compute(Vector2[] pos, int n, Matrix4x4 vp, float projX, float projY, float d0, float[] visible)
        {
            if (covered == null || covered.Length != W * H) covered = new bool[W * H];
            System.Array.Clear(covered, 0, covered.Length);
            if (depth == null || depth.Length < n)
            {
                depth = new float[n]; order = new int[n];
                x0 = new int[n]; x1 = new int[n]; y0 = new int[n]; y1 = new int[n];
            }
            filled = 0;
            for (int i = 0; i < n; i++)
            {
                visible[i] = 0f;
                var foot = vp * new Vector4(pos[i].x, 0f, pos[i].y, 1f);
                var head = vp * new Vector4(pos[i].x, Height, pos[i].y, 1f);
                if (foot.w <= 0.1f || head.w <= 0.1f) continue;
                float fx = foot.x / foot.w, fy = foot.y / foot.w, hx = head.x / head.w, hy = head.y / head.w;
                float hw = HalfWidth * projX / (0.5f * (foot.w + head.w));
                float ax = Mathf.Min(fx, hx) - hw, bx = Mathf.Max(fx, hx) + hw;
                float ay = Mathf.Min(fy, hy), by = Mathf.Max(fy, hy);
                int px0 = Mathf.Clamp(Mathf.FloorToInt((ax + 1f) * 0.5f * W), 0, W);
                int px1 = Mathf.Clamp(Mathf.CeilToInt((bx + 1f) * 0.5f * W), 0, W);
                int py0 = Mathf.Clamp(Mathf.FloorToInt((ay + 1f) * 0.5f * H), 0, H);
                int py1 = Mathf.Clamp(Mathf.CeilToInt((by + 1f) * 0.5f * H), 0, H);
                if (px1 <= px0 || py1 <= py0) continue;
                x0[i] = px0; x1[i] = px1; y0[i] = py0; y1[i] = py1;
                depth[filled] = foot.w;
                order[filled] = i;
                filled++;
            }
            System.Array.Sort(depth, order, 0, filled);          // nearest first
            for (int k = 0; k < filled; k++)
            {
                int i = order[k], c = 0;
                for (int y = y0[i]; y < y1[i]; y++)
                {
                    int row = y * W;
                    for (int x = x0[i]; x < x1[i]; x++)
                        if (!covered[row + x]) { covered[row + x] = true; c++; }
                }
                visible[i] = c;
            }
            float refW = 2f * HalfWidth * projX / d0 * 0.5f * W;
            float refH = Height * projY / d0 * 0.5f * H;
            return Mathf.Max(refW * refH, 1f);
        }
    }
}
