// Port of alloc/ledger.py (invariant 3). D_i accumulates the divergence rate of whatever
// configuration the allocator chose, and is reset only by reconciliation. headroom = cap - D
// feeds the allocator's feasibility mask, which overrides the time budget -- so an agent at
// the cap is restored to full fidelity no matter where the camera is looking. Rate-0 rows
// always exist, so the mask can never empty out.
using Unity.Collections;
using Alloc = Unity.Collections.Allocator;

namespace Parity
{
    public sealed class ErrorLedger : System.IDisposable
    {
        public float Cap;
        public NativeArray<float> D;
        public NativeArray<float> Headroom;
        public int Restorations;

        public ErrorLedger(int n, float cap, uint seed)
        {
            Cap = cap;
            D = new NativeArray<float>(n, Alloc.Persistent);
            Headroom = new NativeArray<float>(n, Alloc.Persistent);
            // desynchronise the initial ledgers so restorations do not arrive as one wave
            var rng = new Rng(seed);
            for (int i = 0; i < n; i++) D[i] = rng.NextFloat() * 0.5f * cap;
            Refresh(n);
        }

        public void Refresh(int n)
        {
            for (int i = 0; i < n; i++) { float h = Cap - D[i]; Headroom[i] = h > 0f ? h : 0f; }
        }

        public void Accrue(int i, float rate) { D[i] += rate; }

        public void Reset(int i) { D[i] = 0f; Restorations++; }

        public float Max(int n)
        {
            float m = 0f;
            for (int i = 0; i < n; i++) if (D[i] > m) m = D[i];
            return m;
        }

        public void Dispose()
        {
            if (D.IsCreated) D.Dispose();
            if (Headroom.IsCreated) Headroom.Dispose();
        }
    }

    /// <summary>xorshift32, so a run is reproducible from a seed without UnityEngine.Random
    /// global state and without pulling in Unity.Mathematics.</summary>
    public struct Rng
    {
        uint s;
        public Rng(uint seed) { s = seed == 0u ? 0x9E3779B9u : seed; }
        public uint NextUInt() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
        public float NextFloat() { return (NextUInt() >> 8) * (1.0f / 16777216.0f); }
        public float Range(float a, float b) { return a + (b - a) * NextFloat(); }
        public int Range(int a, int b) { return a + (int)(NextUInt() % (uint)(b - a)); }
    }
}
