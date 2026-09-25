// Minimal stand-ins for the UnityEngine / Unity.Collections / Unity.Jobs surface that the
// ported runtime touches, so unity/Parity3D/Assets/Scripts/Runtime/*.cs compiles and runs
// outside the editor. The ported files are compiled UNMODIFIED -- that is the whole point:
// this harness proves they build and that their arithmetic matches alloc/serial.py before
// anyone opens Unity.
//
// Jobs execute serially here. Every Execute(i) is independent of every other, so serial and
// parallel give the same answer, and serial makes the comparison deterministic.
using System;

namespace Unity.Collections
{
    public enum Allocator { Invalid = 0, None = 1, Temp = 2, TempJob = 3, Persistent = 4 }

    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Parameter)]
    public sealed class ReadOnlyAttribute : Attribute { }

    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Parameter)]
    public sealed class WriteOnlyAttribute : Attribute { }

    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Parameter)]
    public sealed class NativeDisableParallelForRestrictionAttribute : Attribute { }

    public struct NativeArray<T> : IDisposable where T : struct
    {
        T[] a;
        public NativeArray(int length, Allocator _) { a = new T[length]; }
        public T this[int i] { get { return a[i]; } set { a[i] = value; } }
        public int Length { get { return a == null ? 0 : a.Length; } }
        public bool IsCreated { get { return a != null; } }
        public void Dispose() { a = null; }
    }
}

namespace Unity.Jobs
{
    public struct JobHandle { public void Complete() { } }

    public interface IJobParallelFor { void Execute(int index); }

    public static class IJobParallelForExtensions
    {
        public static JobHandle Schedule<T>(this T job, int arrayLength, int innerBatch)
            where T : struct, IJobParallelFor
        {
            for (int i = 0; i < arrayLength; i++) job.Execute(i);
            return default(JobHandle);
        }
    }
}

namespace UnityEngine
{
    public static class Mathf
    {
        public const float PI = 3.14159265358979f;
        public const float Rad2Deg = 57.29577951308232f;
        public static float Max(float a, float b) { return a > b ? a : b; }
        public static int Max(int a, int b) { return a > b ? a : b; }
        public static float Min(float a, float b) { return a < b ? a : b; }
        public static int Min(int a, int b) { return a < b ? a : b; }
        public static float Abs(float a) { return Math.Abs(a); }
        public static float Sqrt(float a) { return (float)Math.Sqrt(a); }
        public static float Cos(float a) { return (float)Math.Cos(a); }
        public static float Sin(float a) { return (float)Math.Sin(a); }
        public static float Atan2(float y, float x) { return (float)Math.Atan2(y, x); }
        public static float Log10(float a) { return (float)Math.Log10(a); }
        public static float Clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
        public static int Clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
        public static float Clamp01(float v) { return v < 0f ? 0f : (v > 1f ? 1f : v); }
        public static int RoundToInt(float v) { return (int)Math.Round(v, MidpointRounding.AwayFromZero); }
        public static int CeilToInt(float v) { return (int)Math.Ceiling(v); }
        public static int FloorToInt(float v) { return (int)Math.Floor(v); }
        public static float Lerp(float a, float b, float t) { return a + (b - a) * Clamp01(t); }
    }

    public struct Vector2
    {
        public float x, y;
        public Vector2(float x, float y) { this.x = x; this.y = y; }
        public static Vector2 zero { get { return new Vector2(0f, 0f); } }
        public float magnitude { get { return Mathf.Sqrt(x * x + y * y); } }
        public float sqrMagnitude { get { return x * x + y * y; } }
        public Vector2 normalized { get { float m = magnitude; return m > 1e-9f ? new Vector2(x / m, y / m) : zero; } }
        public static Vector2 operator +(Vector2 a, Vector2 b) { return new Vector2(a.x + b.x, a.y + b.y); }
        public static Vector2 operator -(Vector2 a, Vector2 b) { return new Vector2(a.x - b.x, a.y - b.y); }
        public static Vector2 operator *(Vector2 a, float s) { return new Vector2(a.x * s, a.y * s); }
        public static Vector2 operator /(Vector2 a, float s) { return new Vector2(a.x / s, a.y / s); }
        public static Vector2 Lerp(Vector2 a, Vector2 b, float t) { return new Vector2(Mathf.Lerp(a.x, b.x, t), Mathf.Lerp(a.y, b.y, t)); }
    }
}
