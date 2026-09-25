// Runs the ported allocator over cases exported by tools/gen_oracle_cases.py and writes the
// results back for comparison against alloc/serial.py.
//
//   dotnet run --project unity/OracleCheck -- cases.json results.json
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using Unity.Collections;
using Parity;

static class Program
{
    class Case
    {
        public int n { get; set; }
        public float budget { get; set; }
        public float[] salience { get; set; }
        public float[] rowCost { get; set; }
        public float[] headroom { get; set; }
        public int fillMax { get; set; }
        public bool warm { get; set; }
        public float[] view { get; set; }   // present: the two-salience allocator on the full table
        public float[][] views { get; set; }  // present: several viewers, one row list per viewer
        public int[][] holds { get; set; }    // per viewer: view pair held by the pop ledger, or -1
    }

    static int Main(string[] args)
    {
        var table = ParityTable.Build();
        Console.WriteLine($"C# table: m = {table.M}");
        if (args.Length < 2) { Console.Error.WriteLine("usage: <cases.json> <results.json>"); return 2; }

        var json = File.ReadAllText(args[0]);
        var cases = JsonSerializer.Deserialize<List<Case>>(json,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        var sb = new StringBuilder();
        sb.Append('[');
        FidelityAllocator warmAlloc = null;
        for (int k = 0; k < cases.Count; k++)
        {
            var c = cases[k];
            // a warm case reuses the previous allocator so the multiplier carries over, which
            // is the path that actually runs every frame in the engine. A cold case gets a
            // fresh one; the previous is only released once nothing can still be holding it.
            if (!c.warm || warmAlloc == null)
            {
                if (warmAlloc != null) warmAlloc.Dispose();
                warmAlloc = new FidelityAllocator(table);
            }
            if (c.views != null)
            {
                var fac = FactoredAllocator.TryCreate(table);
                var hm = new float[c.n];
                for (int i = 0; i < c.n; i++) hm[i] = c.headroom != null ? c.headroom[i] : float.MaxValue;
                int V = c.views.Length;
                var am = new int[V][];
                for (int v = 0; v < V; v++) am[v] = new int[c.n];
                fac.SetCosts(c.rowCost);
                var rf = fac.Solve(c.salience, c.views, c.holds, hm, c.n, c.budget, am);
                if (k > 0) sb.Append(',');
                sb.Append("{\"assign\":[");
                for (int v = 0; v < V; v++)
                {
                    if (v > 0) sb.Append(',');
                    sb.Append('[');
                    for (int i = 0; i < c.n; i++) { if (i > 0) sb.Append(','); sb.Append(am[v][i]); }
                    sb.Append(']');
                }
                sb.Append("],\"cost\":").Append(F(rf.Cost))
                  .Append(",\"released\":").Append(fac.Released)
                  .Append(",\"infeasible\":").Append(rf.Infeasible ? "true" : "false")
                  .Append('}');
                continue;
            }
            if (c.view != null)
            {
                var fac = FactoredAllocator.TryCreate(table);
                var hm = new float[c.n];
                for (int i = 0; i < c.n; i++) hm[i] = c.headroom != null ? c.headroom[i] : float.MaxValue;
                var am = new int[c.n];
                fac.SetCosts(c.rowCost);
                var swf = System.Diagnostics.Stopwatch.StartNew();
                var rf = fac.Solve(c.salience, c.view, hm, c.n, c.budget, am);
                swf.Stop();
                if (k > 0) sb.Append(',');
                sb.Append("{\"assign\":[");
                for (int i = 0; i < c.n; i++) { if (i > 0) sb.Append(','); sb.Append(am[i]); }
                sb.Append("],\"cost\":").Append(F(rf.Cost))
                  .Append(",\"utility\":").Append(F(rf.Utility))
                  .Append(",\"lambda\":").Append(F(rf.Lambda))
                  .Append(",\"evals\":").Append(rf.Evals)
                  .Append(",\"infeasible\":").Append(rf.Infeasible ? "true" : "false")
                  .Append(",\"ms\":").Append(F((float)swf.Elapsed.TotalMilliseconds))
                  .Append('}');
                continue;
            }
            var alloc = warmAlloc;
            alloc.FillMax = c.fillMax;

            var s = new NativeArray<float>(c.n, Allocator.Persistent);
            var h = new NativeArray<float>(c.n, Allocator.Persistent);
            var a = new NativeArray<int>(c.n, Allocator.Persistent);
            for (int i = 0; i < c.n; i++)
            {
                s[i] = c.salience[i];
                // no mask in this case: headroom above every row's error rate
                h[i] = c.headroom != null ? c.headroom[i] : float.MaxValue;
            }
            alloc.SetCosts(table, c.rowCost);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var r = alloc.Solve(s, h, c.n, c.budget, a);
            sw.Stop();

            if (k > 0) sb.Append(',');
            sb.Append("{\"assign\":[");
            for (int i = 0; i < c.n; i++) { if (i > 0) sb.Append(','); sb.Append(a[i]); }
            sb.Append("],\"cost\":").Append(F(r.Cost))
              .Append(",\"utility\":").Append(F(r.Utility))
              .Append(",\"lambda\":").Append(F(r.Lambda))
              .Append(",\"evals\":").Append(r.Evals)
              .Append(",\"fill\":").Append(r.FillSteps)
              .Append(",\"slack\":").Append(F(r.Slack))
              .Append(",\"infeasible\":").Append(r.Infeasible ? "true" : "false")
              .Append(",\"forced\":").Append(r.Forced ? "true" : "false")
              .Append(",\"ms\":").Append(F((float)sw.Elapsed.TotalMilliseconds))
              .Append('}');

            s.Dispose(); h.Dispose(); a.Dispose();
        }
        sb.Append(']');
        if (warmAlloc != null) warmAlloc.Dispose();
        File.WriteAllText(args[1], sb.ToString());
        Console.WriteLine($"ran {cases.Count} cases -> {args[1]}");
        return 0;
    }

    static string F(float v)
    {
        if (float.IsNaN(v)) return "null";
        if (float.IsInfinity(v)) return v > 0 ? "1e308" : "-1e308";
        return v.ToString("R", CultureInfo.InvariantCulture);
    }
}
