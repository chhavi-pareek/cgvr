// Headless smoke test: steps both crowds without rendering, so the sim layer is exercised
// before anyone presses Play. Run it from the terminal via tools/sync_unity.sh --smoke:
//
//   Unity -batchmode -quit -nographics -projectPath <p> -executeMethod ParitySmoke.Run
//
// Checks the things that must hold by construction rather than by tuning: the frame budget is
// respected, the error mask is never violated, and no agent's ledger passes the cap.
using System;
using UnityEngine;
using Parity;

public static class ParitySmoke
{
    // MassLOD's per-level caps are 80 / 400 / 1200, so below ~1700 agents nothing is ever
    // forced to the surrogate and the baseline has nothing to diverge. The comparison is
    // only meaningful where both policies are actually under pressure.
    const int N = 2000, Frames = 360;

    public static void Run()
    {
        int failures = 0;
        try
        {
            var table = ParityTable.Build();
            Log($"table rows m = {table.M} (expected 180)");
            if (table.M != 180) { Err($"table has {table.M} rows, not 180"); failures++; }

            var size = new Vector2(120f, 120f);
            double floor;
            CrowdWorld.CalibrateAxes(size, table, 400, out floor);
            var theta = CrowdWorld.CalibratedTheta;
            Log($"calibrated floor (all tier 3) {floor * 1000:F3} us/agent/frame");
            for (int ax = 0; ax < ParityTable.NAxes; ax++)
                Log($"    {ParityTable.AxisNames[ax],-11} t0 {theta[ax, 0] * 1000:F3}" +
                    $"  t1 {theta[ax, 1] * 1000:F3}  t2 {theta[ax, 2] * 1000:F3}  t3 0   us/agent");

            int kept;
            var alloc = CrowdWorld.PricedAndPruned(table, floor, theta, out kept);
            Log($"pruned {table.M} -> {kept} rows   rate-0 {alloc.RateZeroRows()}" +
                $"   surrogate {alloc.SurrogateRows()}   quality {alloc.Quality[0]:F4}..");
            if (alloc.RateZeroRows() == 0) { Err("prune removed every rate-0 row: the mask can empty"); failures++; }
            if (alloc.SurrogateRows() == 0) { Err("prune removed every surrogate row: the ledger cannot engage"); failures++; }

            var bas = new CrowdWorld(Policy.Baseline, N, size, 1u, alloc);
            var par = new CrowdWorld(Policy.Parity, N, size, 1u, alloc);
            bas.ApplyCalibration(floor, theta);
            par.ApplyCalibration(floor, theta);
            for (int ax = 0; ax < 3; ax++)
                if (theta[ax, 0] <= theta[ax, 2])
                    Log($"  NOTE {ParityTable.AxisNames[ax]} is not monotone in cost across tiers");

            float lastB = 0f, lastP = 0f, worstAlloc = 0f;
            int overBudget = 0, maskViol = 0, infeasible = 0;
            for (int f = 0; f < Frames; f++)
            {
                float ang = (f / 1200f) * Mathf.PI * 2f;
                var cam = new Vector2(60f + 54f * Mathf.Cos(ang), 60f + 54f * Mathf.Sin(ang));
                float yaw = Mathf.Atan2(60f - cam.y, 60f - cam.x);

                bas.Step(cam, yaw, lastB); lastB = bas.StepMs;
                par.Step(cam, yaw, lastP); lastP = par.StepMs;

                if (par.AllocMs > worstAlloc) worstAlloc = par.AllocMs;
                if (par.Last.Infeasible) infeasible++;
                else if (par.Last.Cost > par.BudgetMs * 1.0001f) overBudget++;
                for (int i = 0; i < N; i++)
                    if (table.Err[par.Row[i]] > par.Ledger.Headroom[i] + 1e-4f) { maskViol++; break; }
            }

            Log($"stepped {Frames} frames at N = {N}");
            Log($"  baseline  step {bas.StepMs:F2} ms   worst-agent divergence {bas.MaxDivergence():F1} nats");
            Log($"  PARITY    step {par.StepMs:F2} ms   worst-agent divergence {par.MaxDivergence():F2} nats" +
                $"   cap {par.Cap:F2}");
            Log($"  budget {par.BudgetMs:F2} ms   spent {par.Last.Cost:F2} ms" +
                $"   lambda {par.Last.Lambda:G4}   evals {par.Last.Evals}");
            Log("  (geometry prices at 0 here: it costs render time, and this run has no renderer)");
            Log($"  allocator worst {worstAlloc:F2} ms   lambda {par.Last.Lambda:G4}   evals {par.Last.Evals}" +
                $"   fill {par.Last.FillSteps}");
            Log($"  restorations {par.Ledger.Restorations}   infeasible frames {infeasible}");
            Log($"  allocator share of PARITY's step: {100 * par.AllocMs / Mathf.Max(par.StepMs, 1e-6f):F0}%" +
                "  (serial managed here; Unity spreads the jobs over the worker threads)");

            failures += Check("budget overruns", overBudget, 0);
            failures += Check("error-mask violations", maskViol, 0);
            failures += Check("cap breaches (invariant 3)", par.CapBreaches, 0);
            failures += Check("starved agents (no feasible row)", par.Last.Starved, 0);
            if (par.Ledger.Restorations == 0) { Err("no restorations happened -- the ledger is not biting"); failures++; }
            if (par.Last.Lambda <= 0f && par.Last.Evals <= 1)
            {
                Err("the budget never bound (lambda 0, 1 eval): the cost model sees every row as " +
                    "equal, so the allocator has nothing to decide");
                failures++;
            }
            // The claim is not "PARITY diverges less" -- it is that PARITY's divergence is
            // BOUNDED and the baseline's is not. MassLOD has no ledger, so an agent parked at
            // the lowest LOD accrues for as long as it stays there.
            if (bas.MaxDivergence() <= par.Cap)
            {
                Err($"baseline divergence {bas.MaxDivergence():F2} never passed the cap {par.Cap:F2}, " +
                    "so this run does not show the baseline being unbounded -- raise N or Frames");
                failures++;
            }
            if (par.MaxDivergence() > par.Cap + 1e-3f)
            {
                Err($"PARITY divergence {par.MaxDivergence():F2} exceeded its own cap {par.Cap:F2}");
                failures++;
            }
            bas.Dispose(); par.Dispose();
        }
        catch (Exception ex)
        {
            Err("threw: " + ex);
            failures++;
        }
        Log(failures == 0 ? "SMOKE PASS" : $"SMOKE FAIL ({failures} problems)");
        if (Application.isBatchMode) UnityEditor.EditorApplication.Exit(failures == 0 ? 0 : 1);
    }

    static int Check(string what, int got, int want)
    {
        if (got == want) { Log($"  OK   {what}: {got}"); return 0; }
        Err($"  FAIL {what}: {got}, expected {want}");
        return 1;
    }

    static void Log(string s) { Debug.Log("[ParitySmoke] " + s); }
    static void Err(string s) { Debug.LogError("[ParitySmoke] " + s); }
}
