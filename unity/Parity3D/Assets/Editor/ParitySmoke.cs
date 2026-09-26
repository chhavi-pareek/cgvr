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
            var table = ParityTable.Build(surrogateNav: true);
            Log($"table rows m = {table.M} (expected 180)");
            if (table.M != 180) { Err($"table has {table.M} rows, not 180"); failures++; }
            int engineRows = ParityTable.Build().M;
            if (engineRows != 135) { Err($"engine table has {engineRows} rows, not 135"); failures++; }

            var size = SceneSpec.Get(SceneKind.Plaza);
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

            // long enough to cross: up to 77 m of hall, or 28 m past the door, at ~1.3 m/s
            // the same plaza crowd again, through the two-salience allocator on the full table
            failures += Scene(SceneKind.Plaza, 2000, 360);
            failures += Scene(SceneKind.Hub, 800, 3600);
            failures += Scene(SceneKind.Corridor, 260, 3600);
            failures += Coverage();
            failures += ViewSide();
            failures += Pipelined(1500, 600);
        }
        catch (Exception ex)
        {
            Err("threw: " + ex);
            failures++;
        }
        Log(failures == 0 ? "SMOKE PASS" : $"SMOKE FAIL ({failures} problems)");
        if (Application.isBatchMode) UnityEditor.EditorApplication.Exit(failures == 0 ? 0 : 1);
    }

    /// <summary>The concourse and the corridor: the same invariants, plus the scene's own
    /// geometry -- nobody inside the partition, nobody outside the hall, and the crowd actually
    /// flowing (served at the window, through the door and out).</summary>
    static int Scene(SceneKind kind, int n, int frames)
    {
        var spec = SceneSpec.Get(kind);
        int failures = 0;
        var table = ParityTable.Build(spec.EMax);
        double floor, baseFloor;
        CrowdWorld.CalibrateAxes(spec, table, 200, out floor);
        var baseTheta = CrowdWorld.MeasureAxes(spec, table, 200, Policy.Baseline, out baseFloor);
        int kept;
        CrowdWorld.PricedAndPruned(table, floor, CrowdWorld.CalibratedTheta, out kept);
        // the full table, so PARITY runs the two-salience allocator the demo defaults to
        var alloc = table;
        var worlds = new[] { new CrowdWorld(Policy.Baseline, n, spec, 1u, alloc), new CrowdWorld(Policy.Parity, n, spec, 1u, alloc) };
        if (worlds[1].Factored == null) { Err($"{spec.Name}: full table did not factorise"); return 1; }
        // cost-matched, as the demo runs by default: PARITY gets exactly the baseline's spend
        worlds[0].ApplyCalibration(baseFloor, baseTheta);
        worlds[1].ApplyCalibration(floor, CrowdWorld.CalibratedTheta);
        int wall = 0, outside = 0, nan = 0, maskViol = 0, overBudget = 0;
        float worstAlloc = 0f;
        var exits = new int[2];
        float r = 0.45f * Mathf.Min(spec.Size.x, spec.Size.y);
        for (int f = 0; f < frames; f++)
        {
            float ang = (f / 2400f) * Mathf.PI * 2f;
            var cam = spec.Focus + new Vector2(r * Mathf.Cos(ang), r * Mathf.Sin(ang));
            float yaw = Mathf.Atan2(spec.Focus.y - cam.y, spec.Focus.x - cam.x);
            for (int k = 0; k < 2; k++)
            {
                var w = worlds[k];
                if (k == 1) w.MatchBudgetMs = worlds[0].PredictedSpendMs();
                w.Step(cam, yaw, w.StepMs);
                for (int i = 0; i < w.N; i++)
                {
                    var p = w.Pos[i];
                    if (float.IsNaN(p.x) || float.IsNaN(p.y)) nan++;
                    if (p.x < -1e-3f || p.y < -1e-3f || p.x > spec.Size.x + 1e-3f || p.y > spec.Size.y + 1e-3f) outside++;
                    if (w.PrevPos[i].x - p.x > 20f) exits[k]++;
                }
                if (kind == SceneKind.Corridor) wall += CorridorScene.WallViolations(w);
                if (w.Mode == Policy.Parity)
                {
                    if (!w.Last.Infeasible && w.Last.Cost > w.BudgetMs * 1.0001f) overBudget++;
                    if (f > 10) worstAlloc = Mathf.Max(worstAlloc, w.AllocMs);
                    for (int i = 0; i < w.N; i++)
                        if (alloc.Err[w.Row[i]] > w.Ledger.Headroom[i] + 1e-4f) { maskViol++; break; }
                }
            }
        }
        var par = worlds[1];
        int queued = 0;
        for (int i = 0; i < par.N; i++) if (par.Slot[i] >= 0) queued++;
        Log($"{spec.Name}: N = {n}, {frames} frames, cap {par.Cap:F2}, pruned to {kept} rows");
        Log($"  baseline  worst divergence {worlds[0].MaxDivergence():F2}   left the scene {exits[0]}");
        Log($"  PARITY    worst divergence {par.MaxDivergence():F2}   left the scene {exits[1]}" +
            $"   restorations {par.Ledger.Restorations}" + (kind == SceneKind.Hub ? $"   queue {queued}" : ""));
        Log($"  two-salience allocator: worst {worstAlloc:F2} ms, last {par.AllocMs:F2} ms of a {par.StepMs:F2} ms step");
        failures += Check($"{spec.Name} NaN positions", nan, 0);
        failures += Check($"{spec.Name} agents outside the walkable area", outside, 0);
        failures += Check($"{spec.Name} agents inside the partition", wall, 0);
        failures += Check($"{spec.Name} budget overruns", overBudget, 0);
        failures += Check($"{spec.Name} error-mask violations", maskViol, 0);
        failures += Check($"{spec.Name} cap breaches (invariant 3)", par.CapBreaches, 0);
        if (par.MaxDivergence() > par.Cap + 1e-3f) { Err($"{spec.Name} PARITY exceeded its cap"); failures++; }
        if (kind != SceneKind.Plaza && (exits[0] == 0 || exits[1] == 0))
        {
            Err($"{spec.Name}: nobody left the scene, the crowd is stuck");
            failures++;
        }
        if (kind == SceneKind.Hub && queued == 0) { Err("hub: the ticket queue emptied and never refilled"); failures++; }
        foreach (var w in worlds) w.Dispose();
        return failures;
    }

    /// <summary>The software occlusion pass: a lone figure at the view-salience distance reads
    /// as about fully visible, and one directly behind another loses pixels to it.</summary>
    static int Coverage()
    {
        int failures = 0;
        var go = new GameObject("smoke cam");
        var cam = go.AddComponent<Camera>();
        cam.fieldOfView = 55f; cam.aspect = 0.9f;
        go.transform.position = new Vector3(0f, 1.6f, -10f);
        go.transform.LookAt(new Vector3(0f, 0.9f, 0f));
        var p = cam.projectionMatrix;
        var vp = p * cam.worldToCameraMatrix;
        var cov = new CoverageBuffer();
        var vis = new float[2];
        float refPx = cov.Compute(new[] { new Vector2(0f, 0f), new Vector2(40f, 0f) }, 2, vp, p.m00, p.m11, 10f, vis);
        Log($"coverage: lone figure at 10 m {vis[0]:F0} px against {refPx:F0} expected; off-screen {vis[1]:F0}");
        if (Mathf.Abs(vis[0] / refPx - 1f) > 0.35f) { Err("coverage: lone figure is not about fully visible"); failures++; }
        if (vis[1] != 0f) { Err("coverage: an off-screen figure has pixels"); failures++; }
        cov.Compute(new[] { new Vector2(0f, 0f), new Vector2(0f, 4f) }, 2, vp, p.m00, p.m11, 10f, vis);
        var alone = new float[1];
        cov.Compute(new[] { new Vector2(0f, 4f) }, 1, vp, p.m00, p.m11, 10f, alone);
        Log($"coverage: figure 4 m behind another keeps {vis[1]:F0} of its {alone[0]:F0} px");
        if (!(vis[1] < 0.5f * alone[0])) { Err("coverage: the figure behind was not occluded"); failures++; }
        UnityEngine.Object.DestroyImmediate(go);
        return failures;
    }

    /// <summary>Two viewers on one crowd, with the pop ledger. This run has no renderer, so
    /// geometry is priced with the engine's measured plaza values injected by hand.</summary>
    static int ViewSide()
    {
        int failures = 0;
        var spec = SceneSpec.Get(SceneKind.Plaza);
        var table = ParityTable.Build(spec.EMax, new[] { 1.0, 0.891, 0.545, 0.457 });
        double floor;
        CrowdWorld.CalibrateAxes(spec, table, 200, out floor);
        var th = CrowdWorld.CalibratedTheta;
        th[3, 0] = 0.0148; th[3, 1] = 0.0038; th[3, 2] = 0.0010;
        int worstFree = 0, worstHeld = 0, differ = 0;
        foreach (bool ledger in new[] { false, true })
        {
            var w = new CrowdWorld(Policy.Parity, 1500, spec, 1u, table)
            {
                Viewers = 2, PopLedger = ledger, SwitchCost = ledger ? 0.12f : 0f, BudgetFrac = 0.25f,
            };
            w.ApplyCalibration(floor, th);
            int maskViol = 0, over = 0;
            for (int f = 0; f < 600; f++)
            {
                float a1 = f / 2400f * Mathf.PI * 2f, a2 = a1 + Mathf.PI;
                var c1 = spec.Focus + 54f * new Vector2(Mathf.Cos(a1), Mathf.Sin(a1));
                var c2 = spec.Focus + 54f * new Vector2(Mathf.Cos(a2), Mathf.Sin(a2));
                w.Cam2 = c2; w.Yaw2 = Mathf.Atan2(spec.Focus.y - c2.y, spec.Focus.x - c2.x);
                w.Step(c1, Mathf.Atan2(spec.Focus.y - c1.y, spec.Focus.x - c1.x), w.StepMs);
                if (!w.Last.Infeasible && w.Last.Cost > w.BudgetMs * 1.0001f) over++;
                for (int i = 0; i < w.N; i++)
                    if (table.Err[w.Row[i]] > w.Ledger.Headroom[i] + 1e-4f) { maskViol++; break; }
            }
            int diffRun = 0;
            for (int i = 0; i < w.N; i++) if (w.GeoV[0][i] != w.GeoV[1][i]) diffRun++;
            differ += diffRun;
            int worst = Mathf.Max(w.Pops[0].WorstWindow, w.Pops[1].WorstWindow);
            if (ledger) worstHeld = worst; else worstFree = worst;
            Log($"two viewers, pops {(ledger ? "bounded + cost" : "free")}: viewer pops/agent-min " +
                $"{w.Pops[0].PerAgentMinute:F1} / {w.Pops[1].PerAgentMinute:F1} (area-weighted " +
                $"{w.Pops[0].WeightedPerAgentMinute:F2}), worst agent in 2 s {worst}, " +
                $"holds released {w.HoldsReleased}, agents drawn differently by the two viewers {diffRun}, " +
                $"allocator {w.AllocMs:F2} ms");
            failures += Check($"two viewers ({(ledger ? "bounded" : "free")}) budget overruns", over, 0);
            failures += Check($"two viewers ({(ledger ? "bounded" : "free")}) error-mask violations", maskViol, 0);
            float bound = w.Pops[0].Capacity + w.Pops[0].Refill * w.Pops[0].Window;
            if (ledger && w.HoldsReleased == 0 && worst > bound + 1e-4f)
            {
                Err($"pop ledger: an agent popped {worst} times in 2 s against a bound of {bound:F1}");
                failures++;
            }
            w.Dispose();
        }
        if (differ == 0) { Err("the two viewers were never given different detail"); failures++; }
        Log($"pop ledger: worst agent in any 2 s {worstFree} free -> {worstHeld} bounded");
        return failures;
    }

    /// <summary>Decisions on a worker thread, a frame ahead, under a frame-time target: the same
    /// invariants as the serial path, checked with the worker joined.</summary>
    static int Pipelined(int n, int frames)
    {
        var spec = SceneSpec.Get(SceneKind.Plaza);
        int failures = 0;
        var table = ParityTable.Build(spec.EMax);
        double floor, baseFloor;
        CrowdWorld.CalibrateAxes(spec, table, 200, out floor);
        var baseTheta = CrowdWorld.MeasureAxes(spec, table, 200, Policy.Baseline, out baseFloor);
        var worlds = new[] { new CrowdWorld(Policy.Baseline, n, spec, 1u, table) { Pipelined = true },
                             new CrowdWorld(Policy.Parity, n, spec, 1u, table) { Pipelined = true } };
        worlds[0].ApplyCalibration(baseFloor, baseTheta);
        worlds[1].ApplyCalibration(floor, CrowdWorld.CalibratedTheta);
        var par = worlds[1];
        par.FrameTargetMs = 4.5f;
        int nan = 0, overBudget = 0, baseLive = 0;
        long parSur = 0;
        float peak = 0f;
        float r = 0.45f * Mathf.Min(spec.Size.x, spec.Size.y);
        for (int f = 0; f < frames; f++)
        {
            float ang = (f / 2400f) * Mathf.PI * 2f;
            var cam = spec.Focus + new Vector2(r * Mathf.Cos(ang), r * Mathf.Sin(ang));
            float yaw = Mathf.Atan2(spec.Focus.y - cam.y, spec.Focus.x - cam.x);
            foreach (var w in worlds)
            {
                // a synthetic frame: the step plus a fixed render share
                w.Step(cam, yaw, w.StepMs + 4f);
                w.Join();
                for (int i = 0; i < w.N; i++)
                    if (float.IsNaN(w.Pos[i].x) || float.IsNaN(w.Pos[i].y)) nan++;
            }
            for (int i = 0; i < n; i++) { if (worlds[0].Beh[i] < 3) baseLive++; if (par.Beh[i] == 3) parSur++; }
            if (!par.Last.Infeasible && par.Last.Cost > par.BudgetMs * 1.0001f) overBudget++;
            peak = Mathf.Max(peak, par.MaxDivergence());
        }
        Log($"pipelined: N = {n}, {frames} frames, PARITY worst divergence {peak:F2} of cap {par.Cap:F2}," +
            $" surrogates {parSur / frames}/frame, restorations {par.Ledger.Restorations}, blocked {par.Prof[11] / frames:F3} ms/frame; baseline live {baseLive / frames}/frame");
        failures += Check("pipelined NaN positions", nan, 0);
        failures += Check("pipelined budget overruns", overBudget, 0);
        failures += Check("pipelined cap breaches (invariant 3)", par.CapBreaches, 0);
        if (peak > par.Cap + 1e-3f) { Err("pipelined PARITY exceeded its cap"); failures++; }
        if (parSur == 0 || par.Ledger.Restorations <= n) { Err("pipelined PARITY never had to restore anyone: the ledger went untested"); failures++; }
        if (baseLive == 0) { Err("pipelined baseline assigned no live tier"); failures++; }
        foreach (var w in worlds) w.Dispose();
        return failures;
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
