// Frame-time benchmark, headless on the GPU: the proposal's H1 in the engine.
//
//   Unity -batchmode -quit -projectPath <p> -executeMethod ParityFrameBench.Run
//
// Every frame is timed end to end -- crowd step, allocator, draw submission, render, and a
// 1-pixel readback that waits for the GPU -- for the MassLOD baseline with its shipped parameters
// and for PARITY given nothing but a target frame time (CrowdWorld.FrameTargetMs). Each policy
// runs alone at each crowd size, so the timings do not share a frame. Writes
// Logs/frame_bench.csv and logs one line per cell.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using Parity;

public static class ParityFrameBench
{
    const int Warm = 120, Measure = 300;
    static bool Pipelined = Array.IndexOf(Environment.GetCommandLineArgs(), "-serial") < 0;
    // -overlap: the CPU runs a frame ahead of the GPU, as an engine does -- each frame waits only
    // for the PREVIOUS frame's GPU work (an async readback of a resolved 4x4 copy), so the frame
    // time is about max(CPU, GPU) instead of their sum
    static bool Overlap = Array.IndexOf(Environment.GetCommandLineArgs(), "-overlap") >= 0;

    public sealed class Profile
    {
        public string Name; public int W, H, Msaa; public bool Shadows;
    }
    public static readonly Profile Desktop = new Profile { Name = "desktop", W = 1920, H = 1080, Msaa = 4, Shadows = true };
    public static readonly Profile Low = new Profile { Name = "low", W = 1280, H = 720, Msaa = 1, Shadows = false };

    static string Arg(string key)
    {
        var a = Environment.GetCommandLineArgs();
        int i = Array.IndexOf(a, key);
        return i >= 0 && i + 1 < a.Length ? a[i + 1] : null;
    }

    public static void Run()
    {
        var prof = Arg("-profile") == "low" ? Low : Desktop;
        var sizes = Arg("-sizes") is string z ? Array.ConvertAll(z.Split(','), int.Parse) : new[] { 1000, 4000, 8000 };
        Log($"code optimization {UnityEditor.Compilation.CompilationPipeline.codeOptimization}, decisions {(Pipelined ? "pipelined" : "serial")}");
        int failures = 0;
        var sb = new StringBuilder("profile,scene,policy,knob,n,ft_mean,ft_p50,ft_p95,ft_p99,ft_std,step_ms,alloc_ms,worst_over_cap,quality,quality_view,quality_perceived,quality_screen,surrogates,restorations,geo0,geo1,geo2,geo3,cpu_ms,gpu_wait_ms,quality_image,pop_image,quality_state,option_mix,thermo_ms\n");
        try
        {
            // quality against frame time: PARITY at several target frame times, MassLOD at several
            // scalings of its per-level caps (the knob a developer turns to hit a frame time)
            // -profile low: the same sweep on the low profile (portability); -sizes a,b: crowd sizes
            if (Array.IndexOf(Environment.GetCommandLineArgs(), "-profileOnly") >= 0)
                Sweep(SceneKind.Plaza, Desktop, sb, new[] { 8000 }, new[] { 33.3f }, new[] { 1f }, only: true);
            else
                Sweep(SceneKind.Plaza, prof, sb, sizes,
                      Arg("-targets") is string tg ? Array.ConvertAll(tg.Split(','), float.Parse)
                                                   : new[] { 12f, 14f, 17f, 20f, 21.5f, 23f, 26f, 30f, 33.3f, 36f, 40f },
                      new[] { 0.25f, 0.5f, 1f, 2f, 4f, 8f });
        }
        catch (Exception ex) { Debug.LogError("[FrameBench] threw: " + ex); failures++; }
        var path = Path.GetFullPath(Path.Combine(Application.dataPath, prof == Desktop ? "../Logs/frame_bench.csv" : $"../Logs/frame_bench_{prof.Name}.csv"));
        File.WriteAllText(path, sb.ToString());
        Log("wrote " + path);
        if (Application.isBatchMode) UnityEditor.EditorApplication.Exit(failures == 0 ? 0 : 1);
    }

    /// <summary>One scene under one profile: calibrate once, then every size under both policies.
    /// `massLod` overrides the baseline's parameters (for the portability study); null = shipped.</summary>
    public static void Sweep(SceneKind kind, Profile prof, StringBuilder sb, int[] sizes, float[] targets, float[] capScales,
                             bool only = false)
    {
        var spec = SceneSpec.Get(kind);
        // -scale s: render at s x the profile's resolution and upscale; -shadows off. Both are
        // global render settings, applied to the calibration as well as to the timed frames.
        float scale = Arg("-scale") is string sc ? float.Parse(sc, CultureInfo.InvariantCulture) : 1f;
        bool shadows = prof.Shadows && Arg("-shadows") != "off";
        string tag = prof.Name + (scale < 1f ? $"@{scale:0.##}" : "") + (shadows ? "" : "-noshadow");
        Director.ApplyQuality(shadows);
        QualitySettings.antiAliasing = prof.Msaa;
        var set = Stage.Build(spec, out var sun);
        sun.shadows = shadows ? LightShadows.Soft : LightShadows.None;
        var rend = new CrowdRenderer(Shader.Find("Parity/CrowdInstanced"), Shader.Find("Parity/CrowdImpostor"),
                                     Shader.Find("Parity/CrowdDecal")) { Shadows = shadows };
        var cam = Director.MakeCamera("Bench", new Rect(0f, 0f, 1f, 1f));
        UnityEngine.Object.DestroyImmediate(cam.GetComponent<PostFx>());
        cam.enabled = false;
        var rt = new RenderTexture(Mathf.RoundToInt(prof.W * scale), Mathf.RoundToInt(prof.H * scale), 24, RenderTextureFormat.DefaultHDR) { antiAliasing = prof.Msaa };
        var full = scale < 1f ? new RenderTexture(prof.W, prof.H, 0, RenderTextureFormat.DefaultHDR) : null;
        var shown = full ?? rt;
        cam.targetTexture = rt;
        var probe = new Texture2D(1, 1, TextureFormat.RGBA32, false);
        var shot = new Texture2D(prof.W, prof.H, TextureFormat.RGBA32, false);
        var tiny = new[] { new RenderTexture(4, 4, 0), new RenderTexture(4, 4, 0) };
        var camRec = UnityEngine.Profiling.Recorder.Get("Camera.Render");
        camRec.enabled = true;
        Log($"global settings {tag}: {rt.width}x{rt.height}, shadows {(shadows ? "on" : "off")}");
        Log($"frames {(Overlap ? "overlapped (CPU one frame ahead)" : "serial (GPU waited on every frame)")}; GPU recorder supported {SystemInfo.supportsGpuRecorder}");

        // calibrate exactly as the demo does: sim axes per policy, geometry by render time, the
        // pixel judge for mesh quality
        var table = ParityTable.Build(spec.EMax);
        double floor, baseFloor;
        CrowdWorld.CalibrateAxes(spec, table, 400, out floor);
        var baseTheta = CrowdWorld.MeasureAxes(spec, table, 400, Policy.Baseline, out baseFloor);
        Orbit(cam, spec, 0f, out _, out _);
        var cw = new CrowdWorld(Policy.Parity, 400, spec, 99u, table);
        var geo = rend.MeasureGeometry(cw, cam);
        cw.Dispose();
        for (int t = 0; t < 3; t++)
            CrowdWorld.CalibratedTheta[3, t] = baseTheta[3, t] = Math.Max(geo[t] - geo[3], 0.0);
        var gcpu = rend.GeometryCpu;
        for (int t = 0; t < ParityTable.NTiers; t++)
            CrowdWorld.GeoGpuTheta[t] = t == 3 ? 0.0
                : Math.Min(Math.Max((geo[t] - gcpu[t]) - (geo[3] - gcpu[3]), 0.0), CrowdWorld.CalibratedTheta[3, t]);
        Log($"geometry us per drawn agent: {geo[0] * 1000:F2} {geo[1] * 1000:F2} {geo[2] * 1000:F2} {geo[3] * 1000:F2} (drawn {rend.Drawn} of 400)" +
            $"; of which CPU submit {gcpu[0] * 1000:F2} {gcpu[1] * 1000:F2} {gcpu[2] * 1000:F2} {gcpu[3] * 1000:F2}");
        var jw = new CrowdWorld(Policy.Parity, 12, spec, 5u, table);
        var gq = Director.MeasureQuality(rend, jw, cam, spec);
        var vg = Director.MeasureViewGrid(rend, jw, cam, spec);
        jw.Dispose();
        for (int a = 0; a < ParityTable.NTiers; a++)
        {
            var row = new StringBuilder($"view pixel grid anim {a}:");
            for (int g = 0; g < ParityTable.NTiers; g++)
                row.Append($" {vg[a, g]:F3} (label {(ParityTable.AxisQuality[2, a] + gq[g]) / 2.0:F3})");
            Log(row.ToString());
        }
        table = ParityTable.Build(spec.EMax, gq);

        // B: global shadow settings, each measured against option 0 (full shadows) -- frame cost
        // by render time, geometry quality and the scene's own loss by the pixel judge
        CrowdWorld.GlobalOption[] options = null;
        double[][] rawOptTheta = null;
        Action<int> applyOption = o => ApplyOption(o, sun, rend);
        if (shadows)
        {
            options = CalibrateOptions(spec, table, rend, cam, applyOption, probe, shot, rt, shown);
            rawOptTheta = Array.ConvertAll(options, op => (double[])op.GeoTheta.Clone());
            foreach (var op in options)
                Log($"option {op.Name}: fixed {op.FixedMs:+0.00;-0.00} ms + {op.DrawnMs * 1000:+0.00;-0.00} us per drawn agent;" +
                    $" geometry us {op.GeoTheta[0] * 1000:F2}/{op.GeoTheta[1] * 1000:F2}/{op.GeoTheta[2] * 1000:F2}," +
                    $" quality {op.GeoQuality[0]:F3}/{op.GeoQuality[1]:F3}/{op.GeoQuality[2]:F3}/{op.GeoQuality[3]:F3}, scene loss {op.SceneLoss:F1}");
        }

        // masslod: UE5 MassLOD, knob = cap scale. timeslice: uniform staggered AI time slicing,
        // knob = update period. knapsack: Funkhouser-Sequin render knapsack on the frame target
        // over MassLOD's simulation levels. knapsack_fullsim: the same over full simulation for
        // everyone (LOD for rendering only). parity_nocap: PARITY without its ledger (ablation).
        var cells = new List<(string name, Policy pol, float knob)>();
        foreach (var k in capScales) cells.Add(("masslod", Policy.Baseline, k));
        if (!only)
        {
            foreach (var k in new[] { 1f, 3f, 10f }) cells.Add(("timeslice", Policy.Baseline, k));
            foreach (var k in Arg("-targets") != null ? targets : new[] { 12f, 14f, 17f, 20f, 26f, 33.3f, 40f })
            {
                cells.Add(("knapsack", Policy.Baseline, k));
                cells.Add(("knapsack_fullsim", Policy.Baseline, k));
            }
        }
        foreach (var k in targets) cells.Add(("parity", Policy.Parity, k));
        if (Overlap) foreach (var k in targets) cells.Add(("parity_dual", Policy.Parity, k));
        if (!only) foreach (var k in new[] { 14f, 20f, 26f, 33.3f }) cells.Add(("parity_nocap", Policy.Parity, k));
        // B: parity_b picks the shadow setting every frame; knapsack_fullsim_opt<k> holds setting k;
        // knapsack_fullsim_scaler moves it with frame time, as an engine's scalability system does
        if (!only && shadows)
            foreach (var bn in new[] { "parity_b", "parity_opt2", "knapsack_fullsim_opt1", "knapsack_fullsim_opt2", "knapsack_fullsim_scaler" })
                foreach (var k in targets) cells.Add((bn, bn.StartsWith("parity") ? Policy.Parity : Policy.Baseline, k));
        if (!only)
            foreach (var abl in new[] { "parity_free", "parity_area", "parity_area_free", "parity_nohold", "parity_area_nohold",
                                        "parity_holdonly", "parity_sw0", "parity_sw0.005", "parity_sw0.01", "parity_sw0.02", "parity_sw0.04",
                                        "parity_sw0_gain0.02", "parity_sw0_gain0.005",
                                        "parity_sw0_vs0.1", "parity_sw0_vs0.05", "parity_sw0_vs0.02", "parity_area_sw0_vs0.1" })
                foreach (var k in targets) cells.Add((abl, Policy.Parity, k));
        // -policies a,b: only these
        if (Arg("-policies") is string only_) cells.RemoveAll(c => Array.IndexOf(only_.Split(','), c.name) < 0);
        // A fanless machine throttles within minutes of sustained load, so cells run in rounds --
        // one per policy per round, in an order shuffled by -seed, knobs descending on odd seeds --
        // and each cell records a fixed-workload thermometer beside its timings
        int seed = Arg("-seed") is string sd ? int.Parse(sd) : 0;
        cells = Interleave(cells, seed);
        Log($"cell order seed {seed}: {cells.Count} cells in rounds");
        foreach (int n in sizes)
        {
            // Invariant 1 at the conditions it is used in: the simulation axes are re-measured at
            // each crowd size, for both policies' cost models. Separation is nearly free in the
            // near-empty plaza of N = 400 and the dominant cost at N = 8000; priced from the
            // former, navigation reads as free and no configuration trades perception for cost.
            CrowdWorld.CalibrateAxes(spec, table, n, out floor);
            baseTheta = CrowdWorld.MeasureAxes(spec, table, n, Policy.Baseline, out baseFloor);
            for (int t = 0; t < 3; t++)
                CrowdWorld.CalibratedTheta[3, t] = baseTheta[3, t] = Math.Max(geo[t] - geo[3], 0.0);
            if (options != null)
            {
                AnchorOptions(options, rawOptTheta, gq, spec, table, n, rend, cam, applyOption, probe, rt);
                Log($"N={n} option frame cost over full shadows (all impostors): " +
                    string.Join(", ", Array.ConvertAll(options, op => $"{op.Name} {op.FixedMs:+0.00;-0.00} ms")));
            }
            var ct = CrowdWorld.CalibratedTheta;
            Log($"N={n} calibrated us/agent: behaviour {ct[0, 0] * 1000:F2}/{ct[0, 1] * 1000:F2}/{ct[0, 2] * 1000:F2}" +
                $"  navigation {ct[1, 0] * 1000:F2}/{ct[1, 1] * 1000:F2}/{ct[1, 2] * 1000:F2}" +
                $"  animation {ct[2, 0] * 1000:F2}/{ct[2, 1] * 1000:F2}/{ct[2, 2] * 1000:F2}  floor {floor * 1000:F2}");
            foreach (var (name, pol, knob) in cells)
            {
                // the baseline's measured divergence is an instrument, kept out of the timed frame
                double thermo = Thermometer();
                var w = new CrowdWorld(pol, n, spec, 1u, table) { Pipelined = Pipelined, Instrumented = false };
                if (name == "masslod")
                {
                    w.ApplyCalibration(baseFloor, baseTheta);
                    for (int l = 0; l < 3; l++)
                    {
                        w.MassLod.VisCaps[l] = Mathf.Max(1, Mathf.RoundToInt(w.MassLod.VisCaps[l] * knob));
                        w.MassLod.SimCaps[l] = Mathf.Max(1, Mathf.RoundToInt(w.MassLod.SimCaps[l] * knob));
                    }
                }
                else if (name == "timeslice") { w.ApplyCalibration(baseFloor, baseTheta); w.TimeSlice = (int)knob; }
                else if (name.StartsWith("knapsack"))
                {
                    w.ApplyCalibration(baseFloor, baseTheta);
                    w.RenderKnapsack = true; w.FrameTargetMs = knob;
                    if (name.StartsWith("knapsack_fullsim")) w.TimeSlice = 1;
                    w.PopLedger = false; w.SwitchCost = 0f;      // neither is Funkhouser's
                }
                else
                {
                    w.ApplyCalibration(floor, CrowdWorld.CalibratedTheta); w.FrameTargetMs = knob;
                    w.Uncapped = name == "parity_nocap";
                    w.DualResource = name == "parity_dual";
                    // ablations: _free drops pop bounding (ledger, switching cost, salience
                    // smoothing); _area replaces the coverage buffer with the (d0 / d)^2 area model
                    if (name.Contains("_free")) { w.PopLedger = false; w.SwitchCost = 0f; w.ViewSmoothing = 1f; }
                    if (name.Contains("_nohold")) w.PopLedger = false;
                    // _sw<c>: switching cost c and no pop ledger; _holdonly: the ledger, no switching cost
                    if (Token(name, "_sw") is float swc) { w.PopLedger = false; w.SwitchCost = swc; }
                    if (name.Contains("_holdonly")) w.SwitchCost = 0f;
                    if (Token(name, "_gain") is float gain) w.FrameGain = gain;
                    if (Token(name, "_vs") is float vsm) w.ViewSmoothing = vsm;
                    if (name.Contains("_area")) w.Occlusion = false;
                }
                bool scaler = name.EndsWith("_scaler");
                if (options != null && (name.Contains("_b") || name.Contains("_opt") || scaler))
                {
                    w.Options = options;
                    w.OptionExternal = !name.Contains("_b");
                    w.Option = Token(name, "_opt") is float ok ? (int)ok : 0;
                }
                float scEma = 0f; int scOver = 0, scUnder = 0, prevOption = 0, applied = 0;
                var optFrames = new long[options?.Length ?? 1];
                double stepMs = 0, allocMs = 0, cpuMs = 0, waitMs = 0, gpuRec = 0;
                var pend = default(UnityEngine.Rendering.AsyncGPUReadbackRequest);
                bool hasPend = false;
                float lastCpu = -1f, lastGpu = -1f, prevDone = 0f;
                bool censored = false;
                var ft = new List<float>(Measure);
                float last = 0f;
                double qSum = 0, qView = 0, qPer = 0, qScr = 0, aScr = 0, qSt = 0; long qN = 0, qvN = 0;
                var snaps = new List<Snap>();
                sbyte[] prevGeo = null, prevAnim = null;
                for (int f = 0; f < Warm + Measure; f++)
                {
                    float t0 = Time.realtimeSinceStartup;
                    Orbit(cam, spec, f / 2400f, out var xz, out float yaw);
                    var pm = cam.projectionMatrix;
                    w.SetView(0, pm * cam.worldToCameraMatrix, pm.m00, pm.m11);
                    w.MeasuredCpuMs = lastCpu; w.MeasuredGpuMs = lastGpu; w.GpuCensored = censored;
                    w.Step(xz, yaw, last);
                    if (w.Options != null && w.Option != applied) { applyOption(w.Option); applied = w.Option; }
                    rend.Draw(w, cam);
                    cam.Render();
                    if (full != null) Graphics.Blit(rt, full);
                    float tc = Time.realtimeSinceStartup;
                    if (Overlap)
                    {
                        Graphics.Blit(shown, tiny[f & 1]);
                        var req = UnityEngine.Rendering.AsyncGPUReadback.Request(tiny[f & 1]);
                        if (hasPend) pend.WaitForCompletion();              // the previous frame's GPU work
                        pend = req; hasPend = true;
                    }
                    else
                    {
                        var prev = RenderTexture.active;
                        RenderTexture.active = shown;
                        probe.ReadPixels(new Rect(0, 0, 1, 1), 0, 0, false);      // wait for the GPU
                        RenderTexture.active = prev;
                    }
                    float tw = Time.realtimeSinceStartup;
                    last = (tw - t0) * 1000f;
                    lastCpu = (tc - t0) * 1000f;
                    // GPU time: the GPU timer where the platform has one. Else, overlapped, the
                    // previous frame's work could start no earlier than the one before it was
                    // seen done, and was seen done now: exact when this frame had to wait for it,
                    // an upper bound (censored) when it was already done
                    double recMs = camRec.gpuElapsedNanoseconds * 1e-6;
                    lastGpu = recMs > 0 ? (float)recMs : !Overlap ? (tw - tc) * 1000f : f > 0 ? (tw - prevDone) * 1000f : -1f;
                    censored = recMs <= 0 && Overlap && (tw - tc) * 1000f < 0.2f;
                    prevDone = tw;
                    if (f >= Warm)
                    {
                        cpuMs += (tc - t0) * 1000.0; waitMs += (tw - tc) * 1000.0;
                        gpuRec += camRec.gpuElapsedNanoseconds * 1e-6;
                    }
                    w.Instrument();
                    if (scaler)
                    {
                        // an engine scalability rule: cheaper setting after 30 frames over target,
                        // richer after 60 frames with 20% headroom
                        scEma = scEma <= 0f ? last : scEma + 0.1f * (last - scEma);
                        scOver = scEma > knob ? scOver + 1 : 0;
                        scUnder = scEma < 0.8f * knob ? scUnder + 1 : 0;
                        if (scOver >= 30 && w.Option < options.Length - 1) { w.Option++; scOver = scUnder = 0; }
                        else if (scUnder >= 60 && w.Option > 0) { w.Option--; scOver = scUnder = 0; }
                    }
                    if (f >= Warm) optFrames[w.Options != null ? w.Option : 0]++;
                    if (f >= Warm)
                    {
                        ft.Add(last);
                        stepMs += w.StepMs; allocMs += w.AllocMs;
                        for (int i = 0; i < n; i++)
                        {
                            double q = 0;
                            q += table.AxisQ[0, w.Beh[i]] + table.AxisQ[1, NavRun(w, i)] + table.AxisQ[2, AnimRun(w, i)] + table.AxisQ[3, w.GeoV[0][i]];
                            q /= ParityTable.NAxes;
                            qSum += q; qN++;
                            if (w.InView[i]) { qView += q; qvN++; }
                            // perceived: an agent off screen is judged on behaviour and navigation only
                            double st = table.AxisQ[0, w.Beh[i]] + table.AxisQ[1, NavRun(w, i)];
                            qSt += st / 2.0;
                            qPer += w.InView[i] ? (st + table.AxisQ[2, AnimRun(w, i)] + table.AxisQ[3, w.GeoV[0][i]]) / 4.0 : st / 2.0;
                            // screen: what the viewer sees, each on-screen agent weighted by its
                            // projected area (d0 / d)^2 -- Funkhouser's benefit, nobody's objective
                            if (w.InView[i])
                            {
                                float r = w.ViewD0 / Mathf.Max(w.Sig[i], 1e-3f);
                                double a = Mathf.Min(1f, r * r);
                                qScr += a * q; aScr += a;
                            }
                        }
                        if ((f - Warm) % ImageEvery == 0)
                        {
                            var sn = Snap.Of(w, f, prevGeo, prevAnim);
                            sn.Option = w.Options != null ? w.Option : 0; sn.PrevOption = prevOption;
                            snaps.Add(sn);
                        }
                    }
                    if (f + 1 >= Warm && (f + 1 - Warm) % ImageEvery == 0)
                    {
                        prevGeo = (sbyte[])w.GeoV[0].Clone(); prevAnim = (sbyte[])w.AnimV[0].Clone();
                        prevOption = w.Options != null ? w.Option : 0;
                    }
                }
                w.Join();
                if (hasPend) pend.WaitForCompletion();
                double img = ImageQuality(w, rend, cam, spec, snaps, shot, rt, shown, options != null ? applyOption : null, out double pop);
                if (options != null) applyOption(0);
                string mix = string.Join("/", Array.ConvertAll(optFrames, c => (c / (double)Measure).ToString("F2", CultureInfo.InvariantCulture)));
                ft.Sort();
                float mean = 0; foreach (var v in ft) mean += v; mean /= ft.Count;
                float var_ = 0; foreach (var v in ft) var_ += (v - mean) * (v - mean);
                float std = Mathf.Sqrt(var_ / ft.Count);
                float P(float q) => ft[Mathf.Clamp((int)(q * ft.Count), 0, ft.Count - 1)];
                int sur = 0; var g = new int[4];
                for (int i = 0; i < n; i++) { if (w.Beh[i] == 3) sur++; g[w.GeoV[0][i]]++; }
                float worst = w.MaxDivergence();
                sb.Append(string.Join(",", tag, spec.Name, name, F(knob), n.ToString(), F(mean), F(P(0.5f)), F(P(0.95f)),
                                      F(P(0.99f)), F(std), F((float)(stepMs / Measure)), F((float)(allocMs / Measure)), F(worst / w.Cap),
                                      F((float)(qSum / Math.Max(qN, 1))), F((float)(qView / Math.Max(qvN, 1))), F((float)(qPer / Math.Max(qN, 1))), F((float)(qScr / Math.Max(aScr, 1e-9))),
                                      sur.ToString(), w.Restorations.ToString(), g[0].ToString(), g[1].ToString(),
                                      g[2].ToString(), g[3].ToString(),
                                      F((float)(cpuMs / Measure)), F((float)(waitMs / Measure)), F((float)img), F((float)pop), F((float)(qSt / Math.Max(qN, 1))), mix, F((float)thermo))).Append('\n');
                if (n >= 8000 && (knob == 1f || knob == 33.3f))
                {
                    var parts = new StringBuilder();
                    for (int k = 0; k < w.Prof.Length; k++)
                        parts.Append($" {CrowdWorld.ProfNames[k]} {w.Prof[k] / (Warm + Measure):F2}");
                    Log($"  profile {name} N={n} ms/frame:{parts} evals {(double)w.Evals / (Warm + Measure):F1}");
                    Log($"  frame split: cpu (step+draw+submit) {cpuMs / Measure:F2}  wait for gpu {waitMs / Measure:F2}  gpu recorder (Camera.Render) {gpuRec / Measure:F2} ms");
                    Log("  coverage project/sort/raster ms/frame: " + string.Join(" ", Array.ConvertAll(CoverageBuffer.Ticks, t => (t * 1000.0 / System.Diagnostics.Stopwatch.Frequency / (Warm + Measure)).ToString("F2"))) + $" filled/frame {(double)CoverageBuffer.Filled / (Warm + Measure):F0}");
                    if (w.Factored != null)
                        Log("  solve stages ms/frame: " + string.Join(" ", Array.ConvertAll(w.Factored.StageTicks, t => (t * 1000.0 / System.Diagnostics.Stopwatch.Frequency / (Warm + Measure)).ToString("F2"))) + $" fellback {w.Factored.FellBack} (state {w.Factored.FellBackS}) inversions/agent {(double)w.Factored.LastInversions / n:F2} released {w.HoldsReleased}");
                }
                Log($"{name,-16} {(name == "masslod" ? "caps x" : name == "timeslice" ? "every " : "T* ")}{knob,5:G3} N={n,5}: p95 {P(0.95f),6:F1} ms (mean {mean,5:F1}, step {stepMs / Measure,5:F1}, alloc {allocMs / Measure,5:F2})" +
                    $"  quality {qSum / Math.Max(qN, 1):F3} perceived {qPer / Math.Max(qN, 1):F3} screen {qScr / Math.Max(aScr, 1e-9):F3} image {img:F3} pop {pop:F4} state {qSt / Math.Max(qN, 1):F3}{(options != null && w.Options != null ? " options " + mix : "")}  worst/cap {worst / w.Cap,5:F2}  surrogates {sur}  thermo {thermo:F1}");
                var hv = new int[2, 3, 4];
                for (int i = 0; i < n; i++)
                {
                    int v = w.InView[i] ? 0 : 1;
                    hv[v, 0, w.Beh[i]]++; hv[v, 1, NavRun(w, i)]++; hv[v, 2, w.AnimV[0][i]]++;
                }
                string H(int v, int ax) => $"{hv[v, ax, 0]}/{hv[v, ax, 1]}/{hv[v, ax, 2]}/{hv[v, ax, 3]}";
                Log($"  tiers in view: beh {H(0, 0)} nav {H(0, 1)} anim {H(0, 2)} | off view: beh {H(1, 0)} nav {H(1, 1)} anim {H(1, 2)}");
                w.Dispose();
            }
        }

        UnityEngine.Object.DestroyImmediate(probe);
        UnityEngine.Object.DestroyImmediate(shot);
        cam.targetTexture = null;
        UnityEngine.Object.DestroyImmediate(rt);
        if (full != null) UnityEngine.Object.DestroyImmediate(full);
        UnityEngine.Object.DestroyImmediate(cam.gameObject);
        UnityEngine.Object.DestroyImmediate(set);
    }

    static void Orbit(Camera c, SceneSpec s, float t, out Vector2 xz, out float yaw)
    {
        float r = 0.45f * Mathf.Min(s.Size.x, s.Size.y);
        float ang = t * Mathf.PI * 2f;
        xz = new Vector2(s.Focus.x + r * Mathf.Cos(ang), s.Focus.y + r * Mathf.Sin(ang));
        var look = new Vector3(s.Focus.x, 1.2f, s.Focus.y);
        c.transform.position = new Vector3(xz.x, s.CamHeight, xz.y);
        c.transform.LookAt(look);
        c.fieldOfView = 55f;
        yaw = Mathf.Atan2(look.z - xz.y, look.x - xz.x);
    }

    static string F(float v) => v.ToString("G6", CultureInfo.InvariantCulture);
    // -- the image judge ---------------------------------------------------------------
    // Every policy's frames are judged by the same pixel judge that prices geometry quality: a
    // sampled frame is re-rendered afterwards, out of the timed loop, as the policy drew it and
    // at full fidelity (every agent at geometry and animation tier 0), and scored as the share of
    // the crowd's image it kept, 1 - |policy - reference| / |empty - reference|. This sees what
    // the tier labels cannot: occlusion, screen size, and the pose itself.
    const int ImageEvery = 15;

    sealed class Snap
    {
        public int F, Option, PrevOption; public Vector2[] Pos; public float[] Heading, Phase, Joint, Walk; public sbyte[] Geo, Anim, PrevGeo, PrevAnim;
        public static Snap Of(CrowdWorld w, int f, sbyte[] prevGeo = null, sbyte[] prevAnim = null) => new Snap
        {
            F = f, PrevGeo = prevGeo, PrevAnim = prevAnim, Pos = (Vector2[])w.Pos.Clone(), Heading = (float[])w.Heading.Clone(), Phase = (float[])w.Phase.Clone(),
            Joint = (float[])w.JointBlend.Clone(), Walk = (float[])w.Walk.Clone(),
            Geo = (sbyte[])w.GeoV[0].Clone(), Anim = (sbyte[])w.AnimV[0].Clone(),
        };
        public void Restore(CrowdWorld w)
        {
            Array.Copy(Pos, w.Pos, w.N); Array.Copy(Heading, w.Heading, w.N); Array.Copy(Phase, w.Phase, w.N);
            Array.Copy(Joint, w.JointBlend, w.N); Array.Copy(Walk, w.Walk, w.N);
            Array.Copy(Geo, w.GeoV[0], w.N); Array.Copy(Anim, w.AnimV[0], w.N);
        }
    }

    static double ImageQuality(CrowdWorld w, CrowdRenderer rend, Camera cam, SceneSpec spec, List<Snap> snaps,
                               Texture2D shot, RenderTexture rt, RenderTexture shown, Action<int> apply, out double pop)
    {
        var keepLook = rend.Look;
        rend.Look = Look.Natural;
        var keep = Snap.Of(w, 0);
        var keepD = (float[])w.Dmeas.Clone();
        Array.Clear(w.Dmeas, 0, w.N);          // the divergence glow is a visualisation, not rendering
        Color32[] Grab(bool draw)
        {
            if (draw) rend.Draw(w, cam);
            return GrabFrame(cam, rt, shown, shot);
        }
        // the policy's frame under the setting it chose; the reference and the empty frame always
        // under full settings, so a cheaper global setting is charged for what it loses
        // pop: the same state drawn with this frame's tiers and with the last frame's, both poses
        // decoded afresh from the same phase, so the difference is what the switches alone did
        double sum = 0, popSum = 0; int k = 0, kp = 0;
        foreach (var s in snaps)
        {
            s.Restore(w);
            Orbit(cam, spec, s.F / 2400f, out _, out _);
            apply?.Invoke(s.Option);
            var a = Grab(true);
            double[] popDiff = null;
            if (s.PrevGeo != null)
            {
                w.Pose(s.Anim);
                var now = Grab(true);
                Array.Copy(s.PrevGeo, w.GeoV[0], w.N); Array.Copy(s.PrevAnim, w.AnimV[0], w.N);
                w.Pose(s.PrevAnim);
                apply?.Invoke(s.PrevOption);
                popDiff = new[] { Diff(now, Grab(true)) };
            }
            apply?.Invoke(0);
            var e = Grab(false);
            for (int i = 0; i < w.N; i++) { w.GeoV[0][i] = 0; w.AnimV[0][i] = 0; }
            w.Pose(0);
            var r = Grab(true);
            double absent = Diff(e, r);
            if (absent > 0)
            {
                sum += Math.Max(0.0, 1.0 - Diff(a, r) / absent); k++;
                if (popDiff != null) { popSum += popDiff[0] / absent; kp++; }
            }
        }
        rend.Look = keepLook;
        keep.Restore(w);
        Array.Copy(keepD, w.Dmeas, w.N);
        pop = kp > 0 ? popSum / kp : double.NaN;
        return k > 0 ? sum / k : double.NaN;
    }

    static Color32[] GrabFrame(Camera cam, RenderTexture rt, RenderTexture shown, Texture2D shot)
    {
        cam.Render();
        if (shown != rt) Graphics.Blit(rt, shown);
        var prev = RenderTexture.active;
        RenderTexture.active = shown;
        shot.ReadPixels(new Rect(0, 0, shown.width, shown.height), 0, 0, false);
        shot.Apply(false);
        RenderTexture.active = prev;
        return shot.GetPixels32();
    }

    // -- B: global shadow settings -------------------------------------------------------
    static readonly string[] OptionNames = { "full shadows", "hard 2-cascade 50 m", "no shadows" };

    static void ApplyOption(int o, Light sun, CrowdRenderer rend)
    {
        switch (o)
        {
            case 0:
                QualitySettings.shadows = ShadowQuality.All; QualitySettings.shadowCascades = 4; QualitySettings.shadowDistance = 110f;
                sun.shadows = LightShadows.Soft; rend.Shadows = true; break;
            case 1:
                QualitySettings.shadows = ShadowQuality.HardOnly; QualitySettings.shadowCascades = 2; QualitySettings.shadowDistance = 50f;
                sun.shadows = LightShadows.Hard; rend.Shadows = true; break;
            default:
                QualitySettings.shadows = ShadowQuality.Disable; sun.shadows = LightShadows.None; rend.Shadows = false; break;
        }
    }

    // the empty scene, rendered and waited for: the setting's fixed cost
    static double TimeEmpty(Camera cam, Texture2D probe, RenderTexture rt, int rounds = 40)
    {
        double best = double.MaxValue;
        for (int r = 0; r < rounds; r++)
        {
            float t0 = Time.realtimeSinceStartup;
            cam.Render();
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            probe.ReadPixels(new Rect(0, 0, 1, 1), 0, 0, false);
            RenderTexture.active = prev;
            if (r >= 5) best = Math.Min(best, (Time.realtimeSinceStartup - t0) * 1000.0);
        }
        return best;
    }

    // Option 0 is priced and judged exactly as every other world is (the main calibration and
    // table); each other option carries its measured difference from option 0 on top. Frame cost
    // is measured at the operating crowd size, everyone at the cheapest tier.
    static void AnchorOptions(CrowdWorld.GlobalOption[] options, double[][] raw, double[] gq, SceneSpec spec, ParityTable table,
                              int n, CrowdRenderer rend, Camera cam, Action<int> apply, Texture2D probe, RenderTexture rt)
    {
        var w = new CrowdWorld(Policy.Parity, n, spec, 99u, table);
        for (int i = 0; i < n; i++) { w.Geo[i] = 3; w.GeoV[0][i] = 3; w.AnimV[0][i] = 3; }
        Orbit(cam, spec, 0f, out _, out _);
        // interleaved blocks, fastest frame per option: clocks drift over a calibration, and a
        // sequential pass would hand that drift to whichever option went last
        var frame = new double[options.Length];
        for (int o = 0; o < options.Length; o++) frame[o] = double.MaxValue;
        for (int blk = 0; blk < 6; blk++)
            for (int o = 0; o < options.Length; o++)
            {
                apply(o);
                for (int r = 0; r < 6; r++)
                {
                    float t0 = Time.realtimeSinceStartup;
                    rend.Draw(w, cam);
                    cam.Render();
                    var prev = RenderTexture.active;
                    RenderTexture.active = rt;
                    probe.ReadPixels(new Rect(0, 0, 1, 1), 0, 0, false);
                    RenderTexture.active = prev;
                    if (r >= 2) frame[o] = Math.Min(frame[o], (Time.realtimeSinceStartup - t0) * 1000.0);
                }
            }
        apply(0);
        w.Dispose();
        for (int o = 0; o < options.Length; o++)
        {
            var op = options[o];
            op.FixedMs = (float)(frame[o] - frame[0]);
            op.DrawnMs = 0f;
            for (int t = 0; t < 3; t++)
                op.GeoTheta[t] = Math.Max(CrowdWorld.CalibratedTheta[3, t] + raw[o][t] - raw[0][t], 0.0);
            if (o == 0) op.GeoQuality = (double[])gq.Clone();
        }
    }

    static CrowdWorld.GlobalOption[] CalibrateOptions(SceneSpec spec, ParityTable table, CrowdRenderer rend, Camera cam,
                                                      Action<int> apply, Texture2D probe, Texture2D shot,
                                                      RenderTexture rt, RenderTexture shown)
    {
        var opts = new CrowdWorld.GlobalOption[OptionNames.Length];
        double empty0 = 0, imp0 = 0, abs0 = 0;
        Color32[] ref0 = null;
        var scene0 = new Color32[4][];
        for (int o = 0; o < opts.Length; o++)
        {
            apply(o);
            Orbit(cam, spec, 0f, out _, out _);
            var cw = new CrowdWorld(Policy.Parity, 400, spec, 99u, table);
            var geo = rend.MeasureGeometry(cw, cam);
            int drawn = Math.Max(rend.Drawn, 1);
            cw.Dispose();
            double empty = TimeEmpty(cam, probe, rt);
            double imp = geo[3] - empty / drawn;
            var jw = new CrowdWorld(Policy.Parity, 12, spec, 5u, table);
            var gq = Director.MeasureQuality(rend, jw, cam, spec, ref0, abs0, out var ownRef, out double ownAbs);
            int judged = jw.N;
            jw.Dispose();
            if (o == 0) { ref0 = ownRef; abs0 = ownAbs; empty0 = empty; imp0 = imp; }
            // the scene alone at the bench camera, against full settings, in objective units:
            // one agent at d0 carries abs0 / judged of pixel mass, and a unit of its geometry
            // quality is worth 1 / NAxes of objective
            double sceneDiff = 0;
            for (int p = 0; p < scene0.Length; p++)
            {
                Orbit(cam, spec, p / (float)scene0.Length, out _, out _);
                var img = GrabFrame(cam, rt, shown, shot);
                if (o == 0) scene0[p] = img; else sceneDiff += Diff(img, scene0[p]) / scene0.Length;
            }
            opts[o] = new CrowdWorld.GlobalOption
            {
                Name = OptionNames[o],
                FixedMs = (float)(empty - empty0),
                DrawnMs = (float)(imp - imp0),
                GeoTheta = new[] { Math.Max(geo[0] - geo[3], 0.0), Math.Max(geo[1] - geo[3], 0.0), Math.Max(geo[2] - geo[3], 0.0), 0.0 },
                GeoQuality = gq,
                SceneLoss = abs0 > 0 ? (float)(sceneDiff / (ParityTable.NAxes * abs0 / judged)) : 0f,
            };
        }
        apply(0);
        return opts;
    }

    static double Diff(Color32[] a, Color32[] b)
    {
        long s = 0;
        for (int i = 0; i < a.Length; i++)
            s += Math.Abs(a[i].r - b[i].r) + Math.Abs(a[i].g - b[i].g) + Math.Abs(a[i].b - b[i].b);
        return s;
    }

    static List<(string name, Policy pol, float knob)> Interleave(List<(string name, Policy pol, float knob)> cells, int seed)
    {
        var groups = new List<List<(string name, Policy pol, float knob)>>();
        var at = new Dictionary<string, int>();
        foreach (var c in cells)
        {
            if (!at.TryGetValue(c.name, out int k)) { k = groups.Count; at[c.name] = k; groups.Add(new List<(string, Policy, float)>()); }
            groups[k].Add(c);
        }
        if (seed % 2 == 1) foreach (var g in groups) g.Reverse();
        var rng = new System.Random(seed);
        var order = new List<(string name, Policy pol, float knob)>();
        for (int round = 0; ; round++)
        {
            var pick = new List<int>();
            for (int k = 0; k < groups.Count; k++) if (round < groups[k].Count) pick.Add(k);
            if (pick.Count == 0) break;
            for (int i = pick.Count - 1; i > 0; i--) { int j = rng.Next(i + 1); (pick[i], pick[j]) = (pick[j], pick[i]); }
            foreach (int k in pick) order.Add(groups[k][round]);
        }
        return order;
    }

    // a fixed parallel workload, best of three, in ms: how fast the machine is running right now
    static readonly double[] thermoSink = new double[64];
    static double Thermometer()
    {
        double best = double.MaxValue;
        for (int r = 0; r < 3; r++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            System.Threading.Tasks.Parallel.For(0, 64, c =>
            {
                double a = 0;
                for (int i = 0; i < 100000; i++) a += Math.Sin(i * 1e-3 + c);
                thermoSink[c] = a;
            });
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    // the number after a name token, up to the next '_': "parity_sw0.01_gain0.02" -> 0.01, 0.02
    static float? Token(string name, string key)
    {
        int at = name.IndexOf(key, StringComparison.Ordinal);
        if (at < 0) return null;
        int from = at + key.Length, to = name.IndexOf('_', from);
        return float.Parse(to < 0 ? name.Substring(from) : name.Substring(from, to - from), CultureInfo.InvariantCulture);
    }

    // the navigation that actually ran: none for a surrogate, whatever tier it is labelled with
    static int NavRun(CrowdWorld w, int i) => w.Beh[i] == 3 ? 3 : w.Nav[i];
    // and the animation that was shown: none on an impostor
    static int AnimRun(CrowdWorld w, int i) => w.GeoV[0][i] == 3 ? 3 : w.AnimV[0][i];

    static void Log(string s) { Debug.Log("[FrameBench] " + s); }
}
