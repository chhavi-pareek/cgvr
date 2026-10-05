// Renders the palace demo headlessly to numbered JPEGs plus a caption table; figures/palace_video.py
// stamps the captions and encodes the MP4. Run with graphics (no -nographics):
//
//   Unity -batchmode -quit -projectPath <p> -executeMethod ParityVideo.Run -out <dir>
//         [-n 8000] [-realtime 45] [-ff 12] [-maxmin 15]
//
// Part 1, real time: baseline | PARITY side by side on the demo's orbit, from the seated crowd
// through the release. Part 2, fast-forward x ff: PARITY alone from above the whole compound
// until everyone is out (or -maxmin minutes after release). Both worlds are the demo's: matched
// budget, measured costs, the measured-rate gate plan.
using System;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using Parity;

public static class ParityVideo
{
    const int W = 1920, H = 1080, Fps = 30;

    static string Arg(string key)
    {
        var a = Environment.GetCommandLineArgs();
        int i = Array.IndexOf(a, key);
        return i >= 0 && i + 1 < a.Length ? a[i + 1] : null;
    }

    static float ArgF(string key, float dflt) =>
        Arg(key) is string s ? float.Parse(s, CultureInfo.InvariantCulture) : dflt;

    public static void Run()
    {
        int failures = 0;
        try { Record(); }
        catch (Exception ex) { Debug.LogError("[ParityVideo] threw: " + ex); failures++; }
        if (Application.isBatchMode) UnityEditor.EditorApplication.Exit(failures == 0 ? 0 : 1);
    }

    static void Record()
    {
        string dir = Arg("-out") ?? Path.GetFullPath(Path.Combine(Application.dataPath, "../Logs/video"));
        Directory.CreateDirectory(dir);
        foreach (var f in Directory.GetFiles(dir, "frame_*.jpg")) File.Delete(f);
        int n = (int)ArgF("-n", 8000);
        float realtime = ArgF("-realtime", 45f), ff = ArgF("-ff", 12f), maxMin = ArgF("-maxmin", 15f);

        var spec = SceneSpec.Get(SceneKind.Palace);
        PalaceScene.ReleaseSeconds = 3f;
        PalaceScene.ResetSeconds = 1e9f;   // one cycle
        Director.ApplyQuality(true);
        var set = Stage.Build(spec, out _);
        var rend = new CrowdRenderer(Shader.Find("Parity/CrowdInstanced"), Shader.Find("Parity/CrowdImpostor"),
                                     Shader.Find("Parity/CrowdDecal"));
        var cam = Director.MakeCamera("Video", new Rect(0f, 0f, 1f, 1f));
        var fx = cam.GetComponent<PostFx>();
        cam.enabled = false;
        var hdrHalf = new RenderTexture(W / 2, H, 24, RenderTextureFormat.DefaultHDR) { antiAliasing = 4 };
        var ldrHalf = new RenderTexture(W / 2, H, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        var hdrFull = new RenderTexture(W, H, 24, RenderTextureFormat.DefaultHDR) { antiAliasing = 4 };
        var ldrFull = new RenderTexture(W, H, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);

        // calibration exactly as ParityShots does it: measured costs and the pixel judge
        var table = ParityTable.Build(spec.EMax);
        CrowdWorld.CalibrateAxes(spec, table, 400, out double floor);
        var baseTheta = CrowdWorld.MeasureAxes(spec, table, 400, Policy.Baseline, out double baseFloor);
        cam.targetTexture = hdrHalf;
        Orbit(cam, spec, 0f, out _, out _);
        var cw = new CrowdWorld(Policy.Parity, 400, spec, 99u, table);
        var geo = rend.MeasureGeometry(cw, cam);
        cw.Dispose();
        for (int t = 0; t < 3; t++) CrowdWorld.CalibratedTheta[3, t] = baseTheta[3, t] = Math.Max(geo[t] - geo[3], 0.0);
        var jw = new CrowdWorld(Policy.Parity, 12, spec, 5u, table);
        var gq = Director.MeasureQuality(rend, jw, cam, spec);
        jw.Dispose();
        table = ParityTable.Build(spec.EMax, gq);
        CrowdWorld.PricedAndPruned(table, floor, CrowdWorld.CalibratedTheta, out _);

        var bas = new CrowdWorld(Policy.Baseline, n, spec, 1u, table);
        var par = new CrowdWorld(Policy.Parity, n, spec, 1u, table);
        bas.ApplyCalibration(baseFloor, baseTheta);
        par.ApplyCalibration(floor, CrowdWorld.CalibratedTheta);
        var ps = (PalaceScene)par.Scene;
        var bs = (PalaceScene)bas.Scene;

        var half = new Texture2D(W / 2, H, TextureFormat.RGB24, false);
        var full = new Texture2D(W, H, TextureFormat.RGB24, false);
        var caps = new StringBuilder("frame,part,t_release_s,speed,out_par,out_base,n,base_over_cap,par_over_cap,gates\n");
        int frame = 0, step = 0;
        float releaseAt = PalaceScene.ReleaseSeconds;
        int realSteps = Mathf.RoundToInt(realtime / CrowdWorld.Dt);
        int ffEvery = Mathf.Max(1, Mathf.RoundToInt(ff / (CrowdWorld.Dt * Fps)));
        int maxSteps = Mathf.RoundToInt((releaseAt + maxMin * 60f) / CrowdWorld.Dt);
        var t0 = System.Diagnostics.Stopwatch.StartNew();

        // part 1: both worlds, split screen, real time (one frame every 2 steps at 60 Hz)
        for (; step < realSteps; step++)
        {
            float vt = frame / (float)Fps;
            Orbit(cam, spec, vt / 120f, out var xz, out float yaw);
            var pm = cam.projectionMatrix;
            par.SetView(0, pm * cam.worldToCameraMatrix, pm.m00, pm.m11);
            bas.Step(xz, yaw, bas.StepMs);
            par.MatchBudgetMs = bas.PredictedSpendMs();
            par.Step(xz, yaw, par.StepMs);
            if (step % 2 != 0) continue;
            rend.Look = Look.Natural;
            rend.Cap = par.Cap;
            cam.targetTexture = hdrHalf;
            Grab(cam, fx, rend, bas, hdrHalf, ldrHalf, half); full.SetPixels(0, 0, W / 2, H, half.GetPixels());
            Grab(cam, fx, rend, par, hdrHalf, ldrHalf, half); full.SetPixels(W / 2, 0, W / 2, H, half.GetPixels());
            full.Apply();
            Write(dir, frame, full);
            Caption(caps, frame++, 1, step, releaseAt, 1f, ps, bs, n, bas, par);
        }
        Debug.Log($"[ParityVideo] part 1: {frame} frames in {t0.Elapsed.TotalSeconds:F0} s");

        // part 2: PARITY alone, from above the compound, fast-forward until everyone is out
        bas.Dispose();
        cam.targetTexture = hdrFull;
        cam.farClipPlane = 6000f;   // the whole compound and the ground beyond it, not the sky's lower half
        var centre = new Vector3(spec.Size.x * 0.5f, 0f, spec.Size.y * 0.5f);
        for (; step < maxSteps && ps.Out < n; step++)
        {
            float vt = frame / (float)Fps;
            float ang = -0.6f + 0.25f * Mathf.Sin(vt / 40f);   // a slow drift, so the set reads as 3-D
            var eye = centre + new Vector3(Mathf.Sin(ang) * 420f, 360f, -Mathf.Cos(ang) * 420f);
            cam.transform.position = eye;
            cam.transform.LookAt(centre + new Vector3(0f, 0f, 10f));
            cam.fieldOfView = 46f;
            var pm = cam.projectionMatrix;
            par.SetView(0, pm * cam.worldToCameraMatrix, pm.m00, pm.m11);
            par.Step(new Vector2(eye.x, eye.z), Mathf.Atan2(centre.z - eye.z, centre.x - eye.x), par.StepMs);
            if (step % ffEvery != 0) continue;
            rend.Look = Look.Natural;
            Grab(cam, fx, rend, par, hdrFull, ldrFull, full);
            Write(dir, frame, full);
            Caption(caps, frame++, 2, step, releaseAt, ff, ps, bs, n, null, par);
        }
        File.WriteAllText(Path.Combine(dir, "captions.csv"), caps.ToString());
        Debug.Log($"[ParityVideo] done: {frame} frames, PARITY out {ps.Out}/{n}, " +
                  $"{(step * CrowdWorld.Dt - releaseAt) / 60f:F1} min after release, {t0.Elapsed.TotalSeconds:F0} s wall -> {dir}");
        par.Dispose();
        cam.targetTexture = null;
        UnityEngine.Object.DestroyImmediate(cam.gameObject);
        UnityEngine.Object.DestroyImmediate(set);
    }

    static void Caption(StringBuilder sb, int frame, int part, int step, float releaseAt, float speed,
                        PalaceScene ps, PalaceScene bs, int n, CrowdWorld bas, CrowdWorld par)
    {
        var gates = new StringBuilder();
        foreach (int g in ps.Site.Open)
            gates.Append(ps.Site.Gates[g].Key).Append(':').Append(ps.ExitsByGate[g]).Append(' ');
        sb.Append(frame).Append(',').Append(part).Append(',')
          .Append((step * CrowdWorld.Dt - releaseAt).ToString("F2", CultureInfo.InvariantCulture)).Append(',')
          .Append(speed.ToString("F0", CultureInfo.InvariantCulture)).Append(',')
          .Append(ps.Out).Append(',').Append(bas != null ? bs.Out : -1).Append(',').Append(n).Append(',')
          .Append(bas != null ? (bas.MaxDivergence() / par.Cap).ToString("F2", CultureInfo.InvariantCulture) : "").Append(',')
          .Append((par.MaxDivergence() / par.Cap).ToString("F2", CultureInfo.InvariantCulture)).Append(',')
          .Append(gates.ToString().Trim()).Append('\n');
    }

    static void Write(string dir, int frame, Texture2D tex) =>
        File.WriteAllBytes(Path.Combine(dir, $"frame_{frame:D5}.jpg"), tex.EncodeToJPG(90));

    static void Orbit(Camera c, SceneSpec s, float t, out Vector2 xz, out float yaw)
    {
        float r = s.OrbitRadius;
        float ang = t * Mathf.PI * 2f;
        xz = new Vector2(s.Focus.x + r * Mathf.Cos(ang), s.Focus.y + r * Mathf.Sin(ang));
        var look = new Vector3(s.Focus.x, 1.2f, s.Focus.y);
        c.transform.position = new Vector3(xz.x, s.CamHeight, xz.y);
        c.transform.LookAt(look);
        c.fieldOfView = 55f;
        yaw = Mathf.Atan2(look.z - xz.y, look.x - xz.x);
    }

    static void Grab(Camera cam, PostFx fx, CrowdRenderer rend, CrowdWorld w, RenderTexture hdr, RenderTexture ldr,
                     Texture2D tex)
    {
        cam.targetTexture = hdr;
        rend.Draw(w, cam);
        cam.Render();
        fx.Process(hdr, ldr);
        var prev = RenderTexture.active;
        RenderTexture.active = ldr;
        tex.ReadPixels(new Rect(0, 0, ldr.width, ldr.height), 0, 0, false);
        tex.Apply();
        RenderTexture.active = prev;
    }
}
