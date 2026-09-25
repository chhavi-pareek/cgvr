// Renders every set headlessly and writes PNGs, so the look can be checked without opening
// the editor. Run with graphics (no -nographics):
//
//   Unity -batchmode -quit -projectPath <p> -executeMethod ParityShots.Run
//
// Writes Logs/shots/<scene>_split.png (baseline | PARITY, as the demo shows them),
// <scene>_err.png (divergence colouring) and <scene>_near_<policy>.png (the mesh tiers up close),
// and logs the measured per-tier geometry cost.
using System;
using System.IO;
using UnityEngine;
using Parity;

public static class ParityShots
{
    const int PanelW = 960, PanelH = 1080, Frames = 900;

    public static void Run()
    {
        int failures = 0;
        string dir = Path.GetFullPath(Path.Combine(Application.dataPath, "../Logs/shots"));
        Directory.CreateDirectory(dir);
        try
        {
            foreach (var k in new[] { SceneKind.Plaza, SceneKind.Hub, SceneKind.Corridor }) Shoot(k, dir);
        }
        catch (Exception ex)
        {
            Debug.LogError("[ParityShots] threw: " + ex);
            failures++;
        }
        Log(failures == 0 ? "SHOTS DONE " + dir : "SHOTS FAILED");
        if (Application.isBatchMode) UnityEditor.EditorApplication.Exit(failures == 0 ? 0 : 1);
    }

    static void Shoot(SceneKind kind, string dir)
    {
        var spec = SceneSpec.Get(kind);
        Director.ApplyQuality(true);
        var set = Stage.Build(spec, out _);
        var rend = new CrowdRenderer(Shader.Find("Parity/CrowdInstanced"), Shader.Find("Parity/CrowdImpostor"),
                                     Shader.Find("Parity/CrowdDecal"));
        var cam = Director.MakeCamera("Shot", new Rect(0f, 0f, 1f, 1f));
        var fx = cam.GetComponent<PostFx>();
        fx.enabled = false;
        cam.enabled = false;
        var hdr = new RenderTexture(PanelW, PanelH, 24, RenderTextureFormat.DefaultHDR) { antiAliasing = 4 };
        var ldr = new RenderTexture(PanelW, PanelH, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        cam.targetTexture = hdr;

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
        Log($"{spec.Name}: geometry us/agent  {geo[0] * 1e3:F2}  {geo[1] * 1e3:F2}  {geo[2] * 1e3:F2}  {geo[3] * 1e3:F2}");
        var jw = new CrowdWorld(Policy.Parity, 12, spec, 5u, table);
        var gq = Director.MeasureQuality(rend, jw, cam, spec);
        jw.Dispose();
        Log($"{spec.Name}: geometry quality (pixel judge)  {gq[0]:F3}  {gq[1]:F3}  {gq[2]:F3}  {gq[3]:F3}");
        table = ParityTable.Build(spec.EMax, gq);
        int kept;
        CrowdWorld.PricedAndPruned(table, floor, CrowdWorld.CalibratedTheta, out kept);
        var alloc = table;   // full table: the demo's default two-salience allocator

        int n = spec.DefaultAgents;
        var bas = new CrowdWorld(Policy.Baseline, n, spec, 1u, alloc);
        var par = new CrowdWorld(Policy.Parity, n, spec, 1u, alloc);
        bas.ApplyCalibration(baseFloor, baseTheta);
        par.ApplyCalibration(floor, CrowdWorld.CalibratedTheta);
        float tOrbit = 0.12f;
        for (int f = 0; f < Frames; f++)
        {
            Orbit(cam, spec, tOrbit * f / Frames, out var xz, out float yaw);
            bas.Step(xz, yaw, bas.StepMs);
            par.MatchBudgetMs = bas.PredictedSpendMs();
            par.Step(xz, yaw, par.StepMs);
        }
        var c = par.Counts;
        Log($"{spec.Name}: N {n}  matched budget {par.BudgetMs:F2} ms  PARITY spent {par.Last.Cost:F2}" +
            $"   behaviour {c[0, 0]}/{c[0, 1]}/{c[0, 2]}/{c[0, 3]}   geometry {c[3, 0]}/{c[3, 1]}/{c[3, 2]}/{c[3, 3]}");
        var cb = bas.Counts;
        Log($"{spec.Name}: baseline behaviour {cb[0, 0]}/{cb[0, 1]}/{cb[0, 2]}/{cb[0, 3]}" +
            $"   geometry {cb[3, 0]}/{cb[3, 1]}/{cb[3, 2]}/{cb[3, 3]}   rows kept {kept}" +
            $"   worst divergence base {bas.MaxDivergence():F1} / parity {par.MaxDivergence():F2} (cap {par.Cap:F2})");

        var tex = new Texture2D(PanelW, PanelH, TextureFormat.RGBA32, false);
        var split = new Texture2D(PanelW * 2, PanelH, TextureFormat.RGBA32, false);
        foreach (var look in new[] { Look.Natural, Look.Divergence })
        {
            rend.Look = look;
            rend.Cap = par.Cap;
            Orbit(cam, spec, tOrbit, out _, out _);
            Grab(cam, fx, rend, bas, hdr, ldr, tex); split.SetPixels(0, 0, PanelW, PanelH, tex.GetPixels());
            Grab(cam, fx, rend, par, hdr, ldr, tex); split.SetPixels(PanelW, 0, PanelW, PanelH, tex.GetPixels());
            split.Apply();
            File.WriteAllBytes(Path.Combine(dir, $"{spec.Name}_{(look == Look.Natural ? "split" : "err")}.png"),
                               split.EncodeToPNG());
        }
        // up close: from the orbit position, looking at the ground 10 m ahead, at a longer lens
        rend.Look = Look.Natural;
        Orbit(cam, spec, tOrbit, out var eye2, out _);
        var fwd = new Vector3(spec.Focus.x - eye2.x, 0f, spec.Focus.y - eye2.y).normalized;
        cam.transform.LookAt(new Vector3(eye2.x, 0.8f, eye2.y) + fwd * 10f);
        cam.fieldOfView = 38f;
        foreach (var w in new[] { bas, par })
        {
            Grab(cam, fx, rend, w, hdr, ldr, tex);
            File.WriteAllBytes(Path.Combine(dir, $"{spec.Name}_near_{(w == bas ? "baseline" : "parity")}.png"),
                               tex.EncodeToPNG());
        }

        bas.Dispose(); par.Dispose();
        cam.targetTexture = null;
        UnityEngine.Object.DestroyImmediate(hdr);
        UnityEngine.Object.DestroyImmediate(ldr);
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

    static void Grab(Camera cam, PostFx fx, CrowdRenderer rend, CrowdWorld w, RenderTexture hdr, RenderTexture ldr,
                     Texture2D tex)
    {
        rend.Draw(w, cam);
        cam.Render();
        fx.Process(hdr, ldr);
        var prev = RenderTexture.active;
        RenderTexture.active = ldr;
        tex.ReadPixels(new Rect(0, 0, PanelW, PanelH), 0, 0, false);
        tex.Apply();
        RenderTexture.active = prev;
    }

    static void Log(string s) { Debug.Log("[ParityShots] " + s); }
}
