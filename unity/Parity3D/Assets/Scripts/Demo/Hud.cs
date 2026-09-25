// On-screen instrumentation. The two numbers that matter are the allocator's own cost (it has
// to be small enough to be worth paying) and the worst-agent divergence against the cap --
// the baseline's climbs without bound, PARITY's saws.
using UnityEngine;

namespace Parity
{
    public sealed class Hud : MonoBehaviour
    {
        const int TraceW = 300, TraceH = 74;
        Texture2D trace, px;
        Color32[] buf;
        GUIStyle head, small, mono;
        bool styled;

        static readonly Color ColBase = new Color(0.84f, 0.35f, 0.10f);
        static readonly Color ColPar = new Color(0.00f, 0.45f, 0.70f);
        static readonly Color ColCap = new Color(0.10f, 0.78f, 0.45f);

        void Awake()
        {
            trace = new Texture2D(TraceW, TraceH, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point };
            buf = new Color32[TraceW * TraceH];
            px = new Texture2D(1, 1); px.SetPixel(0, 0, Color.white); px.Apply();
        }

        void Style()
        {
            if (styled) return;
            styled = true;
            head = new GUIStyle(GUI.skin.label) { fontSize = 17, fontStyle = FontStyle.Bold };
            small = new GUIStyle(GUI.skin.label) { fontSize = 11 };
            mono = new GUIStyle(GUI.skin.label) { fontSize = 12, font = Font.CreateDynamicFontFromOSFont("Menlo", 12) };
            if (mono.font == null) mono = new GUIStyle(GUI.skin.label) { fontSize = 12 };
        }

        void OnGUI()
        {
            var d = Director.Instance;
            if (d == null || d.Par == null) return;
            Style();
            float w = Screen.width, h = Screen.height;

            Box(new Rect(w * 0.5f - 1, 0, 2, h), new Color(0, 0, 0, 0.6f));
            Panel(new Rect(12, 10, w * 0.5f - 24, 132), "MassLOD baseline", ColBase, d.Base, d, false);
            Panel(new Rect(w * 0.5f + 12, 10, w * 0.5f - 24, 132), "PARITY", ColPar, d.Par, d, true);

            Controls(new Rect(12, h - 168, 330, 156), d);
            TracePanel(new Rect(w - 330, h - 168, 318, 156), d);
        }

        void Panel(Rect r, string title, Color c, CrowdWorld world, Director d, bool parity)
        {
            Box(r, new Color(0.05f, 0.06f, 0.08f, 0.82f));
            GUILayout.BeginArea(new Rect(r.x + 10, r.y + 6, r.width - 20, r.height - 10));
            var prev = GUI.color; GUI.color = c;
            GUILayout.Label(title, head);
            GUI.color = prev;

            if (parity)
            {
                var a = world.Last;
                GUILayout.Label($"crowd step {world.StepMs,6:F2} ms     allocator {world.AllocMs,5:F2} ms" +
                                $"     budget {world.BudgetMs,5:F1} ms", mono);
                GUILayout.Label($"lambda {a.Lambda,9:G4}   evals {a.Evals,2}   fill {a.FillSteps,2}" +
                                $"   slack {a.Slack,6:F2} ms{(a.Infeasible ? "   INFEASIBLE" : "")}" +
                                $"{(a.Forced ? "   forced" : "")}", mono);
                GUILayout.Label($"restorations {world.Restorations,6}   this frame  +{world.Promotes,3} promoted" +
                                $"  -{world.Demotes,3} demoted", mono);
                var prevC = GUI.color;
                GUI.color = world.CapBreaches == 0 ? new Color(0.2f, 0.85f, 0.5f) : Color.red;
                GUILayout.Label($"invariant 3: agents over the cap  {world.CapBreaches}" +
                                "   (the mask makes this impossible; nonzero = bug)", mono);
                GUI.color = prevC;
            }
            else
            {
                GUILayout.Label($"crowd step {world.StepMs,6:F2} ms     (no allocator: distance bands," +
                                " frustum, per-level caps)", mono);
                GUILayout.Label("one LOD level per agent, applied to every axis at once", small);
                GUILayout.Label("no error ledger: nothing ever forces a restoration", small);
            }
            Histogram(world, GUILayoutUtility.GetRect(r.width - 24, 40));
            GUILayout.EndArea();
        }

        static readonly Color[] TierCol =
        {
            new Color(0.42f, 0.74f, 1.00f), new Color(0.38f, 0.85f, 0.62f),
            new Color(0.95f, 0.82f, 0.35f), new Color(0.70f, 0.70f, 0.78f),
        };

        void Histogram(CrowdWorld world, Rect r)
        {
            var counts = world.Counts;
            float rowH = r.height / ParityTable.NAxes;
            for (int ax = 0; ax < ParityTable.NAxes; ax++)
            {
                float x = r.x + 74, y = r.y + ax * rowH;
                GUI.Label(new Rect(r.x, y - 2, 72, rowH), ParityTable.AxisNames[ax], small);
                float avail = r.width - 80;
                for (int t = 0; t < ParityTable.NTiers; t++)
                {
                    float frac = counts[ax, t] / (float)Mathf.Max(world.N, 1);
                    float bw = frac * avail;
                    if (bw > 0.5f) Box(new Rect(x, y + 1, bw, rowH - 3), TierCol[t]);
                    x += bw;
                }
            }
        }

        void Controls(Rect r, Director d)
        {
            Box(r, new Color(0.05f, 0.06f, 0.08f, 0.82f));
            GUILayout.BeginArea(new Rect(r.x + 10, r.y + 6, r.width - 20, r.height - 10));
            GUILayout.Label($"agents  {d.Agents}", mono);
            int n = Mathf.RoundToInt(GUILayout.HorizontalSlider(d.Agents, 100, 12000));
            n = (n / 100) * 100;
            if (n != d.Agents) d.PendingAgents = n;

            if (d.AbsoluteBudget)
            {
                GUILayout.Label($"frame budget  {d.TargetMs:F1} ms absolute  (spent {d.Par.Last.Cost:F1})", mono);
                d.TargetMs = GUILayout.HorizontalSlider(d.TargetMs, 0.5f, 40f);
            }
            else
            {
                GUILayout.Label($"frame budget  {100 * d.BudgetFrac:F0}% of the tier span" +
                                $"  = {d.Par.BudgetMs:F1} ms", mono);
                d.BudgetFrac = GUILayout.HorizontalSlider(d.BudgetFrac, 0.02f, 1f);
            }

            GUILayout.Label($"divergence cap  {d.Cap:F2} nats", mono);
            d.Cap = GUILayout.HorizontalSlider(d.Cap, 0.5f, 20f);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button(d.Paused ? "resume" : "pause")) d.Paused = !d.Paused;
            if (GUILayout.Button(d.Orbit ? "stop cam" : "orbit")) d.Orbit = !d.Orbit;
            if (GUILayout.Button(d.ColourByDivergence ? "colour: error" : "colour: tier"))
                d.ColourByDivergence = !d.ColourByDivergence;
            if (GUILayout.Button(d.AbsoluteBudget ? "budget: ms" : "budget: %")) d.AbsoluteBudget = !d.AbsoluteBudget;
            GUILayout.EndHorizontal();
            var b = BenchmarkRunner.Instance;
            if (b != null && GUILayout.Button(b.Running ? $"benchmarking  {b.Progress}" : "run benchmark sweep -> CSV"))
                b.Begin();
            GUILayout.EndArea();
        }

        void TracePanel(Rect r, Director d)
        {
            Box(r, new Color(0.05f, 0.06f, 0.08f, 0.82f));
            GUI.Label(new Rect(r.x + 10, r.y + 4, r.width, 18), "worst-agent accumulated divergence", small);

            float hi = 1e-3f;
            for (int i = 0; i < d.TraceBase.Length; i++)
            {
                if (d.TraceBase[i] > hi) hi = d.TraceBase[i];
                if (d.TracePar[i] > hi) hi = d.TracePar[i];
            }
            hi = Mathf.Max(hi, d.Cap * 1.4f);

            for (int i = 0; i < buf.Length; i++) buf[i] = new Color32(14, 16, 20, 255);
            PlotLine(d.TraceCap(), hi, ColCap);
            Plot(d.TraceBase, d.TraceHead, hi, ColBase);
            Plot(d.TracePar, d.TraceHead, hi, ColPar);
            if (Event.current.type == EventType.Repaint) { trace.SetPixels32(buf); trace.Apply(false); }
            GUI.DrawTexture(new Rect(r.x + 10, r.y + 22, TraceW, TraceH), trace);

            GUI.Label(new Rect(r.x + 10, r.y + 100, r.width, 18),
                      $"baseline {d.Base.MaxDivergence(),8:F1}      PARITY {d.Par.MaxDivergence(),6:F2}" +
                      $"   cap {d.Cap:F2}", mono);
            GUI.Label(new Rect(r.x + 10, r.y + 120, r.width, 32),
                      "log scale. the baseline climbs and never returns; PARITY saws because the\n" +
                      "ledger forces a restoration before any agent reaches the cap.", small);
        }

        void Plot(float[] series, int head, float hi, Color c)
        {
            int prevY = -1;
            for (int x = 0; x < TraceW; x++)
            {
                int k = (head + 1 + Mathf.FloorToInt(x * (series.Length - 1f) / (TraceW - 1f))) % series.Length;
                float v = Mathf.Max(series[k], 1e-3f);
                int y = Mathf.Clamp(Mathf.RoundToInt(Mathf.Log10(v / 1e-3f) / Mathf.Log10(hi / 1e-3f) * (TraceH - 1)), 0, TraceH - 1);
                if (prevY >= 0)
                    for (int yy = Mathf.Min(prevY, y); yy <= Mathf.Max(prevY, y); yy++) Set(x, yy, c);
                else Set(x, y, c);
                prevY = y;
            }
        }

        void PlotLine(float value, float hi, Color c)
        {
            int y = Mathf.Clamp(Mathf.RoundToInt(Mathf.Log10(Mathf.Max(value, 1e-3f) / 1e-3f) / Mathf.Log10(hi / 1e-3f) * (TraceH - 1)), 0, TraceH - 1);
            for (int x = 0; x < TraceW; x += 3) { Set(x, y, c); Set(x + 1, y, c); }
        }

        void Set(int x, int y, Color c)
        {
            if (x < 0 || x >= TraceW || y < 0 || y >= TraceH) return;
            buf[y * TraceW + x] = c;
        }

        void Box(Rect r, Color c)
        {
            var prev = GUI.color; GUI.color = c;
            GUI.DrawTexture(r, px); GUI.color = prev;
        }
    }
}
