// On-screen instrumentation. Laid out with explicit rects rather than GUILayout: inside a
// narrow area GUILayout word-wraps every label, which silently ate the invariant-3 counter and
// both tier histograms the first time this ran.
//
// The numbers that matter: the allocator's own cost (it has to be worth paying), the
// invariant-3 counter (which must read 0), and the followed agent's divergence, which saws.
using UnityEngine;

namespace Parity
{
    public sealed class Hud : MonoBehaviour
    {
        const int TraceW = 340, TraceH = 84;
        Texture2D trace, px;
        Color32[] buf;
        GUIStyle head, small, mono;
        bool styled;
        int line = 15, pad = 9;

        static readonly Color ColBase = new Color(0.84f, 0.35f, 0.10f);
        static readonly Color ColPar = new Color(0.20f, 0.60f, 0.95f);
        static readonly Color ColOne = new Color(0.55f, 0.85f, 1.00f);
        static readonly Color ColCap = new Color(0.10f, 0.82f, 0.48f);
        static readonly Color Panel = new Color(0.04f, 0.05f, 0.07f, 0.88f);

        static readonly Color[] TierCol =
        {
            new Color(0.42f, 0.74f, 1.00f), new Color(0.38f, 0.85f, 0.62f),
            new Color(0.95f, 0.82f, 0.35f), new Color(0.70f, 0.70f, 0.78f),
        };

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
            int fs = Mathf.Clamp(Screen.height / 78, 10, 15);
            line = fs + 4;
            var mf = Font.CreateDynamicFontFromOSFont("Menlo", fs);
            head = new GUIStyle { fontSize = fs + 5, fontStyle = FontStyle.Bold, wordWrap = false,
                                  normal = { textColor = Color.white } };
            small = new GUIStyle { fontSize = fs - 1, wordWrap = false,
                                   normal = { textColor = new Color(0.72f, 0.74f, 0.80f) } };
            mono = new GUIStyle { fontSize = fs, wordWrap = false, font = mf,
                                  normal = { textColor = new Color(0.90f, 0.92f, 0.96f) } };
        }

        int PanelH => pad + line + 5 * line + 4 * (line - 3) + pad;

        void OnGUI()
        {
            var d = Director.Instance;
            if (d == null || d.Par == null || d.Base == null) return;   // mid domain reload
            if (trace == null) Awake();                                 // textures die with the reload
            Style();
            float w = Screen.width, h = Screen.height, half = w * 0.5f;

            Box(new Rect(half - 1, 0, 2, h), new Color(0, 0, 0, 0.55f));
            Panels(d, half);
            Controls(new Rect(12, h - 196, Mathf.Min(360, half - 24), 186), d);
            Trace(new Rect(w - TraceW - 34, h - 196, TraceW + 22, 186), d);
        }

        void Panels(Director d, float half)
        {
            float pw = half - 24;
            var p = d.Par; var b = d.Base;

            var r = new Rect(12, 10, pw, PanelH);
            Box(r, Panel);
            Title(r, "MassLOD baseline", ColBase);
            int i = 0;
            Row(r, i++, $"crowd step {b.StepMs,7:F2} ms      no allocator");
            Row(r, i++, "distance bands, frustum, per-level caps");
            Row(r, i++, "one LOD level per agent, every axis at once");
            Row(r, i++, "ticks at 1 / 3 / 10 frames: skipped frames ARE the divergence");
            RowCol(r, i++, "no error ledger: nothing ever forces a restoration", ColBase);
            Histogram(r, i, b);

            r = new Rect(half + 12, 10, pw, PanelH);
            Box(r, Panel);
            Title(r, "PARITY", ColPar);
            i = 0;
            Row(r, i++, $"crowd step {p.StepMs,7:F2} ms      allocator {p.AllocMs,6:F2} ms");
            Row(r, i++, $"tier work  {p.AllocatableMs,7:F2} ms  of budget {p.BudgetMs,6:F2} ms" +
                        $"   slack {p.Last.Slack,5:F2}");
            Row(r, i++, $"lambda {p.Last.Lambda,9:G4}   evals {p.Last.Evals,2}   fill {p.Last.FillSteps,2}" +
                        (p.Last.Infeasible ? "   INFEASIBLE" : "") + (p.Last.Forced ? "   forced" : ""));
            Row(r, i++, $"restorations {p.Ledger.Restorations,7}   this frame +{p.Promotes,3} / -{p.Demotes,3}");
            RowCol(r, i++, $"invariant 3: agents over the cap  {p.CapBreaches}" +
                           (p.Last.Starved > 0 ? $"   STARVED {p.Last.Starved}" : ""),
                   p.CapBreaches == 0 && p.Last.Starved == 0 ? ColCap : Color.red);
            Histogram(r, i, p);
        }

        void Title(Rect r, string s, Color c)
        {
            head.normal.textColor = c;
            GUI.Label(new Rect(r.x + pad, r.y + pad - 3, r.width - 2 * pad, line + 6), s, head);
            head.normal.textColor = Color.white;
        }

        void Row(Rect r, int i, string s)
        {
            GUI.Label(new Rect(r.x + pad, r.y + pad + line + i * line, r.width - 2 * pad, line), s, mono);
        }

        void RowCol(Rect r, int i, string s, Color c)
        {
            var prev = mono.normal.textColor;
            mono.normal.textColor = c;
            Row(r, i, s);
            mono.normal.textColor = prev;
        }

        void Histogram(Rect r, int startRow, CrowdWorld world)
        {
            var counts = world.Counts;
            int bh = line - 3;
            float x0 = r.x + pad + 76, avail = r.width - 2 * pad - 80;
            for (int ax = 0; ax < ParityTable.NAxes; ax++)
            {
                float y = r.y + pad + line + (startRow + ax) * line - (line - bh) * 0.5f;
                GUI.Label(new Rect(r.x + pad, y - 1, 74, bh + 2), ParityTable.AxisNames[ax], small);
                float x = x0;
                for (int t = 0; t < ParityTable.NTiers; t++)
                {
                    float bw = counts[ax, t] / (float)Mathf.Max(world.N, 1) * avail;
                    if (bw > 0.4f) Box(new Rect(x, y, bw, bh - 2), TierCol[t]);
                    x += bw;
                }
            }
        }

        void Controls(Rect r, Director d)
        {
            Box(r, Panel);
            float x = r.x + pad, w = r.width - 2 * pad, y = r.y + pad;
            GUI.Label(new Rect(x, y, w, line), $"agents  {d.Agents}", mono); y += line;
            int n = Mathf.RoundToInt(GUI.HorizontalSlider(new Rect(x, y + 3, w, 14), d.Agents, 200, 12000) / 100) * 100;
            if (n != d.Agents) d.PendingAgents = n;
            y += line + 6;

            if (d.AbsoluteBudget)
            {
                GUI.Label(new Rect(x, y, w, line), $"budget  {d.TargetMs:F1} ms absolute", mono); y += line;
                d.TargetMs = GUI.HorizontalSlider(new Rect(x, y + 3, w, 14), d.TargetMs, 0.5f, 40f);
            }
            else
            {
                GUI.Label(new Rect(x, y, w, line), $"budget  {100 * d.BudgetFrac:F0}% of tier span" +
                                                   $"  = {d.Par.BudgetMs:F2} ms", mono); y += line;
                d.BudgetFrac = GUI.HorizontalSlider(new Rect(x, y + 3, w, 14), d.BudgetFrac, 0.02f, 1f);
            }
            y += line + 6;

            GUI.Label(new Rect(x, y, w, line), $"divergence cap  {d.Cap:F2} nats", mono); y += line;
            d.Cap = GUI.HorizontalSlider(new Rect(x, y + 3, w, 14), d.Cap, 0.5f, 20f);
            y += line + 8;

            float bw2 = (w - 18) / 4f;
            if (GUI.Button(new Rect(x, y, bw2, 22), d.Paused ? "resume" : "pause")) d.Paused = !d.Paused;
            if (GUI.Button(new Rect(x + bw2 + 6, y, bw2, 22), d.Orbit ? "hold cam" : "orbit")) d.Orbit = !d.Orbit;
            if (GUI.Button(new Rect(x + 2 * (bw2 + 6), y, bw2, 22), d.ColourByDivergence ? "col: err" : "col: tier"))
                d.ColourByDivergence = !d.ColourByDivergence;
            if (GUI.Button(new Rect(x + 3 * (bw2 + 6), y, bw2, 22), d.AbsoluteBudget ? "bud: ms" : "bud: %"))
                d.AbsoluteBudget = !d.AbsoluteBudget;
            y += 26;
            var bm = BenchmarkRunner.Instance;
            if (bm != null && GUI.Button(new Rect(x, y, w, 22),
                    bm.Running ? $"benchmarking  {bm.Progress}" : "run benchmark sweep -> CSV"))
                bm.Begin();
        }

        void Trace(Rect r, Director d)
        {
            Box(r, Panel);
            float x = r.x + pad, y = r.y + pad;
            GUI.Label(new Rect(x, y, r.width, line), "accumulated divergence, log scale", small);
            y += line;

            float hi = Mathf.Max(d.Cap * 1.6f, 1e-3f);
            for (int i = 0; i < d.TraceBase.Length; i++) if (d.TraceBase[i] > hi) hi = d.TraceBase[i];

            for (int i = 0; i < buf.Length; i++) buf[i] = new Color32(12, 14, 18, 255);
            Dashed(d.Cap, hi, ColCap);
            Plot(d.TraceBase, d.TraceHead, hi, ColBase);
            Plot(d.TracePar, d.TraceHead, hi, ColPar);
            Plot(d.TraceOne, d.TraceHead, hi, ColOne);
            if (Event.current.type == EventType.Repaint) { trace.SetPixels32(buf); trace.Apply(false); }
            GUI.DrawTexture(new Rect(x, y, TraceW, TraceH), trace);
            y += TraceH + 4;

            Key(x, y, ColBase, $"baseline max {d.Base.MaxDivergence(),7:F1}   unbounded"); y += line;
            Key(x, y, ColPar, $"PARITY max   {d.Par.MaxDivergence(),7:F2}   cap {d.Cap:F2}"); y += line;
            Key(x, y, ColOne, $"one agent    {d.Par.TrackedD,7:F2}   restored {d.Par.TrackedRestores}x  <- the sawtooth");
        }

        void Key(float x, float y, Color c, string s)
        {
            Box(new Rect(x, y + 5, 9, 4), c);
            var prev = mono.normal.textColor;
            mono.normal.textColor = c;
            GUI.Label(new Rect(x + 14, y, TraceW, line), s, mono);
            mono.normal.textColor = prev;
        }

        void Plot(float[] series, int headIdx, float hi, Color c)
        {
            int prevY = -1;
            for (int x = 0; x < TraceW; x++)
            {
                int k = (headIdx + 1 + Mathf.FloorToInt(x * (series.Length - 1f) / (TraceW - 1f))) % series.Length;
                int y = Y(series[k], hi);
                if (prevY >= 0) for (int yy = Mathf.Min(prevY, y); yy <= Mathf.Max(prevY, y); yy++) Set(x, yy, c);
                else Set(x, y, c);
                prevY = y;
            }
        }

        static int ClampY(int y) => Mathf.Clamp(y, 0, TraceH - 1);

        int Y(float v, float hi)
        {
            float lo = 1e-2f;
            return ClampY(Mathf.RoundToInt(Mathf.Log10(Mathf.Max(v, lo) / lo) / Mathf.Log10(hi / lo) * (TraceH - 1)));
        }

        void Dashed(float value, float hi, Color c)
        {
            int y = Y(value, hi);
            for (int x = 0; x < TraceW; x += 5) { Set(x, y, c); Set(x + 1, y, c); }
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
