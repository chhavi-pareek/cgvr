// In-engine benchmark: ramps the agent count and measures both policies under the same seed
// and the same camera path, with rendering off so the numbers are the crowd's, not the GPU's.
// This is the "vs existing system" table, generated in the engine rather than in Python.
//
// Columns mirror bench/sweep.py so the two can sit side by side: per-policy step time, the
// allocator's own share, the tier mix, worst-agent divergence, and how often the frame budget
// was actually met.
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

namespace Parity
{
    public sealed class BenchmarkRunner : MonoBehaviour
    {
        public static BenchmarkRunner Instance;

        public int[] Sizes = { 200, 500, 1000, 2000, 4000, 8000 };
        public int WarmupFrames = 60;
        public int MeasureFrames = 240;
        public bool Running { get; private set; }
        public string Progress { get; private set; } = "";

        // static fields are cleared by a domain reload, so re-register on enable too
        void Awake() { Instance = this; }

        void OnEnable() { Instance = this; }

        public void Begin()
        {
            if (Running) return;
            StartCoroutine(Run());
        }

        sealed class Acc
        {
            public readonly List<float> Step = new List<float>(), Alloc = new List<float>();
            public long Evals, Fill, Met;
        }

        IEnumerator Run()
        {
            Running = true;
            var d = Director.Instance;
            bool wasPaused = d.Paused;
            d.Paused = true;   // stop the live worlds stepping while we measure
            var sb = new StringBuilder();
            sb.AppendLine("n,policy,step_ms_mean,step_ms_p95,alloc_ms_mean,evals_mean,fill_mean," +
                          "budget_met_frac,kl_max_end,restorations,beh0,beh1,beh2,beh3");
            var table = d.AllocTable;
            bool matched = d.Budget == BudgetMode.Matched;
            foreach (int n in Sizes)
            {
                Progress = $"N={n}";
                // Both policies step in lockstep so that, cost-matched, PARITY's budget each
                // frame is the baseline's predicted spend on that same frame.
                var worlds = new CrowdWorld[2];
                for (int k = 0; k < 2; k++)
                {
                    worlds[k] = new CrowdWorld(k == 0 ? Policy.Baseline : Policy.Parity, n, d.Spec, 1u, table,
                                               d.TargetMs, d.Cap)
                    {
                        BudgetFrac = d.BudgetFrac, AbsoluteBudget = d.Budget == BudgetMode.Absolute,
                        TargetMs = d.TargetMs,
                    };
                }
                worlds[0].ApplyCalibration(d.BaseFloor, d.BaseTheta);
                worlds[1].ApplyCalibration(d.CalibFloor, CrowdWorld.CalibratedTheta);
                var acc = new[] { new Acc(), new Acc() };
                float lastB = 0f, lastP = 0f;
                for (int f = 0; f < WarmupFrames + MeasureFrames; f++)
                {
                    var (cp, yaw) = OrbitAt(d.Spec, f);
                    worlds[0].Step(cp, yaw, lastB);
                    lastB = worlds[0].StepMs;
                    worlds[1].MatchBudgetMs = matched ? worlds[0].PredictedSpendMs() : -1f;
                    worlds[1].Step(cp, yaw, lastP);
                    lastP = worlds[1].StepMs;
                    if (f >= WarmupFrames)
                    {
                        acc[0].Step.Add(worlds[0].StepMs);
                        if (worlds[0].StepMs <= worlds[0].BudgetMs) acc[0].Met++;
                        var p = worlds[1];
                        acc[1].Step.Add(p.StepMs);
                        acc[1].Alloc.Add(p.AllocMs);
                        acc[1].Evals += p.Last.Evals;
                        acc[1].Fill += p.Last.FillSteps;
                        if (!p.Last.Infeasible && p.Last.Cost <= p.BudgetMs * 1.0001f) acc[1].Met++;
                    }
                    // keep the editor responsive; the measurement is per-frame wall time
                    // of the crowd step itself, so yielding between frames does not bias it
                    if ((f & 15) == 0) yield return null;
                }
                for (int k = 0; k < 2; k++)
                {
                    var w = worlds[k];
                    var c = w.Counts;
                    sb.Append(n.ToString(CultureInfo.InvariantCulture)).Append(',').Append(w.Mode).Append(',')
                      .Append(F(Mean(acc[k].Step))).Append(',').Append(F(Pct(acc[k].Step, 0.95f))).Append(',')
                      .Append(F(k == 1 ? Mean(acc[k].Alloc) : 0f)).Append(',')
                      .Append(F(acc[k].Evals / (float)MeasureFrames)).Append(',')
                      .Append(F(acc[k].Fill / (float)MeasureFrames)).Append(',')
                      .Append(F(acc[k].Met / (float)MeasureFrames)).Append(',')
                      .Append(F(w.MaxDivergence())).Append(',')
                      .Append(w.Restorations).Append(',')
                      .Append(c[0, 0]).Append(',').Append(c[0, 1]).Append(',')
                      .Append(c[0, 2]).Append(',').Append(c[0, 3]).AppendLine();
                    w.Dispose();
                }
                yield return null;
            }
            // the plaza keeps the file name the recorded unity/bench/engine_bench.csv came from
            string file = d.Kind == SceneKind.Plaza ? "parity_engine_bench.csv" : $"parity_engine_bench_{d.Spec.Name}.csv";
            if (matched) file = file.Replace(".csv", "_matched.csv");
            string path = System.IO.Path.Combine(Application.persistentDataPath, file);
            System.IO.File.WriteAllText(path, sb.ToString());
            Debug.Log("PARITY benchmark written to " + path + "\n" + sb);
            Progress = "";
            Running = false;
            d.Paused = wasPaused;
        }

        static (Vector2, float) OrbitAt(SceneSpec s, int frame)
        {
            var focus = s.Focus;
            float r = 0.45f * Mathf.Min(s.Size.x, s.Size.y);
            float ang = (frame / 2400f) * Mathf.PI * 2f;
            var p = new Vector2(focus.x + r * Mathf.Cos(ang), focus.y + r * Mathf.Sin(ang));
            return (p, Mathf.Atan2(focus.y - p.y, focus.x - p.x));
        }

        static string F(float v) => v.ToString("G6", CultureInfo.InvariantCulture);

        static float Mean(List<float> v)
        {
            if (v.Count == 0) return 0f;
            double s = 0.0; for (int i = 0; i < v.Count; i++) s += v[i];
            return (float)(s / v.Count);
        }

        static float Pct(List<float> v, float p)
        {
            if (v.Count == 0) return 0f;
            var a = v.ToArray(); System.Array.Sort(a);
            return a[Mathf.Clamp(Mathf.RoundToInt(p * (a.Length - 1)), 0, a.Length - 1)];
        }
    }
}
