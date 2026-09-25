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
            foreach (int n in Sizes)
            {
                foreach (Policy pol in new[] { Policy.Baseline, Policy.Parity })
                {
                    Progress = $"N={n} {pol}";
                    var w = new CrowdWorld(pol, n, Director.SceneSize, 1u, table, d.TargetMs, d.Cap)
                    {
                        BudgetFrac = d.BudgetFrac, AbsoluteBudget = d.AbsoluteBudget, TargetMs = d.TargetMs,
                    };
                    w.ApplyCalibration(d.CalibFloor, CrowdWorld.CalibratedTheta);
                    var step = new List<float>(MeasureFrames);
                    var alloc = new List<float>(MeasureFrames);
                    long evals = 0, fill = 0, met = 0;
                    float last = 0f;

                    for (int f = 0; f < WarmupFrames + MeasureFrames; f++)
                    {
                        var (cp, yaw) = OrbitAt(f);
                        w.Step(cp, yaw, last);
                        last = w.StepMs;
                        if (f >= WarmupFrames)
                        {
                            step.Add(w.StepMs);
                            if (pol == Policy.Parity)
                            {
                                alloc.Add(w.AllocMs);
                                evals += w.Last.Evals;
                                fill += w.Last.FillSteps;
                                if (!w.Last.Infeasible && w.Last.Cost <= w.BudgetMs) met++;
                            }
                            else if (w.StepMs <= w.BudgetMs) met++;
                        }
                        // keep the editor responsive; the measurement is per-frame wall time
                        // of the crowd step itself, so yielding between frames does not bias it
                        if ((f & 15) == 0) yield return null;
                    }

                    var c = w.Counts;
                    sb.Append(n.ToString(CultureInfo.InvariantCulture)).Append(',').Append(pol).Append(',')
                      .Append(F(Mean(step))).Append(',').Append(F(Pct(step, 0.95f))).Append(',')
                      .Append(F(pol == Policy.Parity ? Mean(alloc) : 0f)).Append(',')
                      .Append(F(evals / (float)MeasureFrames)).Append(',')
                      .Append(F(fill / (float)MeasureFrames)).Append(',')
                      .Append(F(met / (float)MeasureFrames)).Append(',')
                      .Append(F(w.MaxDivergence())).Append(',')
                      .Append(w.Restorations).Append(',')
                      .Append(c[0, 0]).Append(',').Append(c[0, 1]).Append(',')
                      .Append(c[0, 2]).Append(',').Append(c[0, 3]).AppendLine();
                    w.Dispose();
                    yield return null;
                }
            }

            string path = System.IO.Path.Combine(Application.persistentDataPath, "parity_engine_bench.csv");
            System.IO.File.WriteAllText(path, sb.ToString());
            Debug.Log("PARITY benchmark written to " + path + "\n" + sb);
            Progress = "";
            Running = false;
            d.Paused = wasPaused;
        }

        static (Vector2, float) OrbitAt(int frame)
        {
            var focus = new Vector2(60f, 60f);
            float r = 0.45f * Mathf.Min(Director.SceneSize.x, Director.SceneSize.y);
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
