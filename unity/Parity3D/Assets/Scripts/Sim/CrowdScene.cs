// The three scenes of sim/scenes.py, ported so the demo can show each one. The geometry and
// the goal logic follow the Python classes; the rates come from each scene's own phase 7
// calibration (bench/logs/phase7_<scene>_calib.npz), so every scene gets its own cap and its
// own admission rate exactly as sim/tiered.py builds them:
//
//   cap   = 300 * e_sur,  e_sur = e_rate . ctx_freq          (occupancy-weighted mean)
//   admit at e_max = max e_rate over contexts above 1% occupancy
//
// Nothing in the environment art is walkable geometry the sim does not know about: every
// wall below is one the Python constrain() enforces, and every prop sits outside the
// walkable rectangle or overhead.
using UnityEngine;

namespace Parity
{
    public enum SceneKind { Plaza, Hub, Corridor }

    public sealed class SceneSpec
    {
        public SceneKind Kind;
        public string Name, Title;
        public Vector2 Size, Focus;     // bench/camerapaths.py FOCUS
        public double ESur, EMax;
        public double[] ERate;
        public int DefaultAgents;
        public float CamHeight;

        public float Cap => (float)(300.0 * ESur);

        static readonly SceneSpec[] All =
        {
            new SceneSpec
            {
                Kind = SceneKind.Plaza, Name = "plaza", Title = "City plaza",
                Size = new Vector2(120f, 120f), Focus = new Vector2(60f, 60f),
                // e_rate [0.00393 0.01033 0.05716 0.00393], ctx_freq [0 .938 .062 0]
                ESur = ParityTable.PlazaESur, EMax = ParityTable.PlazaEMax, ERate = ParityTable.PlazaERate,
                DefaultAgents = 1200, CamHeight = 14f,
            },
            new SceneSpec
            {
                Kind = SceneKind.Hub, Name = "hub", Title = "Station concourse",
                Size = new Vector2(100f, 40f), Focus = new Vector2(60f, 20f),
                // ctx_freq [.028 .928 .045 0]
                ESur = 0.01315, EMax = 0.06179, ERate = new[] { 0.0201, 0.0106, 0.06179, 0.00399 },
                DefaultAgents = 800, CamHeight = 8.5f,
            },
            new SceneSpec
            {
                Kind = SceneKind.Corridor, Name = "corridor", Title = "Evacuation corridor",
                Size = new Vector2(60f, 10f), Focus = new Vector2(30f, 5f),
                // ctx_freq [0 .828 .093 .079]
                ESur = 0.01828, EMax = 0.05098, ERate = new[] { 0.00431, 0.01202, 0.04615, 0.05098 },
                DefaultAgents = 260, CamHeight = 11f,
            },
        };

        public static SceneSpec Get(SceneKind k) => All[(int)k];
    }

    /// <summary>Spawn, goal and wall logic for one scene, acting on a CrowdWorld's arrays.</summary>
    public abstract class CrowdScene
    {
        public const float Arrive = 1.0f;   // sim/scenes.py ARRIVE
        public readonly SceneSpec Spec;
        protected CrowdScene(SceneSpec s) { Spec = s; }

        public static CrowdScene Make(SceneSpec s)
        {
            switch (s.Kind)
            {
                case SceneKind.Hub: return new HubScene(s);
                case SceneKind.Corridor: return new CorridorScene(s);
                default: return new PlazaScene(s);
            }
        }

        public virtual void Begin(CrowdWorld w) { }
        /// <summary>Position, goal and walking speed of one agent at construction.</summary>
        public abstract void SpawnOne(CrowdWorld w, int i, ref Rng r);
        /// <summary>After the fine step: walls, with the pre-step positions for crossing tests.</summary>
        public virtual void Constrain(CrowdWorld w)
        {
            var sz = Spec.Size;
            for (int i = 0; i < w.N; i++)
                w.Pos[i] = new Vector2(Mathf.Clamp(w.Pos[i].x, 0f, sz.x), Mathf.Clamp(w.Pos[i].y, 0f, sz.y));
        }
        /// <summary>Arrivals, respawns and scheduled goal changes.</summary>
        public abstract void AfterStep(CrowdWorld w, int frame, ref Rng r);

        protected static bool Reached(CrowdWorld w, int i) => (w.Goal[i] - w.Pos[i]).sqrMagnitude < Arrive * Arrive;
        protected static bool CoreDone(CrowdWorld w, int i) => w.CoreS[i] >= w.CoreLen[i];
    }

    sealed class PlazaScene : CrowdScene
    {
        public PlazaScene(SceneSpec s) : base(s) { }

        public override void SpawnOne(CrowdWorld w, int i, ref Rng r)
        {
            w.Pos[i] = new Vector2(r.Range(0f, Spec.Size.x), r.Range(0f, Spec.Size.y));
            w.SetGoal(i, RandomGoal(ref r));
            w.Speed[i] = r.Range(1.0f, 1.6f);
        }

        Vector2 RandomGoal(ref Rng r) => new Vector2(r.Range(2f, Spec.Size.x - 2f), r.Range(2f, Spec.Size.y - 2f));

        public override void AfterStep(CrowdWorld w, int frame, ref Rng r)
        {
            for (int i = 0; i < w.N; i++)
                if (Reached(w, i) || CoreDone(w, i)) w.SetGoal(i, RandomGoal(ref r));
        }
    }

    /// <summary>sim/scenes.py Hub: a 100 x 40 m concourse with one service window at (60, 20).
    /// A fifth of the crowd (at most 60) queues for it; everyone else crosses to the exits on
    /// the far wall and re-enters through the arcade at x = 0.</summary>
    sealed class HubScene : CrowdScene
    {
        static readonly Vector2 Window = new Vector2(60f, 20f);
        public const float Spacing = 0.8f;
        public const int Service = 45;
        int qmax;

        public HubScene(SceneSpec s) : base(s) { }

        public static Vector2 SlotPos(int slot) => new Vector2(Window.x - 1f - slot * Spacing, Window.y);
        Vector2 Exit(ref Rng r) => new Vector2(Spec.Size.x, r.Range(0f, Spec.Size.y));

        public override void Begin(CrowdWorld w) { qmax = Mathf.Max(1, Mathf.Min(w.N / 5, 60)); }

        public override void SpawnOne(CrowdWorld w, int i, ref Rng r)
        {
            w.Speed[i] = r.Range(1.0f, 1.6f);
            w.Pos[i] = new Vector2(r.Range(0f, Spec.Size.x * 0.6f), r.Range(0f, Spec.Size.y));
            if (i < qmax)
            {
                w.Queuer[i] = true;
                w.Slot[i] = i;
                w.Pos[i] = SlotPos(i) + new Vector2(r.Range(-0.3f, 0.3f), r.Range(-0.3f, 0.3f));
                w.SetGoal(i, SlotPos(i));
            }
            else w.SetGoal(i, Exit(ref r));
        }

        public override void AfterStep(CrowdWorld w, int frame, ref Rng r)
        {
            if (frame % Service == 0)
            {
                int front = -1, best = int.MaxValue;
                for (int i = 0; i < w.N; i++)
                    if (w.Slot[i] >= 0 && w.Slot[i] < best) { best = w.Slot[i]; front = i; }
                if (front >= 0)
                {
                    for (int i = 0; i < w.N; i++)
                    {
                        if (w.Slot[i] < 0) continue;
                        if (i == front) { w.Slot[i] = -1; w.SetGoal(i, Exit(ref r)); }
                        else { w.Slot[i]--; w.SetGoal(i, SlotPos(w.Slot[i])); }
                    }
                }
            }

            int nq = 0;
            for (int i = 0; i < w.N; i++) if (w.Slot[i] >= 0) nq++;
            for (int i = 0; i < w.N; i++)
            {
                if (w.Slot[i] < 0 && Reached(w, i))
                {
                    // out through the far wall, back in through the arcade
                    w.Pos[i] = new Vector2(r.Range(0f, 5f), r.Range(0f, Spec.Size.y));
                    if (w.Queuer[i] && nq < qmax) { w.Slot[i] = nq; nq++; w.SetGoal(i, SlotPos(w.Slot[i])); }
                    else w.SetGoal(i, Exit(ref r));
                }
                else if (CoreDone(w, i)) w.SetGoal(i, w.Goal[i]);   // re-seat the core, same goal
            }
        }

        public int QueueLength(CrowdWorld w)
        {
            int n = 0;
            for (int i = 0; i < w.N; i++) if (w.Slot[i] >= 0) n++;
            return n;
        }
    }

    /// <summary>sim/scenes.py Corridor: 60 x 10 m with a partition at x = 30 and one 1.6 m
    /// door in it. Everyone spawns on the left and leaves at x = 58.
    ///
    /// The Python aims straight at a via point past the door and lets constrain() slide agents
    /// along the wall. The core here is a straight segment, and a segment from the spawn to
    /// that via point can cut the partition, so the route is split at the door mouth: spawn
    /// -> mouth -> via -> exit. Every segment then lies on one side of the wall or inside the
    /// doorway, and a surrogate riding the core never passes through the wall.</summary>
    public sealed class CorridorScene : CrowdScene
    {
        public const float DoorX0 = 29.7f, DoorX1 = 30.3f, Gap0 = 4.2f, Gap1 = 5.8f, ExitX = 58f;
        static readonly Vector2 Via = new Vector2(31f, 5f);
        const float MouthX = 28.8f;

        public CorridorScene(SceneSpec s) : base(s) { }

        static Vector2 Mouth(float y) => new Vector2(MouthX, Mathf.Clamp(y, 4.7f, 5.3f));

        public override void SpawnOne(CrowdWorld w, int i, ref Rng r)
        {
            w.Pos[i] = new Vector2(r.Range(0f, 28f), r.Range(0f, 10f));
            w.Final[i] = new Vector2(ExitX, r.Range(3f, 7f));
            w.Speed[i] = r.Range(1.0f, 1.6f);
            w.Stage[i] = 0;
            w.SetGoal(i, Mouth(w.Pos[i].y));
        }

        public override void Constrain(CrowdWorld w)
        {
            base.Constrain(w);
            for (int i = 0; i < w.N; i++)
            {
                Vector2 p = w.Pos[i], q = w.PrevPos[i];
                bool outside = p.y < Gap0 || p.y > Gap1;
                // sim/scenes.py only stops the forward crossing; separation can also push an
                // agent back through the wall from the right, so that side is closed too
                if (outside && p.x >= DoorX0 && q.x < DoorX0) p.x = DoorX0 - 0.01f;
                else if (outside && p.x <= DoorX1 && q.x > DoorX1) p.x = DoorX1 + 0.01f;
                if (p.x >= DoorX0 && p.x <= DoorX1) p.y = Mathf.Clamp(p.y, Gap0, Gap1);
                w.Pos[i] = p;
            }
        }

        public override void AfterStep(CrowdWorld w, int frame, ref Rng r)
        {
            for (int i = 0; i < w.N; i++)
            {
                Vector2 p = w.Pos[i];
                switch (w.Stage[i])
                {
                    case 0:
                        if (Reached(w, i) || p.x >= 29.2f)
                        {
                            // the core restarts from the mouth, not from wherever the crowd
                            // pressed this agent, so the doorway segment never touches the wall
                            var m = w.Goal[i];
                            w.Stage[i] = 1;
                            w.SetGoal(i, Via);
                            w.CoreFrom[i] = m; w.CoreP[i] = m;
                            w.CoreLen[i] = Mathf.Max((Via - m).magnitude, 0.001f);
                        }
                        else if (CoreDone(w, i)) w.SetGoal(i, w.Goal[i]);
                        break;
                    case 1:
                        if (p.x > DoorX1) { w.Stage[i] = 2; w.SetGoal(i, w.Final[i]); }
                        break;   // the doorway segment is never re-seated: see the class note
                    default:
                        if (p.x > ExitX)
                        {
                            w.Pos[i] = new Vector2(r.Range(0f, 10f), r.Range(0f, 10f));
                            w.Final[i] = new Vector2(ExitX, r.Range(3f, 7f));
                            w.Stage[i] = 0;
                            w.SetGoal(i, Mouth(w.Pos[i].y));
                        }
                        else if (CoreDone(w, i)) w.SetGoal(i, w.Goal[i]);
                        break;
                }
            }
        }

        /// <summary>Agents standing inside the partition outside the doorway: must stay 0.</summary>
        public static int WallViolations(CrowdWorld w)
        {
            int n = 0;
            for (int i = 0; i < w.N; i++)
            {
                var p = w.Pos[i];
                if (p.x > DoorX0 + 1e-3f && p.x < DoorX1 - 1e-3f && (p.y < Gap0 - 1e-3f || p.y > Gap1 + 1e-3f)) n++;
            }
            return n;
        }
    }
}
