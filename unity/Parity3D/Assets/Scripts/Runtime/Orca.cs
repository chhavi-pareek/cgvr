// ORCA local collision avoidance (van den Berg, Guy, Lin and Manocha 2011), agents only: the
// port of sim/orca.py, which is its oracle (tools/gen_oracle_cases.py, orca section). Doubles
// throughout so the port and the oracle agree to rounding. One instance per thread: the line
// buffers are its only state.
using System;

namespace Parity
{
    public sealed class Orca
    {
        public const double Eps = 1e-5;
        public const double Radius = 0.3;   // m, body radius
        public const double Reach = 2.5;    // m, neighbour search radius (one grid cell)
        public const double Tau = 1.5;      // s, time horizon

        struct Line { public double Px, Py, Dx, Dy; }

        Line[] lines, proj;

        public Orca(int maxNeighbours)
        {
            lines = new Line[Math.Max(maxNeighbours, 1)];
            proj = new Line[Math.Max(maxNeighbours, 1)];
        }

        static double Det(double ax, double ay, double bx, double by) => ax * by - ay * bx;

        /// <summary>The new velocity of an agent at p moving at v, preferring pref, against m
        /// neighbours (positions nx, ny, velocities nvx, nvy).</summary>
        public void NewVelocity(double px, double py, double vx, double vy, double prefX, double prefY, double maxSpeed,
                                double[] nx, double[] ny, double[] nvx, double[] nvy, int m, double dt,
                                out double rx, out double ry)
        {
            if (lines.Length < m) { lines = new Line[m]; proj = new Line[m]; }
            double invTau = 1.0 / Tau, r = 2.0 * Radius, r2 = r * r;
            for (int j = 0; j < m; j++)
            {
                double relX = nx[j] - px, relY = ny[j] - py;
                double rvX = vx - nvx[j], rvY = vy - nvy[j];
                double dist2 = relX * relX + relY * relY;
                double dirX, dirY, uX, uY;
                if (dist2 > r2)
                {
                    double wX = rvX - invTau * relX, wY = rvY - invTau * relY;
                    double w2 = wX * wX + wY * wY;
                    double dot1 = wX * relX + wY * relY;
                    if (dot1 < 0.0 && dot1 * dot1 > r2 * w2)
                    {
                        // the velocity obstacle's cut-off circle
                        double wl = Math.Sqrt(w2), ux = wX / wl, uy = wY / wl;
                        dirX = uy; dirY = -ux;
                        uX = (r * invTau - wl) * ux; uY = (r * invTau - wl) * uy;
                    }
                    else
                    {
                        // one of its legs
                        double leg = Math.Sqrt(dist2 - r2);
                        if (Det(relX, relY, wX, wY) > 0.0)
                        {
                            dirX = (relX * leg - relY * r) / dist2; dirY = (relX * r + relY * leg) / dist2;
                        }
                        else
                        {
                            dirX = -(relX * leg + relY * r) / dist2; dirY = -(-relX * r + relY * leg) / dist2;
                        }
                        double dot2 = rvX * dirX + rvY * dirY;
                        uX = dot2 * dirX - rvX; uY = dot2 * dirY - rvY;
                    }
                }
                else
                {
                    // already overlapping: resolve within one step
                    double invDt = 1.0 / dt;
                    double wX = rvX - invDt * relX, wY = rvY - invDt * relY;
                    double wl = Math.Sqrt(wX * wX + wY * wY), ux = wX / wl, uy = wY / wl;
                    dirX = uy; dirY = -ux;
                    uX = (r * invDt - wl) * ux; uY = (r * invDt - wl) * uy;
                }
                lines[j] = new Line { Px = vx + 0.5 * uX, Py = vy + 0.5 * uY, Dx = dirX, Dy = dirY };
            }
            int fail = Lp2(lines, m, maxSpeed, prefX, prefY, false, out rx, out ry);
            if (fail < m) Lp3(m, fail, maxSpeed, ref rx, ref ry);
        }

        static bool Lp1(Line[] ls, int n, double radius, double optX, double optY, bool dirOpt, out double rx, out double ry)
        {
            rx = ry = 0.0;
            var ln = ls[n];
            double dot = ln.Px * ln.Dx + ln.Py * ln.Dy;
            double disc = dot * dot + radius * radius - (ln.Px * ln.Px + ln.Py * ln.Py);
            if (disc < 0.0) return false;
            double s = Math.Sqrt(disc), tLeft = -dot - s, tRight = -dot + s;
            for (int i = 0; i < n; i++)
            {
                double den = Det(ln.Dx, ln.Dy, ls[i].Dx, ls[i].Dy);
                double num = Det(ls[i].Dx, ls[i].Dy, ln.Px - ls[i].Px, ln.Py - ls[i].Py);
                if (Math.Abs(den) <= Eps)
                {
                    if (num < 0.0) return false;
                    continue;
                }
                double t = num / den;
                if (den >= 0.0) tRight = Math.Min(tRight, t); else tLeft = Math.Max(tLeft, t);
                if (tLeft > tRight) return false;
            }
            double tt;
            if (dirOpt) tt = optX * ln.Dx + optY * ln.Dy > 0.0 ? tRight : tLeft;
            else tt = Math.Min(Math.Max(ln.Dx * (optX - ln.Px) + ln.Dy * (optY - ln.Py), tLeft), tRight);
            rx = ln.Px + tt * ln.Dx; ry = ln.Py + tt * ln.Dy;
            return true;
        }

        static int Lp2(Line[] ls, int m, double radius, double optX, double optY, bool dirOpt, out double rx, out double ry)
        {
            if (dirOpt) { rx = optX * radius; ry = optY * radius; }
            else if (optX * optX + optY * optY > radius * radius)
            {
                double l = Math.Sqrt(optX * optX + optY * optY);
                rx = optX / l * radius; ry = optY / l * radius;
            }
            else { rx = optX; ry = optY; }
            for (int i = 0; i < m; i++)
                if (Det(ls[i].Dx, ls[i].Dy, ls[i].Px - rx, ls[i].Py - ry) > 0.0)
                {
                    if (!Lp1(ls, i, radius, optX, optY, dirOpt, out double nx, out double ny)) return i;
                    rx = nx; ry = ny;
                }
            return m;
        }

        void Lp3(int m, int begin, double radius, ref double rx, ref double ry)
        {
            double distance = 0.0;
            for (int i = begin; i < m; i++)
            {
                var li = lines[i];
                if (Det(li.Dx, li.Dy, li.Px - rx, li.Py - ry) <= distance) continue;
                int np = 0;
                for (int j = 0; j < i; j++)
                {
                    var lj = lines[j];
                    double det = Det(li.Dx, li.Dy, lj.Dx, lj.Dy);
                    double ptX, ptY;
                    if (Math.Abs(det) <= Eps)
                    {
                        if (li.Dx * lj.Dx + li.Dy * lj.Dy > 0.0) continue;
                        ptX = 0.5 * (li.Px + lj.Px); ptY = 0.5 * (li.Py + lj.Py);
                    }
                    else
                    {
                        double t = Det(lj.Dx, lj.Dy, li.Px - lj.Px, li.Py - lj.Py) / det;
                        ptX = li.Px + t * li.Dx; ptY = li.Py + t * li.Dy;
                    }
                    double dx = lj.Dx - li.Dx, dy = lj.Dy - li.Dy, dl = Math.Sqrt(dx * dx + dy * dy);
                    proj[np++] = new Line { Px = ptX, Py = ptY, Dx = dx / dl, Dy = dy / dl };
                }
                if (Lp2(proj, np, radius, -li.Dy, li.Dx, true, out double nx, out double ny) >= np) { rx = nx; ry = ny; }
                distance = Det(li.Dx, li.Dy, li.Px - rx, li.Py - ry);
            }
        }

        /// <summary>Keeps the k nearest of a stream of candidates, nearest first, ties by index --
        /// the order sim/orca.py::neighbours returns. Returns the new count.</summary>
        public static int Insert(int[] idx, double[] d2, int count, int k, int j, double dj)
        {
            int at = count;
            while (at > 0 && (d2[at - 1] > dj || (d2[at - 1] == dj && idx[at - 1] > j))) at--;
            if (at >= k) return count;
            int last = Math.Min(count, k - 1);
            for (int s = last; s > at; s--) { idx[s] = idx[s - 1]; d2[s] = d2[s - 1]; }
            idx[at] = j; d2[at] = dj;
            return Math.Min(count + 1, k);
        }
    }
}
