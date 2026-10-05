"""Palace demo video: captions on the frames ParityVideo renders, then H.264.

    python -m figures.palace_video <frames dir> [--out figures/out/palace_demo.mp4]

Frames and captions.csv come from the Unity editor script ParityVideo (unity/Parity3D/Assets/
Editor/ParityVideo.cs). A title card opens the video and an end card carries the 48,000-seat plan
comparison from bench/palace.py (docs/site/mysore_palace.md), so the clip stands on its own in a
slide deck.
"""
import argparse
import csv
import io
import pathlib
import subprocess

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

FPS, W, H = 30, 1920, 1080
INK, MUTED, BOX = (250, 250, 248), (205, 204, 198), (10, 10, 10, 165)
FONT = font_manager.findfont("DejaVu Sans")
BOLD = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
GATE = {"north": "North", "south": "South (Varaha)", "east_n": "East side N", "east_s": "East side S", "west_n": "West"}
END = [("Everyone to the nearest gate", "105.6 min"), ("Seats shared by gate width", "85.4 min"),
       ("Seats shared by the gate rates the twin measured", "54.3 min")]


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def panel(draw, xy, lines, pad=14):
    """Text block on a translucent box; lines are (text, size, bold, colour)."""
    x, y = xy
    width = max(draw.textlength(t, font=font(s, b)) for t, s, b, _ in lines) + 2 * pad
    height = sum(s + 8 for _, s, _, _ in lines) + 2 * pad - 8
    draw.rounded_rectangle([x, y, x + width, y + height], 10, fill=BOX)
    y += pad
    for t, s, b, c in lines:
        draw.text((x + pad, y), t, font=font(s, b), fill=c)
        y += s + 8


def clock(seconds):
    s = abs(int(round(seconds)))
    sign = "-" if seconds < 0 and s > 0 else ""
    return f"{sign}{s // 60}:{s % 60:02d}"


def card(title, lines):
    im = Image.new("RGB", (W, H), (18, 20, 24))
    d = ImageDraw.Draw(im)
    d.text((140, 330), title, font=font(54, True), fill=INK)
    y = 430
    for t, s, c in lines:
        d.text((140, y), t, font=font(s), fill=c)
        y += s + 22
    return im


def caption(im, row):
    d = ImageDraw.Draw(im, "RGBA")
    part, t, n = int(row["part"]), float(row["t_release_s"]), int(row["n"])
    title = (f"Mysore Palace grounds, Jamboo Savari: {n:,} seats (one sector)", 30, True, INK)
    if part == 1:
        phase = "Seated: the procession has not left yet" if t < 0 else "The procession has left: everyone stands up and heads for a gate"
        panel(d, (24, 24), [title, (f"{phase}   ·   {clock(t)}   ·   real time", 24, False, MUTED)])
        panel(d, (24, H - 120), [("MassLOD-style baseline", 28, True, INK),
                                 (f"worst behaviour drift {float(row['base_over_cap']):.2f} x cap  (orange rings)", 22, False, MUTED)])
        panel(d, (W // 2 + 24, H - 120), [("PARITY", 28, True, INK),
                                          (f"worst behaviour drift {float(row['par_over_cap']):.2f} x cap", 22, False, MUTED)])
    else:
        gates = dict(g.split(":") for g in row["gates"].split())
        out = int(row["out_par"])
        panel(d, (24, 24), [title, (f"PARITY, fast-forward x{int(float(row['speed']))}   ·   {clock(t)} after the procession left   ·   "
                                    f"{out:,} of {n:,} out", 24, False, MUTED)])
        panel(d, (24, H - 150), [("Through each gate", 24, True, INK),
                                 ("   ·   ".join(f"{GATE.get(k, k)} {int(v):,}" for k, v in gates.items()), 22, False, MUTED),
                                 ("Seats shared by measured gate rates. Gate widths and seat layout are assumptions.", 18, False, MUTED)])
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames")
    p.add_argument("--out", default="figures/out/palace_demo.mp4")
    a = p.parse_args()
    src = pathlib.Path(a.frames)
    rows = list(csv.DictReader(open(src / "captions.csv")))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", "-preset", "slow",
                           "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    def put(im, times=1):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        for _ in range(times):
            ff.stdin.write(buf.getvalue())

    n = int(rows[0]["n"])
    put(card("A real-time digital twin of Mysore Palace crowds during Dasara",
             [(f"Jamboo Savari day: {n:,} seated people (one sector of the 48,000-seat grounds) stand up at once", 30, INK),
              ("and walk to the gates. Left: a MassLOD-style crowd. Right: PARITY, same budget, same camera.", 30, INK),
              ("Site from OpenStreetMap; every number in the study is published, from the literature, or a stated assumption.", 24, MUTED)]),
        int(3.5 * FPS))
    for row in rows:
        im = Image.open(src / f"frame_{int(row['frame']):05d}.jpg").convert("RGB")
        put(caption(im, row))
    put(card("48,000 seats, everyone released at once: time until all are out",
             [(f"{name}:  {t}", 34, INK) for name, t in END]
             + [("From the crowd model in bench/palace.py, one run each. Compare the plans, not the absolute minutes:", 24, MUTED),
                ("gate widths and seat layout are assumptions, and the model is deliberately conservative.", 24, MUTED)]),
        int(5 * FPS))
    ff.stdin.close()
    ff.wait()
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
