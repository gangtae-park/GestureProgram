"""Hand-skeleton rendering, trial playback, and Jackknife template export.

Three uses:

1. Imported by pilot_receiver.py / pilot_overlay.py: draw_hands_view()
   renders the both-hands skeleton panel.

2. Interactive playback + trim + export of a recorded trial:

       python pilot_hands.py pilot_data/P01/Ask/trial_01
       python pilot_hands.py pilot_data/P01/Ask/trial_01 --hand right --label Ask

   Keys:
       SPACE  pause/resume          , / .  step one sample (pauses)
       S      set trim START here   E      set trim END here
       X      export trimmed segment as a Jackknife template
       [ / ]  slower / faster       R      restart      Q  quit

3. Non-interactive export (for batch scripting):

       python pilot_hands.py pilot_data/P01/Ask/trial_01 \
           --export --start 0.8 --end 2.1 --hand right --label Ask

Template format matches template_analyze/gesture_templates_unified.json:

    {"templates": [{"label": "...", "frames": [{"values": [63 floats]}]}]}

values = 63 floats: the MediaPipe-style 21-joint subset of the 25 recorded
joints (finger metacarpals dropped, see TEMPLATE_JOINT_INDICES), with the
WRIST subtracted from every joint so the wrist sits at (0,0,0) and everything
else is wrist-relative — the same convention Unity's HandFeatureSource used
when the original templates were recorded (positions relative to jointOrigin,
rotated into the camera frame). Samples where the chosen hand is untracked
are skipped. Exports APPEND to --out (default: pilot_templates.json inside the
pilot_data directory the trial came from) and log provenance (source csv,
trim window, hand) to pilot_templates_log.csv next to it.
"""
import argparse
import csv
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 25 joints per hand: MediaPipe-style 21 plus the four finger metacarpals
# ("_meta"). Order matches PilotSender.JointsPerHand / pilot_receiver.py.
NUM_JOINTS = 25
JOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_meta", "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_meta", "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_meta", "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_meta", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

# Bone graph: wrist -> each metacarpal -> finger chain, plus a knuckle bridge
# across the MCPs.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),                     # thumb
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),             # index
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),    # middle
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),    # ring
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),    # pinky
    (6, 11), (11, 16), (16, 21),                        # knuckle bridge
]
FINGERTIPS = {4, 9, 14, 19, 24}
WRIST = 0

# Jackknife templates stay in the original 21-joint (63-value) layout:
# wrist, thumb x4, then each finger's mcp/pip/dip/tip -- the finger
# metacarpals are recorded for recognizer-parameter analysis but dropped at
# export so templates remain compatible with gesture_templates_unified.json.
TEMPLATE_JOINT_INDICES = [
    0,                   # wrist
    1, 2, 3, 4,          # thumb
    6, 7, 8, 9,          # index mcp..tip
    11, 12, 13, 14,      # middle
    16, 17, 18, 19,      # ring
    21, 22, 23, 24,      # pinky
]

LEFT_COLOR = (80, 200, 255)    # BGR amber   -> left hand
RIGHT_COLOR = (255, 170, 80)   # BGR sky     -> right hand
GAZE_COLOR = (255, 200, 0)
BG_COLOR = (25, 25, 25)
GRID_COLOR = (48, 48, 48)
TEXT_COLOR = (200, 200, 200)
LOST_COLOR = (90, 90, 110)
TRIM_COLOR = (120, 255, 120)

MIN_Z = 0.05  # ignore joints closer than this to the camera plane


def draw_gaze_marker(img, center, radius=12, color=GAZE_COLOR,
                     thickness=1, center_gap=4, tick_overhang=5):
    """Hollow-circle reticle with a center gap so the exact gaze pixel stays
    visible. Shared by the live preview, the overlay video, and playback."""
    x, y = center
    cv2.circle(img, (x, y), radius, color, thickness, cv2.LINE_AA)
    outer = radius + tick_overhang
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img,
                 (x + dx * center_gap, y + dy * center_gap),
                 (x + dx * outer, y + dy * outer),
                 color, thickness, cv2.LINE_AA)


def _project(p, w, h, f, cy):
    """Camera-local (x right, y up, z forward) -> pixel, or None if behind.
    cy is the pixel row where the straight-ahead axis projects."""
    x, y, z = p
    if z <= MIN_Z:
        return None
    return int(round(w / 2 + f * x / z)), int(round(cy - f * y / z))


def _draw_hand(canvas, joints, color, f, cy):
    h, w = canvas.shape[:2]
    pts = [_project(j, w, h, f, cy) for j in joints]
    for a, b in CONNECTIONS:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(canvas, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        if pt is None:
            continue
        if i == WRIST:
            cv2.circle(canvas, pt, 6, color, -1, cv2.LINE_AA)
        elif i in FINGERTIPS:
            cv2.circle(canvas, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_hands_view(size, left_joints, right_joints, left_tracked, right_tracked,
                    gaze_dir=None, fov_deg=120.0, view_center_y=0.25):
    """Render both hands (camera-local (N,3) arrays) into a BGR canvas.

    Pass left_joints/right_joints as None or all-zeros when untracked.
    gaze_dir is an optional camera-local unit direction, drawn as a red ring.
    view_center_y is the height fraction where the straight-ahead axis
    projects; hands sit well below eye level, so placing it above the middle
    (0.35) keeps them framed instead of hugging the bottom edge.
    """
    w, h = size
    canvas = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)
    f = (w / 2) / math.tan(math.radians(fov_deg) / 2)
    cy = h * view_center_y

    cv2.line(canvas, (w // 2, 0), (w // 2, h), GRID_COLOR, 1)
    cv2.line(canvas, (0, int(cy)), (w, int(cy)), GRID_COLOR, 1)

    if gaze_dir is not None:
        pt = _project(gaze_dir, w, h, f, cy)
        if pt is not None:
            draw_gaze_marker(canvas, pt, radius=9)

    for label, joints, tracked, color, x0 in (
        ("LEFT", left_joints, left_tracked, LEFT_COLOR, 10),
        ("RIGHT", right_joints, right_tracked, RIGHT_COLOR, w - 150),
    ):
        ok = tracked and joints is not None and np.any(joints)
        if ok:
            _draw_hand(canvas, joints, color, f, cy)
            z = float(joints[WRIST][2])
            cv2.putText(canvas, f"{label} z={z:.2f}m", (x0, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"{label} lost", (x0, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, LOST_COLOR, 1, cv2.LINE_AA)

    return canvas


# =================== Trial loading ===================
def _load_trial(samples_path: Path):
    """Parse a trial_XX_samples.csv into playback/export arrays."""
    times, recv_t, gaze, gaze_ok, lt, rt, lj, rj = [], [], [], [], [], [], [], []
    with samples_path.open("r", encoding="utf-8") as fobj:
        for row in csv.DictReader(fobj):
            times.append(float(row["sender_time"]))
            recv_t.append(float(row["receiver_time"]))
            gaze_ok.append(row["gaze_tracked"] == "1")
            gaze.append([float(row["gaze_dir_x"]), float(row["gaze_dir_y"]),
                         float(row["gaze_dir_z"])])
            lt.append(row["left_tracked"] == "1")
            rt.append(row["right_tracked"] == "1")
            lj.append([float(row[f"L_{n}_{a}"]) for n in JOINT_NAMES for a in "xyz"])
            rj.append([float(row[f"R_{n}_{a}"]) for n in JOINT_NAMES for a in "xyz"])
    if not times:
        sys.exit(f"No rows in {samples_path}")
    return {
        "t": np.asarray(times) - times[0],
        "recv_t": np.asarray(recv_t),
        "gaze": np.asarray(gaze),
        "gaze_ok": np.asarray(gaze_ok),
        "lt": np.asarray(lt), "rt": np.asarray(rt),
        "lj": np.asarray(lj).reshape(-1, NUM_JOINTS, 3),
        "rj": np.asarray(rj).reshape(-1, NUM_JOINTS, 3),
        "name": samples_path.stem.replace("_samples", ""),
        "where": samples_path.parent,
        "source": samples_path,
    }


# =================== Jackknife template export ===================
def default_template_path(samples_path: Path) -> Path:
    """pilot_templates.json inside the pilot_data tree the trial belongs to
    (falls back to the trial's grandparent when no pilot_data ancestor)."""
    for parent in samples_path.resolve().parents:
        if parent.name == "pilot_data":
            return parent / "pilot_templates.json"
    return samples_path.resolve().parent.parent / "pilot_templates.json"


def _dump_templates(data: dict, out_path: Path):
    """Indented JSON for readability, but each frame stays on ONE line
    ({"values": [63 floats]}) so a template is scannable instead of being
    thousands of one-number lines."""
    text = json.dumps(data, indent=2)
    text = re.sub(
        r'\{\s*"values": \[[^\]]*\]\s*\}',
        lambda m: re.sub(r"\s+", " ", m.group(0)).replace("[ ", "[").replace(" ]", "]").replace("{ ", "{").replace(" }", "}"),
        text,
    )
    with out_path.open("w", encoding="utf-8") as f:
        f.write(text)


def export_template(d, t0: float, t1: float, hand: str, label: str, out_path: Path):
    """Append the [t0, t1] segment of one hand as a wrist-relative Jackknife
    template to out_path. Returns the number of exported frames (0 = nothing
    usable in the window)."""
    joints_all = d["lj"] if hand == "left" else d["rj"]
    tracked = d["lt"] if hand == "left" else d["rt"]

    mask = (d["t"] >= t0 - 1e-9) & (d["t"] <= t1 + 1e-9) & tracked
    sel = joints_all[mask]
    if len(sel) < 2:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 21-joint subset (drop finger metacarpals), wrist-relative, 63 values.
    frames = [
        {"values": (j[TEMPLATE_JOINT_INDICES] - j[WRIST]).reshape(-1).tolist()}
        for j in sel
    ]

    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("templates", [])
    else:
        data = {"templates": []}
    data["templates"].append({"label": label, "frames": frames})
    _dump_templates(data, out_path)

    # Provenance sidecar: the template JSON itself must stay loader-compatible
    # (label + frames only), so source/trim info goes into a log next to it.
    log_path = out_path.with_name(out_path.stem + "_log.csv")
    new_log = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["exported_at", "template_index", "label", "hand",
                        "source_csv", "trim_start_s", "trim_end_s", "num_frames"])
        w.writerow([datetime.now().isoformat(timespec="seconds"),
                    len(data["templates"]) - 1, label, hand,
                    str(d["source"]), f"{t0:.3f}", f"{t1:.3f}", len(frames)])

    print(f"[EXPORT] label={label!r} hand={hand} frames={len(frames)} "
          f"window={t0:.2f}s~{t1:.2f}s -> {out_path} "
          f"(total {len(data['templates'])} templates)")
    return len(frames)


# =================== Trial playback ===================
def play_trial(samples_path: Path, size=(900, 900), hand="right",
               label=None, out_path=None):
    if out_path is None:
        out_path = default_template_path(samples_path)
    d = _load_trial(samples_path)
    n = len(d["t"])
    duration = float(d["t"][-1])
    label = label or d["where"].name  # referent folder name by default
    title = f"{d['where'].parent.name}/{d['where'].name}/{d['name']}"
    win = "PilotHands"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    speed = 1.0
    paused = False
    clock = 0.0
    last = time.perf_counter()
    trim_start, trim_end = 0.0, duration
    flash, flash_until = "", 0.0

    tl_y = size[1] - 34               # timeline bar geometry
    tl_x0, tl_x1 = 10, size[0] - 10

    def tl_x(tval):
        return tl_x0 + int((tl_x1 - tl_x0) * (tval / duration if duration > 0 else 0))

    while True:
        now = time.perf_counter()
        if not paused:
            clock += (now - last) * speed
        last = now
        if clock > duration + 0.5:  # loop with a short hold on the last frame
            clock = 0.0
        i = min(int(np.searchsorted(d["t"], clock)), n - 1)
        t_i = float(d["t"][i])

        frame = draw_hands_view(
            size,
            d["lj"][i], d["rj"][i], bool(d["lt"][i]), bool(d["rt"][i]),
            gaze_dir=d["gaze"][i] if d["gaze_ok"][i] else None,
        )

        # --- HUD ---
        hud1 = (f"{title}   {t_i:.2f}s / {duration:.2f}s   "
                f"sample {i + 1}/{n}   x{speed:.2f}"
                f"{'   PAUSED' if paused else ''}")
        hud2 = (f"trim {trim_start:.2f}s ~ {trim_end:.2f}s   "
                f"export: hand={hand} label={label!r} -> {out_path.name}   "
                f"[S]tart [E]nd e[X]port  ,/. step")
        cv2.rectangle(frame, (0, 0), (size[0], 52), (15, 15, 15), -1)
        cv2.putText(frame, hud1, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, hud2, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    TRIM_COLOR, 1, cv2.LINE_AA)
        if time.perf_counter() < flash_until:
            cv2.putText(frame, flash, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (120, 255, 255), 1, cv2.LINE_AA)
        if t_i < trim_start - 1e-9 or t_i > trim_end + 1e-9:
            cv2.putText(frame, "outside trim", (size[0] // 2 - 60, tl_y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LOST_COLOR, 1, cv2.LINE_AA)

        # --- timeline: full bar dim, trimmed range bright, playhead dot ---
        cv2.line(frame, (tl_x0, tl_y), (tl_x1, tl_y), GRID_COLOR, 2)
        cv2.line(frame, (tl_x(trim_start), tl_y), (tl_x(trim_end), tl_y), TRIM_COLOR, 2)
        for tv in (trim_start, trim_end):
            x = tl_x(tv)
            cv2.line(frame, (x, tl_y - 7), (x, tl_y + 7), TRIM_COLOR, 2)
        cv2.circle(frame, (tl_x(t_i), tl_y), 5, TEXT_COLOR, -1)

        cv2.imshow(win, frame)
        key = cv2.waitKey(15) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            clock = 0.0
        elif key == ord("["):
            speed = max(0.125, speed / 2)
        elif key == ord("]"):
            speed = min(8.0, speed * 2)
        elif key in (ord(","), ord(".")):
            paused = True
            j = max(0, min(n - 1, i + (1 if key == ord(".") else -1)))
            clock = float(d["t"][j])
        elif key == ord("s"):
            trim_start = t_i
            trim_end = max(trim_end, trim_start)
        elif key == ord("e"):
            trim_end = t_i
            trim_start = min(trim_start, trim_end)
        elif key == ord("x"):
            count = export_template(d, trim_start, trim_end, hand, label, out_path)
            flash = (f"exported {count} frames -> {out_path.name}" if count
                     else "EXPORT FAILED: no tracked frames in trim window")
            flash_until = time.perf_counter() + 2.5
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description="Play back pilot-trial hand data; trim + export Jackknife templates")
    ap.add_argument("target", help="trial prefix (e.g. pilot_data/P01/Ask/trial_01) or the _samples.csv itself")
    ap.add_argument("--hand", choices=["left", "right"], default="right",
                    help="which hand goes into the template (default: right)")
    ap.add_argument("--label", default=None,
                    help="template label (default: the referent folder name)")
    ap.add_argument("--out", type=Path, default=None,
                    help="template JSON to append to (default: pilot_templates.json inside the trial's pilot_data directory)")
    ap.add_argument("--export", action="store_true",
                    help="export without opening the viewer (use with --start/--end)")
    ap.add_argument("--start", type=float, default=None,
                    help="trim start in seconds from trial start (export mode)")
    ap.add_argument("--end", type=float, default=None,
                    help="trim end in seconds from trial start (export mode)")
    args = ap.parse_args()

    target = Path(args.target)
    if target.suffix == ".csv":
        samples_path = target
    else:
        prefix = target.parent / target.stem if target.suffix else target
        samples_path = prefix.parent / f"{prefix.name}_samples.csv"
    if not samples_path.exists():
        sys.exit(f"Not found: {samples_path}")

    out_path = args.out if args.out is not None else default_template_path(samples_path)

    if args.export:
        d = _load_trial(samples_path)
        t0 = args.start if args.start is not None else 0.0
        t1 = args.end if args.end is not None else float(d["t"][-1])
        label = args.label or d["where"].name
        count = export_template(d, t0, t1, args.hand, label, out_path)
        if count == 0:
            sys.exit("Export failed: no tracked frames in the given window.")
    else:
        play_trial(samples_path, hand=args.hand, label=args.label, out_path=out_path)


if __name__ == "__main__":
    main()
