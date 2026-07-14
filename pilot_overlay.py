"""Burn the ridge-mapped gaze point into a pilot-trial screen recording.

Reads the three files pilot_receiver.py wrote for a trial and produces
trial_XX_overlay.mp4 with a red gaze dot (and short trail) on the screen
recording, plus a hand-skeleton side panel (first-person view of the streamed
joints) so gesture and gaze can be reviewed together. Pass --no-hands for the
plain gaze-only overlay.

Usage (paths relative ok):

    python pilot_overlay.py pilot_data/P01/Search/trial_01
    python pilot_overlay.py pilot_data/P01/Search      # all trials in the dir

The positional argument is the trial file prefix (no extension) or a
directory. Frame <-> sample alignment uses receiver_time: each video frame
gets the sample nearest in time (frames CSV vs samples CSV).
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from pilot_hands import _load_trial, draw_hands_view, draw_gaze_marker

DOT_RADIUS = 10
DOT_COLOR = (255, 200, 0)      # BGR red, matches the live preview
TRAIL_COLOR = (0, 255, 255)  # yellow trail of the whole window so far
MAX_TIME_GAP = 0.10          # ignore samples further than this from the frame
HAND_PANEL_W = 520           # matches the live preview side panel


def load_samples(samples_path: Path):
    """Return (times, norm_xy) arrays for rows that have a mapped gaze point."""
    times, points = [], []
    with samples_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nx, ny = row.get("gaze_norm_x", ""), row.get("gaze_norm_y", "")
            if nx == "" or ny == "":
                continue
            times.append(float(row["receiver_time"]))
            points.append((float(nx), float(ny)))
    return np.asarray(times), np.asarray(points)


def load_frame_times(frames_path: Path):
    times = {}
    with frames_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            times[int(row["frame_index"])] = float(row["receiver_time"])
    return times


def _hands_panel_for_time(hands, t, h):
    """Render the hand skeleton panel for the sample nearest receiver_time t
    (empty 'lost' panel when nothing is close enough or t is unknown)."""
    if hands is not None and t is not None:
        j = int(np.argmin(np.abs(hands["recv_t"] - t)))
        if abs(hands["recv_t"][j] - t) <= MAX_TIME_GAP:
            return draw_hands_view(
                (HAND_PANEL_W, h),
                hands["lj"][j], hands["rj"][j],
                bool(hands["lt"][j]), bool(hands["rt"][j]),
                gaze_dir=hands["gaze"][j] if hands["gaze_ok"][j] else None,
            )
    return draw_hands_view((HAND_PANEL_W, h), None, None, False, False)


def overlay_trial(prefix: Path, with_hands: bool = True,
                  latency_frames: float = 0.0, latency_s: float = 0.0) -> bool:
    samples_path = prefix.parent / f"{prefix.name}_samples.csv"
    frames_path = prefix.parent / f"{prefix.name}_frames.csv"
    video_path = prefix.parent / f"{prefix.name}.mp4"
    out_path = prefix.parent / f"{prefix.name}_overlay.mp4"

    for p in (samples_path, frames_path, video_path):
        if not p.exists():
            print(f"[SKIP] {prefix.name}: missing {p.name}")
            return False

    sample_times, sample_points = load_samples(samples_path)
    frame_times = load_frame_times(frames_path)
    if len(sample_times) == 0:
        print(f"[SKIP] {prefix.name}: no mapped gaze samples (was the ridge model loaded?)")
        return False

    hands = _load_trial(samples_path) if with_hands else None

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = w + (HAND_PANEL_W if with_hands else 0)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, h))

    # Recording pipeline lag: the frame stamped at receiver_time T shows the
    # world from ~T - latency, so it gets the CSV sample from that much
    # EARLIER (same convention as compare_bbox_analysis.py).
    latency = float(latency_s) + float(latency_frames) / fps

    frame_idx = 0
    drawn = 0
    trail = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_times.get(frame_idx)
        if t is not None:
            t -= latency
            # gaze trail: everything sampled up to this frame's timestamp
            upto = np.searchsorted(sample_times, t + MAX_TIME_GAP)
            trail = sample_points[:upto]
            if len(trail) >= 2:
                pts = np.round(trail * [w, h]).astype(np.int32)
                cv2.polylines(frame, [pts], False, TRAIL_COLOR, 2)

            nearest = int(np.argmin(np.abs(sample_times - t))) if len(sample_times) else -1
            if nearest >= 0 and abs(sample_times[nearest] - t) <= MAX_TIME_GAP:
                nx, ny = sample_points[nearest]
                draw_gaze_marker(frame, (int(round(nx * w)), int(round(ny * h))),
                                 radius=DOT_RADIUS, color=DOT_COLOR)
                drawn += 1

        if with_hands:
            panel = _hands_panel_for_time(hands, t, h)
            cv2.line(panel, (0, 0), (0, h), (60, 60, 60), 1)
            frame = np.hstack([frame, panel])

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    shown = Path(*out_path.parts[-3:]) if len(out_path.parts) >= 3 else out_path
    print(f"[OK] {shown}: {frame_idx} frames, gaze dot on {drawn}"
          f"{', hands panel on' if with_hands else ''}"
          f", latency={latency * 1000:.0f}ms")
    return True


def main():
    ap = argparse.ArgumentParser(description="Overlay gaze onto pilot trial videos")
    ap.add_argument("target", help="trial prefix (e.g. pilot_data/P01/Search/trial_01) or a directory")
    ap.add_argument("--no-hands", action="store_true",
                    help="skip the hand-skeleton side panel (gaze overlay only)")
    ap.add_argument("--video-latency-frames", type=float, default=8.0,
                    help="Scene-recording pipeline delay in video frames (positive = video lags the "
                    "CSV; converted via each video's FPS, fractional allowed). Default 8.")
    ap.add_argument("--video-latency-s", type=float, default=0.0,
                    help="Extra latency in seconds, added on top of --video-latency-frames.")
    args = ap.parse_args()
    with_hands = not args.no_hands

    target = Path(args.target)
    if target.is_dir():
        # Recursive: works for a referent dir (pilot_data/P01/Search), a whole
        # participant (pilot_data/P01), or the entire pilot_data root.
        prefixes = sorted(
            v.parent / v.stem for v in target.rglob("trial_*.mp4")
            if not v.stem.endswith("_overlay")
        )
        if not prefixes:
            sys.exit(f"No trial_*.mp4 found under {target}")
        for p in prefixes:
            overlay_trial(p, with_hands, args.video_latency_frames, args.video_latency_s)
    else:
        # accept both ".../trial_01" and ".../trial_01.mp4"
        prefix = target.parent / target.stem if target.suffix else target
        if not overlay_trial(prefix, with_hands, args.video_latency_frames, args.video_latency_s):
            sys.exit(1)


if __name__ == "__main__":
    main()
