"""Pilot-study receiver: logs gaze + hand joints to CSV and records the ADB
screen during each 3-second trial window.

Counterpart of Unity's PilotSender (DataCollectionScene). Run from MacProgram/:

    python pilot_receiver.py                 # default: port 5005, preview on
    python pilot_receiver.py --no-preview

Requires:
    - calibration_ridge_model.json next to this script (run calibration.py first)
    - adb connected to the headset + ffmpeg on PATH (same as gesture_vlm.py)

Wire protocol (CSV over UDP, see PilotSender.cs for the authoritative spec):

    PILOT_BEGIN,seq,t,participantId,referent,trialIndex
    PILOT_END,seq,t,participantId,referent,trialIndex
    PILOT_SAMPLE,seq,t,in_trial,gaze_tracked,gx,gy,gz,
                 head_px,head_py,head_pz,head_qx,head_qy,head_qz,head_qw,
                 left_tracked,right_tracked,
                 <25 left joints x,y,z>, <25 right joints x,y,z>

Output layout (one folder per participant/referent, files per trial):

    pilot_data/
      P01/
        Search/
          trial_01_samples.csv    per-sample gaze (raw dir + ridge-mapped
                                  norm x/y) + head pose + 25x2 hand joints
          trial_01_frames.csv     frame_index -> receiver_time for the video
          trial_01.mp4            ADB screen recording of the trial window
        session_log.csv           one row per completed trial

Gaze norm coords are the ridge-mapped normalized ADB-screen coordinates
(0..1), so trial_XX.mp4 + trial_XX_samples.csv together tell you where on the
screen the participant was looking. Use pilot_overlay.py to burn the gaze dot
into a copy of the video for eyeballing.
"""
import argparse
import csv
import json
import socket
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from pilot_hands import draw_hands_view, draw_gaze_marker

SCRIPT_DIR = Path(__file__).resolve().parent

HOST = "0.0.0.0"
DEFAULT_PORT = 5005
OUTPUT_ROOT = SCRIPT_DIR / "pilot_data"
RIDGE_MODEL_PATH = SCRIPT_DIR / "calibration_ridge_model.json"

# Keep the stream geometry identical to vlm_pipeline/config.py so the
# normalized gaze coords land on the same pixels as the main MacProgram.
STREAM_W, STREAM_H = 1100, 1000
ADB_CMD = ["adb", "exec-out", "screenrecord", "--output-format=h264", "-"]
VIDEO_FPS = 30.0  # nominal; true per-frame timing lives in trial_XX_frames.csv

# 25 joints: MediaPipe-style 21 plus the four finger metacarpals ("_meta"),
# which HandPoseRecognizer's extension/curl ratios anchor at. Order matches
# PilotSender.JointsPerHand.
NUM_JOINTS = 25
JOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_meta", "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_meta", "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_meta", "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_meta", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
assert len(JOINT_NAMES) == NUM_JOINTS

SAMPLE_FIELD_COUNT = 17 + NUM_JOINTS * 3 * 2  # 167

SAMPLES_CSV_HEADER = (
    ["receiver_time", "seq", "sender_time", "in_trial", "gaze_tracked",
     "gaze_dir_x", "gaze_dir_y", "gaze_dir_z", "gaze_norm_x", "gaze_norm_y",
     "head_pos_x", "head_pos_y", "head_pos_z",
     "head_rot_x", "head_rot_y", "head_rot_z", "head_rot_w",
     "left_tracked", "right_tracked"]
    + [f"L_{name}_{axis}" for name in JOINT_NAMES for axis in ("x", "y", "z")]
    + [f"R_{name}_{axis}" for name in JOINT_NAMES for axis in ("x", "y", "z")]
)

SESSION_LOG_HEADER = [
    "participant", "referent", "trial_index", "begin_receiver_time",
    "end_receiver_time", "num_samples", "num_frames", "samples_csv", "video",
]


# =================== Ridge model (gaze dir -> normalized screen xy) ===================
_ridge_weights = None


def load_ridge_model() -> bool:
    global _ridge_weights
    try:
        with RIDGE_MODEL_PATH.open("r", encoding="utf-8") as f:
            _ridge_weights = np.array(json.load(f)["weights"], dtype=np.float64)
        print(f"[MODEL] Loaded ridge model: {RIDGE_MODEL_PATH}")
        return True
    except Exception as exc:
        _ridge_weights = None
        print(f"[MODEL][ERROR] Failed to load {RIDGE_MODEL_PATH}: {exc}")
        return False


def map_gaze_dir_to_norm(gx: float, gy: float, gz: float):
    """Same polynomial expansion as calibration.py / vlm_pipeline/ridge.py."""
    if _ridge_weights is None:
        return None
    feat = np.array(
        [1.0, gx, gy, gz, gx * gx, gy * gy, gz * gz, gx * gy, gx * gz, gy * gz],
        dtype=np.float64,
    )
    pred = feat @ _ridge_weights
    return float(pred[0]), float(pred[1])


# =================== Shared state ===================
stop_event = threading.Event()

frame_lock = threading.Lock()
latest_frame = None

gaze_lock = threading.Lock()
latest_gaze_norm = None      # ridge-mapped (norm_x, norm_y) or None
latest_gaze_tracked = False
latest_gaze_dir = None       # camera-local gaze direction (3,) or None
latest_left_joints = None    # camera-local (25, 3) or None
latest_right_joints = None
latest_left_tracked = False
latest_right_tracked = False
sample_recv_count = 0

HAND_PANEL_W = 520           # width of the live hand-skeleton side panel

trial_lock = threading.Lock()
trial = None  # active-trial dict, see _begin_trial()


# =================== Trial lifecycle ===================
def _safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s.strip()) or "Unknown"


def _begin_trial(pkt: dict):
    global trial
    with trial_lock:
        if trial is not None:
            print(f"\n[TRIAL][WARN] BEGIN while trial active; force-closing previous one.")
            _end_trial_locked()

        participant = f"P{pkt['participant']:02d}"
        referent = _safe_name(pkt["referent"])
        idx = pkt["trial_index"]
        out_dir = OUTPUT_ROOT / participant / referent
        out_dir.mkdir(parents=True, exist_ok=True)

        samples_path = out_dir / f"trial_{idx:02d}_samples.csv"
        frames_path = out_dir / f"trial_{idx:02d}_frames.csv"
        video_path = out_dir / f"trial_{idx:02d}.mp4"
        if samples_path.exists():
            print(f"\n[TRIAL][WARN] {samples_path.name} already exists -> overwriting (redo?).")

        samples_f = samples_path.open("w", newline="", encoding="utf-8")
        samples_w = csv.writer(samples_f)
        samples_w.writerow(SAMPLES_CSV_HEADER)

        frames_f = frames_path.open("w", newline="", encoding="utf-8")
        frames_w = csv.writer(frames_f)
        frames_w.writerow(["frame_index", "receiver_time"])

        trial = {
            "participant": pkt["participant"],
            "referent": referent,
            "trial_index": idx,
            "begin_time": time.time(),
            "out_dir": out_dir,
            "samples_f": samples_f, "samples_w": samples_w,
            "samples_path": samples_path, "num_samples": 0,
            "frames_f": frames_f, "frames_w": frames_w,
            "video_path": video_path, "video_writer": None, "num_frames": 0,
        }
    print(f"\n[TRIAL] BEGIN {participant} {referent} trial={idx}")


def _end_trial_locked():
    """Close the active trial's files. Caller must hold trial_lock."""
    global trial
    if trial is None:
        return
    t = trial
    trial = None  # stop sample/frame appends first

    end_time = time.time()
    t["samples_f"].close()
    t["frames_f"].close()
    if t["video_writer"] is not None:
        t["video_writer"].release()

    log_path = t["out_dir"].parent / "session_log.csv"
    new_log = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(SESSION_LOG_HEADER)
        w.writerow([
            t["participant"], t["referent"], t["trial_index"],
            f"{t['begin_time']:.4f}", f"{end_time:.4f}",
            t["num_samples"], t["num_frames"],
            t["samples_path"].name, t["video_path"].name,
        ])

    print(
        f"\n[TRIAL] END   P{t['participant']:02d} {t['referent']} trial={t['trial_index']} "
        f"samples={t['num_samples']} frames={t['num_frames']} "
        f"({end_time - t['begin_time']:.2f}s) -> {t['out_dir']}"
    )


def _end_trial(pkt: dict):
    with trial_lock:
        if trial is None:
            print("\n[TRIAL][WARN] END without active trial; ignoring.")
            return
        _end_trial_locked()


# =================== Packet parsing / handling ===================
def _handle_sample(parts):
    global latest_gaze_norm, latest_gaze_tracked, latest_gaze_dir, sample_recv_count
    global latest_left_joints, latest_right_joints, latest_left_tracked, latest_right_tracked
    if len(parts) != SAMPLE_FIELD_COUNT:
        raise ValueError(f"PILOT_SAMPLE expects {SAMPLE_FIELD_COUNT} fields, got {len(parts)}")

    recv_time = time.time()
    gaze_tracked = parts[4] == "1"
    gx, gy, gz = float(parts[5]), float(parts[6]), float(parts[7])
    norm = map_gaze_dir_to_norm(gx, gy, gz) if gaze_tracked else None

    left_tracked = parts[15] == "1"
    right_tracked = parts[16] == "1"
    left_joints = (
        np.array(parts[17:17 + NUM_JOINTS * 3], dtype=np.float64).reshape(NUM_JOINTS, 3)
        if left_tracked else None
    )
    right_joints = (
        np.array(parts[17 + NUM_JOINTS * 3:], dtype=np.float64).reshape(NUM_JOINTS, 3)
        if right_tracked else None
    )

    with gaze_lock:
        latest_gaze_tracked = gaze_tracked
        latest_gaze_norm = norm
        latest_gaze_dir = (gx, gy, gz) if gaze_tracked else None
        latest_left_tracked = left_tracked
        latest_right_tracked = right_tracked
        latest_left_joints = left_joints
        latest_right_joints = right_joints
        sample_recv_count += 1

    with trial_lock:
        if trial is None:
            return
        norm_x, norm_y = norm if norm is not None else ("", "")
        row = (
            [f"{recv_time:.4f}", parts[1], parts[2], parts[3], parts[4],
             parts[5], parts[6], parts[7], norm_x, norm_y]
            + parts[8:17]          # head pose + tracked flags
            + parts[17:]           # 2 x 25 x 3 hand joint coords
        )
        trial["samples_w"].writerow(row)
        trial["num_samples"] += 1


def _handle_control(parts, event_type):
    if len(parts) != 6:
        raise ValueError(f"{event_type} expects 6 fields, got {len(parts)}")
    pkt = {
        "seq": int(parts[1]),
        "sender_time": float(parts[2]),
        "participant": int(parts[3]),
        "referent": parts[4],
        "trial_index": int(parts[5]),
    }
    if event_type == "PILOT_BEGIN":
        _begin_trial(pkt)
    else:
        _end_trial(pkt)


def udp_receiver_loop(sock: socket.socket):
    sock.settimeout(0.05)
    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            break
        try:
            parts = data.decode("utf-8").strip().split(",")
            ptype = parts[0]
            if ptype == "PILOT_SAMPLE":
                _handle_sample(parts)
            elif ptype in ("PILOT_BEGIN", "PILOT_END"):
                _handle_control(parts, ptype)
            elif ptype == "GAZE":
                pass  # stray MsgSender stream; ignore
            else:
                print(f"\n[UDP][WARN] unknown packet type: {ptype!r}")
        except Exception as exc:
            print(f"\n[UDP][WARN] parse failed: {exc}")


# =================== ADB stream reader (same pattern as vlm_pipeline/network.py) ===================
def _build_ffmpeg_cmd(width: int, height: int):
    return [
        "ffmpeg", "-loglevel", "error", "-i", "-",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-vf", f"scale={width}:{height}", "-",
    ]


def stream_reader_loop():
    global latest_frame
    frame_size = STREAM_W * STREAM_H * 3

    while not stop_event.is_set():
        adb_proc = None
        ffmpeg_proc = None
        try:
            adb_proc = subprocess.Popen(
                ADB_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7,
            )
            ffmpeg_proc = subprocess.Popen(
                _build_ffmpeg_cmd(STREAM_W, STREAM_H),
                stdin=adb_proc.stdout, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=10**7,
            )
            print(f"[STREAM] adb|ffmpeg started ({STREAM_W}x{STREAM_H})")

            while not stop_event.is_set():
                raw = ffmpeg_proc.stdout.read(frame_size)
                if not raw or len(raw) != frame_size:
                    print("[STREAM] frame read failed -> restart")
                    break
                recv_time = time.time()
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((STREAM_H, STREAM_W, 3))

                with frame_lock:
                    latest_frame = arr

                with trial_lock:
                    if trial is not None:
                        if trial["video_writer"] is None:
                            trial["video_writer"] = cv2.VideoWriter(
                                str(trial["video_path"]),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                VIDEO_FPS, (STREAM_W, STREAM_H),
                            )
                        trial["video_writer"].write(arr)
                        trial["frames_w"].writerow([trial["num_frames"], f"{recv_time:.4f}"])
                        trial["num_frames"] += 1
        except Exception as exc:
            print(f"[STREAM][ERROR] {exc}")
        finally:
            for p in (ffmpeg_proc, adb_proc):
                try:
                    if p and p.stdout is not None:
                        p.stdout.close()
                except Exception:
                    pass
                try:
                    if p:
                        p.kill()
                except Exception:
                    pass
            if not stop_event.is_set():
                time.sleep(0.5)


# =================== Live preview ===================
def preview_loop():
    """Main-thread cv2 window: live ADB frame + gaze dot, hand skeleton panel, status."""
    win = "PilotReceiver"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    last_rate_t = time.perf_counter()
    last_count = 0
    sample_hz = 0.0

    while not stop_event.is_set():
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            frame = np.full((STREAM_H, STREAM_W, 3), 30, dtype=np.uint8)
            cv2.putText(frame, "waiting for adb stream...", (30, STREAM_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        with gaze_lock:
            norm = latest_gaze_norm
            tracked = latest_gaze_tracked
            count = sample_recv_count
            gaze_dir = latest_gaze_dir
            lj, rj = latest_left_joints, latest_right_joints
            lt, rt = latest_left_tracked, latest_right_tracked

        now = time.perf_counter()
        if now - last_rate_t >= 1.0:
            sample_hz = (count - last_count) / (now - last_rate_t)
            last_count = count
            last_rate_t = now

        if norm is not None:
            px = int(round(norm[0] * STREAM_W))
            py = int(round(norm[1] * STREAM_H))
            if -50 <= px <= STREAM_W + 50 and -50 <= py <= STREAM_H + 50:
                draw_gaze_marker(frame, (px, py))

        with trial_lock:
            if trial is not None:
                status = (f"REC  P{trial['participant']:02d} {trial['referent']} "
                          f"trial {trial['trial_index']}  "
                          f"samples={trial['num_samples']} frames={trial['num_frames']}")
                color = (0, 0, 255)
            else:
                status = "idle"
                color = (0, 255, 0)

        cv2.rectangle(frame, (0, 0), (STREAM_W, 34), (20, 20, 20), -1)
        cv2.putText(frame, f"{status}   |   samples {sample_hz:.0f} Hz   gaze "
                    f"{'ok' if tracked else 'LOST'}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Hand skeleton side panel (first-person view of the streamed joints)
        panel = draw_hands_view((HAND_PANEL_W, STREAM_H), lj, rj, lt, rt,
                                gaze_dir=gaze_dir)
        cv2.rectangle(panel, (0, 0), (HAND_PANEL_W, 34), (20, 20, 20), -1)
        cv2.putText(panel, "hands (head-cam view)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.line(panel, (0, 0), (0, STREAM_H), (60, 60, 60), 1)

        cv2.imshow(win, np.hstack([frame, panel]))
        if cv2.waitKey(15) & 0xFF == ord("q"):
            stop_event.set()
            break
    cv2.destroyAllWindows()


def headless_loop():
    while not stop_event.is_set():
        with gaze_lock:
            count = sample_recv_count
        with trial_lock:
            rec = trial is not None
        print(f"\r[STATUS] samples_total={count} recording={rec}     ", end="", flush=True)
        time.sleep(1.0)


# =================== Main ===================
def main():
    ap = argparse.ArgumentParser(description="Pilot study receiver (gaze+hands CSV, ADB video)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-preview", action="store_true", help="run without the cv2 preview window")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="where trial folders are written (default: pilot_data/ next to this script). Use a throwaway dir for pipeline tests so real participant data is never mixed with test output.")
    args = ap.parse_args()

    global OUTPUT_ROOT
    if args.data_root is not None:
        OUTPUT_ROOT = args.data_root

    if not load_ridge_model():
        print("[MODEL][WARN] gaze_norm_x/y will be empty. Run calibration.py first for mapping.")

    OUTPUT_ROOT.mkdir(exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, args.port))
    print(f"Listening for pilot UDP packets on {HOST}:{args.port}")
    print(f"Saving trials under: {OUTPUT_ROOT}")

    threading.Thread(target=stream_reader_loop, daemon=True).start()
    threading.Thread(target=udp_receiver_loop, args=(sock,), daemon=True).start()

    try:
        if args.no_preview:
            headless_loop()
        else:
            preview_loop()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        with trial_lock:
            if trial is not None:
                print("\n[TRIAL][WARN] receiver exiting mid-trial; closing files.")
                _end_trial_locked()
        sock.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
