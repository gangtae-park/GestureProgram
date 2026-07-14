import csv
import json
import socket
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

def inline_print(msg: str):
    print(f"\r{msg:<160}", end="", flush=True)

HOST = "0.0.0.0"
PORT = 5005
OUTPUT_CSV_PATH = Path("calibration_gaze_samples.csv")
MODEL_OUTPUT_PATH = Path("calibration_ridge_model.json")
SOCKET_BUFFER_SIZE = 4096
PRINT_LIVE_LOG = True
AUTO_FLUSH_ON_COMPLETE = True
RIDGE_ALPHA = 1e-3
STOP_AFTER_DOT_INDEX = 8

# ---- Saccadic evaluation (after the 9-dot fit) ----
# After the last calibration dot the script does NOT exit: it fits the ridge
# model, then keeps listening. Unity's SaccadicTaskController runs a 30-
# fixation random saccadic task; we map every incoming gaze sample through
# the fresh model and record the ADB screen so the fixation-vs-mapped-gaze
# pixel error can be measured afterwards. Same adb|ffmpeg geometry as the
# other MacProgram tools so norm coords -> pixels is just norm * (W, H).
STREAM_W, STREAM_H = 1100, 1000
ADB_CMD = ["adb", "exec-out", "screenrecord", "--output-format=h264", "-"]
VIDEO_FPS = 30.0  # nominal; true per-frame timing lives in eval frames.csv

# The mapped gaze point is burned into screen.mp4 while the task records, so
# fixation-vs-gaze offset is directly visible in the video. Raw coords stay
# in gaze.csv for numeric analysis. Marker is a hollow-circle reticle with a
# center gap so the exact gaze pixel stays visible.
GAZE_DOT_RADIUS = 12
GAZE_DOT_COLOR = (255, 200, 0)  # BGR red


def draw_gaze_marker(img, center, radius=GAZE_DOT_RADIUS, color=GAZE_DOT_COLOR,
                     thickness=1, center_gap=4, tick_overhang=5):
    """Hollow circle + 4 crosshair ticks that stop `center_gap` px short of
    the center, leaving the exact gaze pixel unobscured."""
    x, y = center
    cv2.circle(img, (x, y), radius, color, thickness, cv2.LINE_AA)
    outer = radius + tick_overhang
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img,
                 (x + dx * center_gap, y + dy * center_gap),
                 (x + dx * outer, y + dy * outer),
                 color, thickness, cv2.LINE_AA)

EVAL_GAZE_HEADER = [
    "receiver_time", "seq", "sender_time", "is_tracked",
    "gaze_dir_x", "gaze_dir_y", "gaze_dir_z", "norm_x", "norm_y",
    "fix_index", "fix_u", "fix_v", "fix_expected_norm_x", "fix_expected_norm_y",
]
EVAL_FIX_HEADER = [
    "receiver_time", "seq", "sender_time", "fix_index", "u", "v",
    "expected_norm_x", "expected_norm_y",
]

TARGET_NORM_BY_DOT = {
    0: (0.359, 0.368),
    1: (0.522, 0.367),
    2: (0.689, 0.368),
    3: (0.356, 0.534),
    4: (0.523, 0.534),
    5: (0.688, 0.534),
    6: (0.356, 0.700),
    7: (0.524, 0.700),
    8: (0.688, 0.700),
}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
sock.setblocking(False)

recv_count = 0
fps_timer = time.perf_counter()
latest_packet = None

samples_by_dot = defaultdict(list)
all_samples = []
current_hold_dot_index = None
current_hold_samples = []
should_stop = False

# ---- evaluation-phase state ----
eval_mode = False            # True once the 9-dot model is fitted
eval_weights = None          # forward ridge weights (gaze dir -> norm xy)
eval_lock = threading.Lock() # guards everything below (UDP thread-free, but the adb stream thread reads/writes too)
task_active = False
current_fix = None           # {"index", "u", "v", "ex", "ey"}
eval_dir = None
eval_gaze_f = eval_gaze_w = None
eval_fix_f = eval_fix_w = None
eval_frames_f = eval_frames_w = None
eval_video_writer = None
eval_num_frames = 0
eval_num_gaze_rows = 0
latest_eval_norm = None
stream_stop = threading.Event()
def reset_output_files():
    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()

    with MODEL_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump({}, f)

CSV_HEADER = [
    "receiver_time",
    "seq",
    "sender_time",
    "is_tracked",
    "calibration_dot_index",
    "gaze_dir_x",
    "gaze_dir_y",
    "gaze_dir_z",
]
reset_output_files()

def append_rows_to_csv(rows):
    if not rows:
        return

    with OUTPUT_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerows(rows)


def build_feature_vector(gaze_dir_x, gaze_dir_y, gaze_dir_z):
    return np.array([
        1.0,
        gaze_dir_x,
        gaze_dir_y,
        gaze_dir_z,
        gaze_dir_x * gaze_dir_x,
        gaze_dir_y * gaze_dir_y,
        gaze_dir_z * gaze_dir_z,
        gaze_dir_x * gaze_dir_y,
        gaze_dir_x * gaze_dir_z,
        gaze_dir_y * gaze_dir_z,
    ], dtype=np.float64)


def build_inverse_feature_vector(norm_x, norm_y):
    """Quadratic polynomial expansion of the ADB-normalised 2D point.
    Used to learn (norm_x, norm_y) -> (gaze_dir_x, gaze_dir_y); z is reconstructed
    at inference as sqrt(1 - x^2 - y^2) under the assumption that the user faces
    the screen (z > 0 over the calibration grid)."""
    return np.array([
        1.0,
        norm_x,
        norm_y,
        norm_x * norm_x,
        norm_y * norm_y,
        norm_x * norm_y,
    ], dtype=np.float64)


def compute_mean_gaze_by_dot():
    mean_gaze_by_dot = {}
    for dot_index in sorted(TARGET_NORM_BY_DOT.keys()):
        rows = samples_by_dot.get(dot_index, [])
        if not rows:
            continue

        mean_gaze_by_dot[dot_index] = {
            "gaze_dir_x": float(np.mean([row["gaze_dir_x"] for row in rows])),
            "gaze_dir_y": float(np.mean([row["gaze_dir_y"] for row in rows])),
            "gaze_dir_z": float(np.mean([row["gaze_dir_z"] for row in rows])),
            "num_samples": len(rows),
        }
    return mean_gaze_by_dot


def fit_ridge_regression_model(alpha=RIDGE_ALPHA):
    mean_gaze_by_dot = compute_mean_gaze_by_dot()
    available_indices = [
        dot_index
        for dot_index in sorted(TARGET_NORM_BY_DOT.keys())
        if dot_index in mean_gaze_by_dot
    ]

    if len(available_indices) < 3:
        print("[MODEL] Not enough completed dots to fit ridge regression model.")
        return None

    X = []
    y = []
    for dot_index in available_indices:
        mean_row = mean_gaze_by_dot[dot_index]
        X.append(
            build_feature_vector(
                mean_row["gaze_dir_x"],
                mean_row["gaze_dir_y"],
                mean_row["gaze_dir_z"],
            )
        )
        y.append(TARGET_NORM_BY_DOT[dot_index])

    X = np.vstack(X)
    y = np.asarray(y, dtype=np.float64)

    reg = alpha * np.eye(X.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(X.T @ X + reg, X.T @ y)
    predictions = X @ weights
    mse = float(np.mean((predictions - y) ** 2))

    # ---- Inverse model: (norm_x, norm_y) -> (gaze_dir_x, gaze_dir_y) ----
    X_inv = []
    y_inv = []
    for dot_index in available_indices:
        target_norm_x, target_norm_y = TARGET_NORM_BY_DOT[dot_index]
        mean_row = mean_gaze_by_dot[dot_index]
        X_inv.append(build_inverse_feature_vector(target_norm_x, target_norm_y))
        y_inv.append([mean_row["gaze_dir_x"], mean_row["gaze_dir_y"]])
    X_inv = np.vstack(X_inv)
    y_inv = np.asarray(y_inv, dtype=np.float64)

    reg_inv = alpha * np.eye(X_inv.shape[1], dtype=np.float64)
    reg_inv[0, 0] = 0.0
    inverse_weights = np.linalg.solve(X_inv.T @ X_inv + reg_inv, X_inv.T @ y_inv)
    inverse_predictions = X_inv @ inverse_weights
    inverse_mse = float(np.mean((inverse_predictions - y_inv) ** 2))

    model_payload = {
        "ridge_alpha": alpha,
        "feature_order": [
            "bias",
            "gaze_dir_x",
            "gaze_dir_y",
            "gaze_dir_z",
            "gaze_dir_x_sq",
            "gaze_dir_y_sq",
            "gaze_dir_z_sq",
            "gaze_dir_x_mul_gaze_dir_y",
            "gaze_dir_x_mul_gaze_dir_z",
            "gaze_dir_y_mul_gaze_dir_z",
        ],
        "weights": weights.tolist(),
        "training_mse": mse,
        "inverse_feature_order": [
            "bias",
            "norm_x",
            "norm_y",
            "norm_x_sq",
            "norm_y_sq",
            "norm_x_mul_norm_y",
        ],
        "inverse_output_order": [
            "gaze_dir_x",
            "gaze_dir_y",
        ],
        "inverse_weights": inverse_weights.tolist(),
        "inverse_training_mse": inverse_mse,
        "inverse_predictions_by_dot": {
            str(dot_index): {
                "pred_gaze_dir_x": float(inverse_predictions[row_idx, 0]),
                "pred_gaze_dir_y": float(inverse_predictions[row_idx, 1]),
                "target_gaze_dir_x": mean_gaze_by_dot[dot_index]["gaze_dir_x"],
                "target_gaze_dir_y": mean_gaze_by_dot[dot_index]["gaze_dir_y"],
            }
            for row_idx, dot_index in enumerate(available_indices)
        },
        "mean_gaze_by_dot": mean_gaze_by_dot,
        "targets_by_dot": {
            str(dot_index): {
                "norm_x": TARGET_NORM_BY_DOT[dot_index][0],
                "norm_y": TARGET_NORM_BY_DOT[dot_index][1],
            }
            for dot_index in available_indices
        },
        "predictions_by_dot": {
            str(dot_index): {
                "pred_norm_x": float(predictions[row_idx, 0]),
                "pred_norm_y": float(predictions[row_idx, 1]),
                "target_norm_x": TARGET_NORM_BY_DOT[dot_index][0],
                "target_norm_y": TARGET_NORM_BY_DOT[dot_index][1],
            }
            for row_idx, dot_index in enumerate(available_indices)
        },
    }

    with MODEL_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

    print(f"[MODEL] Saved ridge regression model to: {MODEL_OUTPUT_PATH.resolve()}")
    print(f"[MODEL] forward training_mse={mse:.8f}")
    for row_idx, dot_index in enumerate(available_indices):
        pred_x = predictions[row_idx, 0]
        pred_y = predictions[row_idx, 1]
        target_x, target_y = TARGET_NORM_BY_DOT[dot_index]
        print(
            f"[MODEL] dot={dot_index} "
            f"pred=({pred_x:.4f}, {pred_y:.4f}) "
            f"target=({target_x:.4f}, {target_y:.4f})"
        )
    print(f"[MODEL] inverse training_mse={inverse_mse:.8f}")
    for row_idx, dot_index in enumerate(available_indices):
        pred_gx = inverse_predictions[row_idx, 0]
        pred_gy = inverse_predictions[row_idx, 1]
        target_gx = mean_gaze_by_dot[dot_index]["gaze_dir_x"]
        target_gy = mean_gaze_by_dot[dot_index]["gaze_dir_y"]
        print(
            f"[MODEL][INV] dot={dot_index} "
            f"pred_gaze=({pred_gx:.4f}, {pred_gy:.4f}) "
            f"target_gaze=({target_gx:.4f}, {target_gy:.4f})"
        )

    return model_payload

# =================== Saccadic evaluation ===================
def _grid_edge_norms():
    """Norm-coord bounds of the calibration grid, from the 9 dot targets.
    u=0 maps to the left-column mean x, u=1 to the right column; v=0 to the
    top-row mean y, v=1 to the bottom row (same u/v convention Unity sends)."""
    left = np.mean([TARGET_NORM_BY_DOT[i][0] for i in (0, 3, 6)])
    right = np.mean([TARGET_NORM_BY_DOT[i][0] for i in (2, 5, 8)])
    top = np.mean([TARGET_NORM_BY_DOT[i][1] for i in (0, 1, 2)])
    bottom = np.mean([TARGET_NORM_BY_DOT[i][1] for i in (6, 7, 8)])
    return left, right, top, bottom


def expected_norm_from_uv(u: float, v: float):
    left, right, top, bottom = _grid_edge_norms()
    return left + u * (right - left), top + v * (bottom - top)


def map_gaze_dir(gx: float, gy: float, gz: float):
    if eval_weights is None:
        return None
    pred = build_feature_vector(gx, gy, gz) @ eval_weights
    return float(pred[0]), float(pred[1])


def enter_evaluation_mode():
    """Fit the ridge model from the completed 9 dots, then keep running so the
    saccadic task can be recorded with the freshly mapped gaze."""
    global eval_mode, eval_weights, should_stop

    print_summary()
    model = fit_ridge_regression_model()
    if model is None:
        print("[EVAL] Model fit failed -- cannot run evaluation. Stopping.")
        should_stop = True
        return

    eval_weights = np.array(model["weights"], dtype=np.float64)
    eval_mode = True
    threading.Thread(target=stream_reader_loop, daemon=True).start()
    print("\n[EVAL] Ridge model fitted. Waiting for the saccadic task "
          "(pinch-hold 'Test Start' in the headset)...")


def begin_saccade_task(packet):
    global task_active, eval_dir, eval_num_frames, eval_num_gaze_rows
    global eval_gaze_f, eval_gaze_w, eval_fix_f, eval_fix_w, eval_frames_f, eval_frames_w

    with eval_lock:
        if task_active:
            print("\n[EVAL][WARN] SACCADE_BEGIN while task already active; ignoring.")
            return
        eval_dir = Path("calibration_eval") / f"{datetime.now():%Y%m%d_%H%M%S}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        eval_gaze_f = (eval_dir / "gaze.csv").open("w", newline="", encoding="utf-8")
        eval_gaze_w = csv.writer(eval_gaze_f)
        eval_gaze_w.writerow(EVAL_GAZE_HEADER)

        eval_fix_f = (eval_dir / "fixations.csv").open("w", newline="", encoding="utf-8")
        eval_fix_w = csv.writer(eval_fix_f)
        eval_fix_w.writerow(EVAL_FIX_HEADER)

        eval_frames_f = (eval_dir / "frames.csv").open("w", newline="", encoding="utf-8")
        eval_frames_w = csv.writer(eval_frames_f)
        eval_frames_w.writerow(["frame_index", "receiver_time"])

        eval_num_frames = 0
        eval_num_gaze_rows = 0
        task_active = True

    print(f"\n[EVAL] SACCADE_BEGIN fixations={packet['num_fixations']} -> {eval_dir}/")


def handle_saccade_fix(packet):
    global current_fix
    ex, ey = expected_norm_from_uv(packet["u"], packet["v"])
    with eval_lock:
        if not task_active:
            return
        current_fix = {
            "index": packet["fix_index"],
            "u": packet["u"], "v": packet["v"],
            "ex": ex, "ey": ey,
        }
        eval_fix_w.writerow([
            f"{time.time():.4f}", packet["seq"], packet["sender_time"],
            packet["fix_index"],
            f"{packet['u']:.4f}", f"{packet['v']:.4f}", f"{ex:.4f}", f"{ey:.4f}",
        ])
    inline_print(
        f"[EVAL] fixation {packet['fix_index']} u={packet['u']:.2f} v={packet['v']:.2f} "
        f"expected_norm=({ex:.3f}, {ey:.3f})"
    )


def end_saccade_task(packet):
    global should_stop
    with eval_lock:
        if not task_active:
            print("\n[EVAL][WARN] SACCADE_END without active task; ignoring.")
            return
        _close_eval_files_locked()
    print(f"\n[EVAL] SACCADE_END shown={packet['num_fixations']} "
          f"gaze_rows={eval_num_gaze_rows} frames={eval_num_frames}")
    print(f"[EVAL] Saved evaluation data to: {eval_dir.resolve()}")
    should_stop = True
    stream_stop.set()


def _close_eval_files_locked():
    """Close all evaluation outputs. Caller must hold eval_lock."""
    global task_active, current_fix, eval_video_writer
    global eval_gaze_f, eval_gaze_w, eval_fix_f, eval_fix_w, eval_frames_f, eval_frames_w

    task_active = False
    current_fix = None
    for f in (eval_gaze_f, eval_fix_f, eval_frames_f):
        if f is not None:
            f.close()
    eval_gaze_f = eval_gaze_w = None
    eval_fix_f = eval_fix_w = None
    eval_frames_f = eval_frames_w = None
    if eval_video_writer is not None:
        eval_video_writer.release()
        eval_video_writer = None


def handle_eval_gaze(packet):
    """Map an idle GAZE (or SAMPLE) packet through the fitted model; log it
    while the saccadic task is running."""
    global latest_eval_norm, eval_num_gaze_rows

    tracked = packet["is_tracked"] == 1
    norm = map_gaze_dir(packet["gaze_dir_x"], packet["gaze_dir_y"], packet["gaze_dir_z"]) if tracked else None
    latest_eval_norm = norm

    with eval_lock:
        if not task_active or eval_gaze_w is None:
            return
        fix = current_fix
        norm_x, norm_y = (f"{norm[0]:.5f}", f"{norm[1]:.5f}") if norm is not None else ("", "")
        eval_gaze_w.writerow([
            f"{time.time():.4f}", packet["seq"], packet["sender_time"],
            packet["is_tracked"],
            packet["gaze_dir_x"], packet["gaze_dir_y"], packet["gaze_dir_z"],
            norm_x, norm_y,
            fix["index"] if fix else -1,
            f"{fix['u']:.4f}" if fix else "",
            f"{fix['v']:.4f}" if fix else "",
            f"{fix['ex']:.4f}" if fix else "",
            f"{fix['ey']:.4f}" if fix else "",
        ])
        eval_num_gaze_rows += 1


def _build_ffmpeg_cmd(width: int, height: int):
    return [
        "ffmpeg", "-loglevel", "error", "-i", "-",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-vf", f"scale={width}:{height}", "-",
    ]


def stream_reader_loop():
    """ADB screen stream (same pattern as pilot_receiver.py). Started when the
    evaluation phase begins so the pipeline is warm before the task; frames
    are written to disk only while the saccadic task is active."""
    global eval_video_writer, eval_num_frames
    frame_size = STREAM_W * STREAM_H * 3

    while not stream_stop.is_set():
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
            print(f"\n[EVAL][STREAM] adb|ffmpeg started ({STREAM_W}x{STREAM_H})")

            while not stream_stop.is_set():
                raw = ffmpeg_proc.stdout.read(frame_size)
                if not raw or len(raw) != frame_size:
                    print("\n[EVAL][STREAM] frame read failed -> restart")
                    break
                recv_time = time.time()
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((STREAM_H, STREAM_W, 3))

                with eval_lock:
                    if task_active and eval_frames_w is not None:
                        if eval_video_writer is None:
                            eval_video_writer = cv2.VideoWriter(
                                str(eval_dir / "screen.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                VIDEO_FPS, (STREAM_W, STREAM_H),
                            )
                        frame = arr.copy()  # frombuffer view is read-only
                        norm = latest_eval_norm
                        if norm is not None:
                            px = int(round(norm[0] * STREAM_W))
                            py = int(round(norm[1] * STREAM_H))
                            if -50 <= px <= STREAM_W + 50 and -50 <= py <= STREAM_H + 50:
                                draw_gaze_marker(frame, (px, py))
                        eval_video_writer.write(frame)
                        eval_frames_w.writerow([eval_num_frames, f"{recv_time:.4f}"])
                        eval_num_frames += 1
        except Exception as exc:
            print(f"\n[EVAL][STREAM][ERROR] {exc}")
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
            if not stream_stop.is_set():
                time.sleep(0.5)


def parse_message(msg: str):
    parts = msg.strip().split(",")
    if not parts:
        raise ValueError("Empty packet")

    event_type = parts[0]

    if event_type in ("BEGIN", "CANCEL", "COMPLETE"):
        if len(parts) != 4:
            raise ValueError(f"{event_type} packet must have 4 values, got {len(parts)}")
        return {
            "event_type": event_type,
            "seq": int(parts[1]),
            "sender_time": float(parts[2]),
            "calibration_dot_index": int(parts[3]),
        }

    if event_type in ("SAMPLE", "GAZE"):
        # Same 8-field shape; GAZE is CalibSender's idle stream (dotIndex -1),
        # which is what the evaluation phase maps through the fitted model.
        if len(parts) != 8:
            raise ValueError(f"{event_type} packet must have 8 values, got {len(parts)}")
        return {
            "event_type": event_type,
            "seq": int(parts[1]),
            "sender_time": float(parts[2]),
            "is_tracked": int(parts[3]),
            "calibration_dot_index": int(parts[4]),
            "gaze_dir_x": float(parts[5]),
            "gaze_dir_y": float(parts[6]),
            "gaze_dir_z": float(parts[7]),
        }

    if event_type == "SACCADE_BEGIN":
        if len(parts) != 4:
            raise ValueError(f"SACCADE_BEGIN packet must have 4 values, got {len(parts)}")
        return {
            "event_type": event_type,
            "seq": int(parts[1]),
            "sender_time": float(parts[2]),
            "num_fixations": int(parts[3]),
        }

    if event_type == "SACCADE_FIX":
        if len(parts) != 6:
            raise ValueError(f"SACCADE_FIX packet must have 6 values, got {len(parts)}")
        return {
            "event_type": event_type,
            "seq": int(parts[1]),
            "sender_time": float(parts[2]),
            "fix_index": int(parts[3]),
            "u": float(parts[4]),
            "v": float(parts[5]),
        }

    if event_type == "SACCADE_END":
        if len(parts) != 4:
            raise ValueError(f"SACCADE_END packet must have 4 values, got {len(parts)}")
        return {
            "event_type": event_type,
            "seq": int(parts[1]),
            "sender_time": float(parts[2]),
            "num_fixations": int(parts[3]),
        }

    raise ValueError(f"Unknown event type: {event_type}")



def begin_hold(packet):
    global current_hold_dot_index, current_hold_samples

    current_hold_dot_index = packet["calibration_dot_index"]
    current_hold_samples = []

    print(f"\n[BEGIN] dot={current_hold_dot_index} seq={packet['seq']}")



def add_sample(packet):
    global current_hold_dot_index, current_hold_samples

    dot_index = packet["calibration_dot_index"]
    if current_hold_dot_index is None:
        return
    if dot_index != current_hold_dot_index:
        return
    if packet["is_tracked"] != 1:
        return

    row = {
        "receiver_time": time.time(),
        "seq": packet["seq"],
        "sender_time": packet["sender_time"],
        "is_tracked": packet["is_tracked"],
        "calibration_dot_index": dot_index,
        "gaze_dir_x": packet["gaze_dir_x"],
        "gaze_dir_y": packet["gaze_dir_y"],
        "gaze_dir_z": packet["gaze_dir_z"],
    }

    current_hold_samples.append(row)

    if PRINT_LIVE_LOG:
        inline_print(
            f"[SAMPLE] dot={dot_index} seq={packet['seq']} "
            f"dir=({packet['gaze_dir_x']:.3f}, {packet['gaze_dir_y']:.3f}, {packet['gaze_dir_z']:.3f}) "
            f"buffered={len(current_hold_samples)}"
        )



def cancel_hold(packet):
    global current_hold_dot_index, current_hold_samples

    dot_index = packet["calibration_dot_index"]
    if current_hold_dot_index == dot_index:
        print(f"[CANCEL] dot={dot_index} discarded_samples={len(current_hold_samples)}")
        current_hold_dot_index = None
        current_hold_samples = []



def complete_hold(packet):
    global current_hold_dot_index, current_hold_samples, should_stop

    dot_index = packet["calibration_dot_index"]
    if current_hold_dot_index != dot_index:
        return

    committed_rows = list(current_hold_samples)
    samples_by_dot[dot_index].extend(committed_rows)
    all_samples.extend(committed_rows)

    if AUTO_FLUSH_ON_COMPLETE:
        append_rows_to_csv(committed_rows)

    print(f"\n[COMPLETE] dot={dot_index} saved_samples={len(committed_rows)}")

    current_hold_dot_index = None
    current_hold_samples = []

    if dot_index >= STOP_AFTER_DOT_INDEX and not eval_mode:
        print(f"[CALIB] Reached dot {dot_index}. Fitting model, then starting the saccadic evaluation phase.")
        enter_evaluation_mode()

def print_summary():
    print("\n=== Calibration Summary ===")
    if not samples_by_dot:
        print("No completed calibration samples collected.")
        return

    total_saved = 0
    for dot_index in sorted(samples_by_dot.keys()):
        rows = samples_by_dot[dot_index]
        total_saved += len(rows)
        print(f"Dot {dot_index}: saved_samples={len(rows)}")
    
    print(f"Total saved samples: {total_saved}")

print(f"Listening for calibration UDP packets on {HOST}:{PORT}")
print(f"Saving calibration samples to: {OUTPUT_CSV_PATH.resolve()}")

try:
    while True:

        if should_stop:
            break

        while True:
            try:
                data, _ = sock.recvfrom(SOCKET_BUFFER_SIZE)
                newest_data = data
            except BlockingIOError:
                break

            try:
                msg = data.decode("utf-8")
                latest_packet = parse_message(msg)
                recv_count += 1

                event_type = latest_packet["event_type"]
                if event_type == "BEGIN":
                    begin_hold(latest_packet)
                elif event_type == "SAMPLE":
                    if eval_mode:
                        handle_eval_gaze(latest_packet)
                    else:
                        add_sample(latest_packet)
                elif event_type == "GAZE":
                    if eval_mode:
                        handle_eval_gaze(latest_packet)
                    # pre-calibration idle stream: ignore
                elif event_type == "CANCEL":
                    cancel_hold(latest_packet)
                elif event_type == "COMPLETE":
                    complete_hold(latest_packet)
                elif event_type == "SACCADE_BEGIN":
                    begin_saccade_task(latest_packet)
                elif event_type == "SACCADE_FIX":
                    handle_saccade_fix(latest_packet)
                elif event_type == "SACCADE_END":
                    end_saccade_task(latest_packet)
            except Exception as exc:
                inline_print(f"[WARN] Failed to parse packet: {exc}")

        now = time.perf_counter()
        if now - fps_timer >= 1.0:
            recv_fps = recv_count / (now - fps_timer)
            recv_count = 0
            fps_timer = now

            if eval_mode:
                with eval_lock:
                    state = (f"task fix={current_fix['index']}" if current_fix
                             else ("task running" if task_active else "waiting for Test Start"))
                    frames = eval_num_frames
                    rows = eval_num_gaze_rows
                norm_txt = (f"({latest_eval_norm[0]:.3f}, {latest_eval_norm[1]:.3f})"
                            if latest_eval_norm else "lost")
                inline_print(f"[EVAL] recv_fps={recv_fps:.1f} | {state} | "
                             f"mapped_gaze={norm_txt} | gaze_rows={rows} frames={frames}")
            else:
                inline_print(f"[STATUS] recv_fps={recv_fps:.1f} | waiting for calibration data...")

        time.sleep(0.001)

except KeyboardInterrupt:
    print("\nStopping receiver...")
finally:
    stream_stop.set()
    with eval_lock:
        if task_active:
            print("\n[EVAL][WARN] exiting mid-task; closing evaluation files.")
        if eval_gaze_f is not None:
            _close_eval_files_locked()

    if not AUTO_FLUSH_ON_COMPLETE and all_samples:
        append_rows_to_csv(all_samples)

    if not eval_mode:
        # Legacy path (interrupted before the 9th dot): fit whatever we have.
        print_summary()
        fit_ridge_regression_model()
    sock.close()
    print(f"CSV saved to: {OUTPUT_CSV_PATH.resolve()}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH.resolve()}")
    if eval_dir is not None:
        print(f"Evaluation data: {eval_dir.resolve()}")