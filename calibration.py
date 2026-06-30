import csv
import json
import socket
import time
from collections import defaultdict
from pathlib import Path

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

    if event_type == "SAMPLE":
        if len(parts) != 8:
            raise ValueError(f"SAMPLE packet must have 8 values, got {len(parts)}")
        return {
            "event_type": "SAMPLE",
            "seq": int(parts[1]),
            "sender_time": float(parts[2]),
            "is_tracked": int(parts[3]),
            "calibration_dot_index": int(parts[4]),
            "gaze_dir_x": float(parts[5]),
            "gaze_dir_y": float(parts[6]),
            "gaze_dir_z": float(parts[7]),
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

    if dot_index >= STOP_AFTER_DOT_INDEX:
        should_stop = True
        print(f"[STOP] Reached dot {dot_index}. Receiver will stop after fitting the model.")

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
                    add_sample(latest_packet)
                elif event_type == "CANCEL":
                    cancel_hold(latest_packet)
                elif event_type == "COMPLETE":
                    complete_hold(latest_packet)
            except Exception as exc:
                inline_print(f"[WARN] Failed to parse packet: {exc}")

        now = time.perf_counter()
        if now - fps_timer >= 1.0:
            recv_fps = recv_count / (now - fps_timer)
            recv_count = 0
            fps_timer = now

            inline_print(f"[STATUS] recv_fps={recv_fps:.1f} | waiting for calibration data...")

        time.sleep(0.001)

except KeyboardInterrupt:
    print("\nStopping receiver...")
finally:
    if not AUTO_FLUSH_ON_COMPLETE and all_samples:
        append_rows_to_csv(all_samples)

    print_summary()
    fit_ridge_regression_model()
    sock.close()
    print(f"CSV saved to: {OUTPUT_CSV_PATH.resolve()}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH.resolve()}")