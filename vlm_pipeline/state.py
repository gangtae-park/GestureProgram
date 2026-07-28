import threading
from collections import deque

# Signaled by main() in finally block; threads observe this to exit cleanly.
stop_event = threading.Event()


# ---- ADB stream frame (mutated by stream_reader_loop) ----
frame_lock = threading.Lock()
latest_frame = None  # np.ndarray (STREAM_H, STREAM_W, 3) uint8 BGR | None


# ---- Gaze + gesture state (mutated by udp_receiver_loop) ----
gaze_lock = threading.Lock()
latest_gaze_norm = None         # (norm_x, norm_y) | None -- DELAYED by config.GAZE_SCREEN_DELAY_S
latest_is_tracked = False      # tracked flag of the delayed sample above
# Raw (recv_time, tracked, norm) samples awaiting screen alignment: the adb
# stream shows the world GAZE_SCREEN_DELAY_S ago, so screen-mapped gaze is
# consumed from this buffer that much later. Guarded by gaze_lock.
gaze_delay_buffer = deque()
is_gesture_active = False
gesture_name_active = None
gesture_norm_points = []        # list[(nx, ny)] inside current START..END window
gaze_logging_frozen = False     # Compare: True after a READY marker -> stop appending gaze
                                # (the "bring hands together" motion must not pollute the trail)
gesture_start_frame = None      # BGR frame showing the world AT gesture START (head-comp targeting)
gesture_start_capture_due = None  # wall time when the delay-aligned start frame arrives; the
                                  # stream thread fulfils it into gesture_start_frame
gesture_start_head_R = None     # 3x3 head rotation at gesture START (matches that frame's content)
pending_gesture_end = None      # dict {gesture_name, norm_points, ready_at, start_frame}, consumed by main loop
last_gesture_fail = None        # dict {gesture_name, reason, fail_time}, consumed by main loop


# ---- Render state (read by main loop, written by handlers + bg threads) ----
target_lock = threading.Lock()
target_canvas = None            # np.ndarray for the second window


# ---- Network: Unity sender side ----
unity_addr_lock = threading.Lock()
last_unity_addr = None          # (host_ip, source_port) | None  -- auto-detected
unity_sender_sock = None        # socket.socket | None  -- created in network.init_unity_sender_socket()


# ---- Ask gesture: latest target cached, waiting for the user's follow-up question. ----
# Populated by handlers/ask.py at gesture END; consumed when an ASK_QUESTION packet
# arrives from Unity. Holds a reference to the cropped image so VLM can be called
# with the user's question + that crop.
ask_lock = threading.Lock()
latest_ask_target = None  # dict | None  -- {"crop", "target_meta", "gesture_name", "timestamp"}

