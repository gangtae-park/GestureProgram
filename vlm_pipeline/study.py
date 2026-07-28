"""User-study session logging (events + gaze/head stream).

Armed by the researcher panel: study_control.py owns the participant number
and ships it as `STUDY,PNUM,<n>` (and as the 5th field of every START). The
first pnum that arrives arms the logger; until then every call is a no-op,
so normal development runs of app.py produce no files.

Files (study_result/, created the moment the logger is armed):
  P<pnum>_log_<run>.csv   one row per study event (session_ready, input_start, ...)
  P<pnum>_gaze_<run>.csv  30 Hz gaze/head samples, recorded ONLY while a
                          session is active (session_ready .. session_end)

<run> is a per-participant counter (01, 02, ...) chosen by scanning the
directory at arm time, so every launch of the panel/app pair gets fresh
files and nothing is ever overwritten.

Write timing: rows buffer in memory and are APPENDED to the CSVs at every
session_end (plus a final flush at app exit), so a crash can lose at most
the in-flight session.

Gaze row contents: the ridge-mapped screen coords (nx, ny; top-left
normalized, same space the YOLO bboxes live in) PLUS the raw camera-local
gaze direction and the head world-rotation quaternion, so any projection
(world ray, head-motion metrics, re-calibration) can be recomputed offline.
NOTE: nx/ny are mapped from the sample as it arrives; screen-content
alignment lags by config.GAZE_SCREEN_DELAY_S (~267 ms) -- shift in analysis
when comparing against frame-content-derived boxes.
"""
import atexit
import csv
import os
import re
import threading
import time
from datetime import datetime

_lock = threading.Lock()
_event_rows = []
_gaze_rows = []
_events_flushed = 0
_gaze_flushed = 0
_log_path = None
_gaze_path = None
_enabled = False
_pnum = 0
_session = None            # {"id", "referent", "method", "target", "t0"}
_session_counter = 0

STUDY_RESULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "study_result"
)

EVENT_COLUMNS = [
    "pnum", "session_id", "referent", "method", "target_object",
    "event", "detail", "wall_time", "unix_ms", "t_session_s",
]
GAZE_COLUMNS = [
    "pnum", "session_id", "unix_ms", "t_session_s", "tracked",
    "nx", "ny", "gx", "gy", "gz", "qx", "qy", "qz", "qw",
]


def init(pnum: int):
    """Arm the logger for participant `pnum` and create this run's files."""
    global _enabled, _pnum
    _pnum = int(pnum)
    _enabled = True
    atexit.register(save)
    _ensure_files()
    print(f"[STUDY] logging armed for participant P{_pnum:02d} "
          f"-> {os.path.basename(_log_path)} / {os.path.basename(_gaze_path)}")


def set_pnum(pnum) -> bool:
    """Participant number from the researcher panel. First call arms the
    logger; a CHANGE flushes the current files and starts a fresh numbered
    pair for the new participant."""
    global _pnum
    try:
        value = int(str(pnum).strip())
    except (TypeError, ValueError):
        print(f"[STUDY][WARN] ignoring non-numeric pnum {pnum!r}")
        return False
    if not _enabled:
        init(value)
    elif value != _pnum:
        print(f"[STUDY] participant changed P{_pnum:02d} -> P{value:02d}; rotating files.")
        flush()
        _pnum = value
        _reset_files()
        _ensure_files()
    return True


def is_enabled() -> bool:
    return _enabled


# ---------------- event log ----------------

def log_event(event: str, detail: str = ""):
    """Append one timestamped event row (thread-safe). No-op until armed."""
    if not _enabled:
        return
    now = time.time()
    with _lock:
        s = _session
        row = {
            "pnum": _pnum,
            "session_id": s["id"] if s else "",
            "referent": s["referent"] if s else "",
            "method": s["method"] if s else "",
            "target_object": s["target"] if s else "",
            "event": event,
            "detail": detail,
            "wall_time": datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "unix_ms": int(now * 1000),
            "t_session_s": round(now - s["t0"], 3) if s else "",
        }
        _event_rows.append(row)
    print(f"[STUDY] {row['session_id'] or '-'} | {event} | {detail}")


# ---------------- gaze log ----------------

def log_gaze(tracked: bool, norm_xy, gaze_dir, head_quat):
    """One 30 Hz gaze/head sample from the Unity GAZE stream. Recorded only
    while a session is active. Called from the UDP receive loop, so the
    fast path (no session) must stay cheap."""
    if not _enabled or _session is None:
        return
    now = time.time()
    with _lock:
        s = _session
        if s is None:
            return
        _gaze_rows.append({
            "pnum": _pnum,
            "session_id": s["id"],
            "unix_ms": int(now * 1000),
            "t_session_s": round(now - s["t0"], 3),
            "tracked": 1 if tracked else 0,
            "nx": round(norm_xy[0], 5) if (tracked and norm_xy) else "",
            "ny": round(norm_xy[1], 5) if (tracked and norm_xy) else "",
            "gx": round(gaze_dir[0], 6) if gaze_dir else "",
            "gy": round(gaze_dir[1], 6) if gaze_dir else "",
            "gz": round(gaze_dir[2], 6) if gaze_dir else "",
            "qx": round(head_quat[0], 5) if head_quat else "",
            "qy": round(head_quat[1], 5) if head_quat else "",
            "qz": round(head_quat[2], 5) if head_quat else "",
            "qw": round(head_quat[3], 5) if head_quat else "",
        })


# ---------------- session control ----------------

def start_session(referent: str, method: str, target: str):
    global _session, _session_counter
    if not _enabled:
        print("[STUDY][WARN] START received but logging is not armed (no pnum yet).")
        return
    if _session is not None:
        # Researcher hit Start twice -- close the dangling session first so
        # rows never straddle two configurations.
        log_event("session_end", "auto-closed by next session_ready")
        with _lock:
            _session = None
        flush()
    with _lock:
        _session_counter += 1
        _session = {
            "id": _session_counter,
            "referent": referent,
            "method": method,
            "target": target,
            "t0": time.time(),
        }
    log_event("session_ready", f"{referent}/{method}/obj{target}")


def end_session():
    global _session
    if not _enabled:
        return
    if _session is None:
        print("[STUDY][WARN] END received with no active session.")
        return
    log_event("session_end")
    with _lock:
        _session = None
    flush()


def handle_packet(parts: list) -> str:
    """Route a STUDY,... packet (fields after the leading 'STUDY').

    Returns "start" when a session just began (network.py then fires the
    Unity countdown), "end" on session end, "" otherwise.
    """
    if not parts:
        return ""
    sub = parts[0].strip().upper()
    if sub == "PNUM" and len(parts) >= 2:
        set_pnum(parts[1])
        return ""
    if sub == "START" and len(parts) >= 4:
        # 5th field (when present) is the pnum -- arms the logger even if the
        # panel was opened after app.py and no PNUM packet ever landed.
        if len(parts) >= 5 and parts[4].strip():
            set_pnum(parts[4])
        start_session(parts[1].strip(), parts[2].strip(), parts[3].strip())
        return "start"
    if sub == "END":
        end_session()
        return "end"
    if sub == "EVENT" and len(parts) >= 2:
        detail = ",".join(p.strip() for p in parts[2:] if p.strip())
        log_event(parts[1].strip(), detail)
        return ""
    print(f"[STUDY][WARN] unrecognised STUDY packet: {parts!r}")
    return ""


# ---------------- file handling ----------------

def _reset_files():
    global _log_path, _gaze_path, _events_flushed, _gaze_flushed
    global _event_rows, _gaze_rows
    with _lock:
        _log_path = None
        _gaze_path = None
        _event_rows = []
        _gaze_rows = []
        _events_flushed = 0
        _gaze_flushed = 0


def _next_run_number() -> int:
    """Scan study_result for this participant's existing numbered files and
    pick the next free run index (01, 02, ...)."""
    os.makedirs(STUDY_RESULT_DIR, exist_ok=True)
    pattern = re.compile(rf"P{_pnum:02d}_(?:log|gaze)_(\d+)\.csv$")
    highest = 0
    for name in os.listdir(STUDY_RESULT_DIR):
        m = pattern.match(name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def _ensure_files():
    """Create this run's CSV pair with headers (idempotent)."""
    global _log_path, _gaze_path
    if _log_path is not None:
        return
    run = _next_run_number()
    _log_path = os.path.join(STUDY_RESULT_DIR, f"P{_pnum:02d}_log_{run:02d}.csv")
    _gaze_path = os.path.join(STUDY_RESULT_DIR, f"P{_pnum:02d}_gaze_{run:02d}.csv")
    for path, cols in ((_log_path, EVENT_COLUMNS), (_gaze_path, GAZE_COLUMNS)):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
    print(f"[STUDY] run files created: {os.path.basename(_log_path)}, "
          f"{os.path.basename(_gaze_path)}")


def flush():
    """Append all not-yet-written rows to the CSVs. Called at every
    session_end and once more at app exit; safe to call any time."""
    global _events_flushed, _gaze_flushed
    if not _enabled:
        return
    _ensure_files()
    with _lock:
        new_events = _event_rows[_events_flushed:]
        new_gaze = _gaze_rows[_gaze_flushed:]
        _events_flushed = len(_event_rows)
        _gaze_flushed = len(_gaze_rows)
    if new_events:
        with open(_log_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=EVENT_COLUMNS).writerows(new_events)
    if new_gaze:
        with open(_gaze_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=GAZE_COLUMNS).writerows(new_gaze)
    if new_events or new_gaze:
        print(f"[STUDY] flushed +{len(new_events)} events, +{len(new_gaze)} gaze rows.")


def save():
    """Final flush at app exit (atexit + app.py's finally both call this)."""
    flush()
