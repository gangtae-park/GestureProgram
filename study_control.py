"""Researcher control panel for the user study.

Run alongside app.py (separate process so Tkinter owns its own main thread,
which macOS requires):

    python app.py                 # terminal 1 -- pipeline
    python study_control.py       # terminal 2 -- this panel

The panel owns the participant number: it is sent as `STUDY,PNUM,<n>` when
changed and rides along on every START, and the first pnum that reaches
app.py arms its study logger. Pick the referent / method / target object,
hit Start, and the panel sends `STUDY,START,<referent>,<method>,<object>,<pnum>`
over UDP to app.py's 5005 socket. app.py logs session_start and fires the
head-locked 3-2-1 countdown + beep on the headset. When the participant
finishes speaking about the result, hit End (`STUDY,END`) to close the
session. All rows land in study_result/ when app.py exits.
"""
import socket
import tkinter as tk
from datetime import datetime

APP_HOST = "127.0.0.1"
APP_PORT = 5005  # app.py's main UDP socket (config.PORT)

REFERENTS = ["Search", "Ask", "Translate", "Compare", "Anchor", "Save", "Capture"]
METHODS = ["Gesture", "UI", "Voice"]
OBJECTS = ["1", "2", "3"]

_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def _send(msg: str):
    _sock.sendto(msg.encode("utf-8"), (APP_HOST, APP_PORT))


class StudyControlPanel:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Study Control")
        root.attributes("-topmost", True)

        self.referent = tk.StringVar(value=REFERENTS[0])
        self.method = tk.StringVar(value=METHODS[0])
        self.target = tk.StringVar(value=OBJECTS[0])
        self.pnum = tk.StringVar(value="0")
        self.session_active = False
        self.session_count = 0

        body = tk.Frame(root, padx=14, pady=12)
        body.pack(fill="both", expand=True)

        pnum_row = tk.Frame(body)
        pnum_row.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        tk.Label(pnum_row, text="Participant #").pack(side="left")
        self.pnum_spin = tk.Spinbox(pnum_row, from_=0, to=99, width=5,
                                    textvariable=self.pnum,
                                    command=self.on_pnum_changed)
        self.pnum_spin.pack(side="left", padx=(6, 0))

        self._radio_column(body, "Referent", REFERENTS, self.referent, 0, row=1)
        self._radio_column(body, "Method", METHODS, self.method, 1, row=1)
        self._radio_column(body, "Object", OBJECTS, self.target, 2, row=1)

        buttons = tk.Frame(body)
        buttons.grid(row=2, column=0, columnspan=3, pady=(14, 4), sticky="ew")
        self.start_btn = tk.Button(
            buttons, text="START (space / s)", width=14, height=2,
            bg="#2e7d32", fg="black", takefocus=0, command=self.on_start,
        )
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.end_btn = tk.Button(
            buttons, text="END (space / e)", width=14, height=2,
            bg="#c62828", fg="black", takefocus=0, state="disabled",
            command=self.on_end,
        )
        self.end_btn.pack(side="left", expand=True, fill="x")

        self.status = tk.Label(body, text="Ready. app.py must be running.",
                               anchor="w", fg="#555")
        self.status.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        # Keyboard shortcuts: space toggles start/end; s / e are explicit.
        # Buttons and radios are takefocus=0, so keys always reach these root
        # binds without also "clicking" a focused widget (no double fire).
        root.bind("<space>", self._on_key_toggle)
        root.bind("<Key-s>", lambda e: self._on_key(self.on_start))
        root.bind("<Key-S>", lambda e: self._on_key(self.on_start))
        root.bind("<Key-e>", lambda e: self._on_key(self.on_end))
        root.bind("<Key-E>", lambda e: self._on_key(self.on_end))

    def _radio_column(self, parent, title, options, var, col, row=0):
        frame = tk.LabelFrame(parent, text=title, padx=8, pady=6)
        frame.grid(row=row, column=col, sticky="nsew", padx=4)
        for opt in options:
            tk.Radiobutton(frame, text=opt, value=opt, variable=var,
                           anchor="w", takefocus=0).pack(fill="x")

    # ---------- keyboard shortcuts ----------

    def _typing_in_pnum(self) -> bool:
        """True while the pnum Spinbox owns keyboard focus -- typing a digit
        or 's'/'e' there must not fire a session command."""
        return self.root.focus_get() is self.pnum_spin

    def _on_key(self, action):
        if not self._typing_in_pnum():
            action()

    def _on_key_toggle(self, _event):
        if self._typing_in_pnum():
            return
        if self.session_active:
            self.on_end()
        else:
            self.on_start()

    def _pnum_value(self) -> str:
        raw = self.pnum.get().strip()
        return raw if raw.isdigit() else "1"

    def on_pnum_changed(self):
        _send(f"STUDY,PNUM,{self._pnum_value()}")

    def on_start(self):
        if self.session_active:
            return  # shortcut pressed twice; END must come first
        ref, method, obj = self.referent.get(), self.method.get(), self.target.get()
        _send(f"STUDY,START,{ref},{method},{obj},{self._pnum_value()}")
        self.session_count += 1
        self.session_active = True
        self.start_btn.config(state="disabled")
        self.end_btn.config(state="normal")
        self._set_status(f"Session #{self.session_count} RUNNING  |  {ref} / {method} / obj{obj}")

    def on_end(self):
        if not self.session_active:
            return  # nothing to end
        _send("STUDY,END")
        self.session_active = False
        self.start_btn.config(state="normal")
        self.end_btn.config(state="disabled")
        self._set_status(f"Session #{self.session_count} ended.")

    def _set_status(self, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.status.config(text=f"[{stamp}] {text}")


if __name__ == "__main__":
    tk_root = tk.Tk()
    StudyControlPanel(tk_root)
    tk_root.mainloop()
