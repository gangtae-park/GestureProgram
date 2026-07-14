# Gesture

## Calibration + saccadic evaluation (CalibrationScene)

`calibration.py` collects the 9-dot samples and fits the ridge model as
before, but no longer exits after dot 9: it enters an evaluation phase for
the random saccadic task that Unity's SaccadicTaskController runs next
(pinch-hold the "Test Start" dot in the headset -> 3s countdown -> 30 random
fixations, 1 s each, >=3 deg apart, then the next scene loads automatically).

During the task the script maps every gaze sample through the freshly fitted
model and records the ADB screen. Output goes to
`calibration_eval/<timestamp>/`:

- `gaze.csv` — mapped gaze (`norm_x/y`) per sample, with the fixation that
  was on screen at that moment (`fix_index`, `u`, `v`,
  `fix_expected_norm_x/y`) for pixel-error analysis
  (`error_px = (norm - expected) * (1100, 1000)`).
- `fixations.csv` — one row per fixation with its expected screen coords
  (interpolated from the 9-dot norm targets via the grid-normalized u/v that
  Unity sends).
- `screen.mp4` + `frames.csv` — ADB recording of the 30 s task window with
  per-frame receiver_time. The mapped gaze point is burned in as a red dot
  (drawn live through the just-fitted model), so the fixation-vs-gaze offset
  is directly visible; raw coords remain in gaze.csv.

## Pilot study (DataCollectionScene)

Collects gaze + both-hand joint trajectories while participants perform each
referent gesture 20 times, plus a screen recording of every 3s trial window.

```
python pilot_receiver.py            # port 5005, live preview with gaze dot
python pilot_receiver.py --no-preview
```

Prereqs: `calibration_ridge_model.json` (run `calibration.py` first),
adb-connected headset, ffmpeg. Unity side: DataCollectionScene with
PilotSender + PilotStudyController (set Participant Id in the Inspector).

Output per trial under `pilot_data/P{participant}/{referent}/`:

- `trial_NN_samples.csv` — receiver/sender time, raw camera-local gaze dir,
  ridge-mapped `gaze_norm_x/y` (normalized ADB screen coords), world head
  pose, and 25 camera-local joints per hand (MediaPipe naming + the four
  finger metacarpals, so HandPoseRecognizer's extension/curl ratios can be
  reproduced exactly for parameter tuning).
- `trial_NN.mp4` + `trial_NN_frames.csv` — ADB screen recording of the window
  and each frame's receiver_time for exact alignment with the samples.
- `session_log.csv` — one row per completed trial.

To eyeball where the participant looked, burn the gaze dot into the video:

```
python pilot_overlay.py pilot_data/P01/Search/trial_01   # one trial
python pilot_overlay.py pilot_data/P01/Search            # whole folder
```

The overlay video also carries a hand-skeleton side panel (first-person view
of the streamed joints); pass `--no-hands` for the gaze-only version. The ADB
recording lags the pose/gaze stream, so frames are matched to CSV samples
`--video-latency-frames` earlier (default 8, ~267ms at 30fps; set 0 for the
old uncorrected alignment).

To watch the recorded hand motion as a skeleton (first-person view, with the
gaze direction as a red ring), play back any trial:

```
python pilot_hands.py pilot_data/P01/Search/trial_01
```

Keys: SPACE pause, R restart, [ / ] slower/faster, ,/. single-step, Q quit.
The same renderer also runs live inside pilot_receiver.py's preview as a side
panel, so hand tracking can be verified before/while recording.

### Jackknife template export

pilot_hands.py can trim the idle lead-in/out of a trial and export just the
gesture segment in the gesture_templates_unified.json format (the 21-joint
MediaPipe subset of the 25 recorded joints x 3, wrist pinned to (0,0,0), all
joints wrist-relative):

- Interactive: play the trial, pause/step to the gesture start and press `S`,
  step to the end and press `E` (the green range on the timeline is what gets
  exported), then `X` to append the segment to `--out`
  (default `pilot_templates.json`).
- Batch: `python pilot_hands.py pilot_data/P01/Ask/trial_01 --export
  --start 0.8 --end 2.1 --hand right --label Ask --out pilot_templates.json`

`--hand` picks which hand becomes the template (default right), `--label`
defaults to the referent folder name. Untracked samples inside the window are
skipped. Every export also appends a provenance row (source csv, trim window,
hand) to `pilot_templates_log.csv` next to the output JSON, since the
template format itself can't carry metadata.
