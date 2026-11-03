# Project Sentinel (edge-vision)

Laptop-side foundation of a real-time object detection and tracking pipeline that will later
deploy to an NVIDIA Jetson Orin Nano Super. The demo target is tracking a thrown sports ball
through brief occlusion with a **persistent ID**, using a gravity-aware Kalman predictor to drive
reacquisition. The hard part, and the part built and unit-tested here, is the tracking state
machine and reacquisition logic.

Everything runs on CPU against synthetic data and video files. Nothing assumes CUDA, a Jetson,
or a live camera. Every frame carries a `timestamp_s` from its source and all `dt` comes from
those timestamps, never wall-clock (wall-clock is used only to report latency).

## Layout

```
configs/default.yaml        every tunable threshold
src/edge_vision/
  camera/                   FrameSource ABC, VideoFileSource, CameraSource stub
  detection/                Detection, Detector ABC, YoloDetector (ultralytics, optional import),
                            ColorBlobDetector (HSV threshold + contour filter, plain OpenCV)
  tracking/                 kalman.py, state_machine.py, tracker.py, metrics.py
  motion/                   trajectory history, velocity estimate, predicted arc
  sim/                      synthetic.py (generator), runner.py (drive tracker over a sequence)
  telemetry/                CSV logger (per-frame rows + transition events)
  visualization/            OpenCV overlay and slow-motion analysis video writer/player
  pipeline.py               source -> detector -> tracker -> overlay/CSV loop (offline)
  recording.py              FrameRecorder: raw/annotated video + real-timestamp sidecar
  realtime.py               real-time harness: capture thread, latest-frame mailbox, drop/latency stats
  evaluation/               ground-truth labels, autolabel, accuracy scoring with confidence intervals
scripts/                    run_video.py, run_realtime.py, run_synthetic.py, evaluate_tracks.py,
                            label_clip.py, evaluate_accuracy.py
tests/
```

## Setup

Tested on Python 3.9; the code uses `from __future__ import annotations` so it also runs on 3.10+
(JetPack 6 ships 3.10).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # numpy, opencv-python, pyyaml, pytest
pip install -e ".[detection]"      # only for run_video.py (pulls in ultralytics + torch)
```

## Run the tests

```bash
python -m pytest -q
```

The tests need no ultralytics: the end-to-end video test renders an mp4 with OpenCV and uses a
stub detector. Every script also has a smoke test. Lint and type checks (also run by CI on
Python 3.9 and 3.12, see `.github/workflows/ci.yml`):

```bash
pip install -e ".[dev]"      # pytest, ruff, mypy
ruff check src scripts tests && mypy
```

## Run the synthetic sweep

```bash
python scripts/run_synthetic.py                  # occlusion length x false-positive rate
python scripts/run_synthetic.py --sweep gate     # gate base radius x growth per lost frame
python scripts/run_synthetic.py --sweep kalman   # process noise x measurement noise
python scripts/evaluate_tracks.py --occlusion 3 8 15 30 --fp-rate 0.2 --seeds 40   # detailed metrics
python scripts/evaluate_tracks.py --set tracker.gate_base_radius_px=30             # any config override
```

Columns: `reacq` = fraction of occlusions after which the **same track id** was observed again on a
true detection within `reacq_deadline_frames`; `id_sw` = ID switches; `fp_adopt` = false-positive
adoption events (tracker matched a detection >25 px from truth); `occl_err` = mean prediction
error inside occlusions; `err_mean` / `err_p95` = prediction error over all frames (px).

## Run on a video

```bash
python scripts/run_video.py path/to/clip.mp4 --config configs/default.yaml --csv logs/out.csv --display
```

Add `--detector color` to use the HSV color-blob detector instead of YOLO (no model, about a
millisecond per frame). Its hue/saturation/value range and blob filters live under
`color_detector:` in the config; the default range is green. For red/pink set `hue_min` greater
than `hue_max` (the range wraps around 0). Detections from either detector are emitted with
`detection.class_id`, so the tracker's class filter matches.

### Slow-motion analysis video

```bash
python scripts/run_video.py data/clip.mov --analysis logs/analysis.mp4 --slowdown 4 --play-analysis
```

After the normal run, `--analysis` also writes an annotated video at `source fps / slowdown`
(default 4x slower; `--play-analysis` opens it in a window when processing finishes, `q` stops
it). It shows the recent path (the last `analysis.trail_frames` = 120 points; `--trail-frames N`, or 0 for the whole path) as a cyan line, filled dots where the ball was **observed**
and hollow magenta markers where the position was **predicted**, a yellow velocity arrow, the
predicted arc, the tracker box and state, and a readout of frame, timestamp, state and speed.
The slowdown only changes the file's playback rate; tracking never uses it.

Click a detection in the window to choose the subject; otherwise the highest-confidence sports
ball is auto-selected whenever there is no live subject (including after a track ends). `q`
quits. The CSV has `frame_idx, timestamp_s, track_id, state, cx, cy, pred_cx, pred_cy,
confidence, detect_latency_ms, total_latency_ms`; state transitions go to `logs/out.events.csv`.
`cx, cy` is the observed box center, or the predicted center while the ball is unobserved;
`confidence` is blank on unobserved frames.

## Real-time testing and accuracy

**Is it real time?** `scripts/run_realtime.py` replays a clip paced to its own timestamps (or reads
a live webcam) through a capture thread that keeps only the newest frame, so latency stays bounded
and frames the pipeline cannot keep up with are counted as *dropped* instead of queueing up. It
reports drops, deadline misses (processing longer than one frame period) and capture-to-output
latency percentiles, and prints PASS/FAIL against the `realtime:` thresholds in the config.

Threads: capture -> latest-frame mailbox -> **processing worker**, with the OpenCV window on the
main thread (required on macOS) showing only the newest annotated frame. This split matters:
when detection, tracking, drawing and `imshow` shared one thread, a live webcam run dropped 1-3%
of frames although detect + track took under 2 ms (the window update stalled the loop). The report
also prints the full per-iteration `loop ms` (including drawing and recording) and the indices of
dropped frames, so a stall can be located.

```bash
python scripts/run_realtime.py data/clip.mov --detector color                  # paced replay
python scripts/run_realtime.py data/clip.mov --detector yolo --speed 2         # 2x faster than real time
python scripts/run_realtime.py --camera 0 --detector color --duration 20 --display   # live webcam
```

Measured on an Apple M1 (CPU only), 1080p 28 FPS gumball clip, 1x replay:

| detector | dropped | latency p50 / p99 | deadline misses | result |
|---|---|---|---|---|
| color | 0% | 3.4 / 11.3 ms | 0 | PASS (also passes at 4x real time, ~105 FPS) |
| YOLO11n, CPU | 18% | 58 / 83 ms | 130 of 133 | **FAIL** |

The tracker itself costs about 0.14 ms per frame. YOLO on laptop CPU cannot hold 30 FPS while
another thread decodes 1080p video; that is the job of the Jetson TensorRT export. The webcam
path (`CameraSource`) is unit-tested with a fake camera but has not been run on a real webcam yet.

**Record a session for analysis.** Add `--record` to save exactly what the camera saw:

```bash
python scripts/run_realtime.py --camera 0 --detector color --display --record data/try1
```

This writes `data/try1.mp4` (raw frames, no drawing), `data/try1.timestamps.csv` (the real
per-frame times; `VideoFileSource` picks this sidecar up automatically, so replays keep the real
frame spacing) and `data/try1_overlay.mp4` (what the tracker drew live, including any frames it
dropped). The raw file can be re-run with any detector or config, labeled with
`label_clip.py`, scored with `evaluate_accuracy.py` or turned into a slow-motion video with
`run_video.py --analysis`. Press `q` to stop and finish the recording.

**How accurate is it?** Accuracy needs ground truth:

```bash
python scripts/label_clip.py data/clip.mov --out data/clip.labels.csv --auto color   # pre-fill, then review
python scripts/evaluate_accuracy.py data/clip.mov --labels data/clip.labels.csv --detector color
python scripts/evaluate_accuracy.py data/clip.mov --labels data/clip.labels.csv --detector yolo --realtime
```

`label_clip.py` pre-fills labels with the color detector and opens a review window (click to set
the ball center, `x` = not visible, `space`/`a` to step, `o` save, `q` quit); labels are
`frame_idx,visible,cx,cy,w,h` CSVs. **Auto-labels are only a starting point:** scoring a detector
against labels it produced is circular, so review them by hand before trusting any number.
`evaluate_accuracy.py` reports, over frames where the ball is visible:

- **hold rate**: tracker center within `center_tolerance_fraction` of the ball's size, with a
  Wilson 95% confidence interval; misses split into no track / wrong location / dropped frame
- ID switches, false observations while the ball is hidden, and reacquisition (same id back within
  `reacq_deadline_frames` after each hidden stretch)
- with `--realtime`, frames dropped by the harness count as misses (deliberately strict; the
  tracker could coast through a drop, which is not credited)

**Claiming 99% takes a lot of data.** A perfect score on 162 frames only supports a lower bound
of 97.7%. Showing at least 99% at 95% confidence needs roughly 300 failure-free frames (more if
there are any failures), and frames from one clip are highly correlated, so pool many clips and
throws. The script prints PASS only when the interval's lower bound clears the target.

## How tracking works

- **Kalman**: constant-acceleration, state `[x, y, vx, vy, ax, ay]`, white-jerk process noise,
  Joseph-form update. The acceleration state learns gravity from the data (an optional prior can be
  set with `gravity_prior_px_s2`). It is predicted every frame with the timestamp delta, so its
  covariance grows through an occlusion.
- **State machine**: `DETECTED -> TRACKING -> TEMPORARILY_LOST -> REACQUIRING -> TRACK_ENDED`,
  driven by consecutive hit/miss counts from the config. Every transition (including creation) is
  emitted as an event with frame index, timestamp, from/to state and reason.
- **Association / reacquisition**: nearest same-class, above-threshold, similar-sized detection
  inside the gate around the Kalman prediction. Gate radius = `max(base, gate_size_scale * object
  size) + growth_per_frame_lost * misses + covariance_scale * predicted_position_sigma`, capped.
  A second, smaller gate is centred on the **last observed position extrapolated with the last
  observed velocity** (finite difference of the last two hits): the smoothed Kalman velocity lags
  an abrupt change such as a throw, a bounce or a hand reversing, the raw observed velocity does
  not. A detection inside either gate inherits the track ID, which is reacquisition. Detections
  outside both gates, or more than `size_ratio_max` times larger or smaller than the track, are
  ignored.

## Chosen defaults and why

Sweeps used 20-30 seeds per cell on the synthetic ball (60 FPS, parabola, 2 px center noise,
2% random dropouts). Final defaults in `configs/default.yaml`:

| parameter | value | note |
|---|---|---|
| `gate_base_radius_px` | 40 | floor; 20 to 60 behaved similarly once growth was small |
| `gate_size_scale` | 1.0 | gate is at least one object size wide; added after real footage (below) |
| `observed_gate_growth_size_fraction` | 0.1 | extrapolation gate grows 10% of object size per missed frame |
| `observed_gate_max_misses` | 10 | extrapolation gate only for the first 10 missed frames (5 is not enough on the real clip; unlimited costs 10 points on long occlusions) |
| `size_ratio_max` | 2.0 | reject detections more than 2x larger/smaller than the track |
| `size_memory_frames` | 15 | a detection may also match the track's largest recent size (a ball that shrinks leaving the frame) |
| `reacquire_search_size_scale` / `_confirm_size_scale` | 6.0 / 1.0 | a lost track adopts a far detection seen twice in a row, within 6 object sizes of where it vanished |
| `reacquire_confirm_window_frames` | 1 | consecutive-frame confirmation; 2-5 tolerate flicker but admit more false positives (below) |
| `gate_growth_per_frame_lost_px` | **2** | was 12 in the first draft; see below |
| `gate_covariance_scale` | 3.0 | adaptive margin from Kalman uncertainty |
| `process_noise_jerk` / `measurement_noise_px` | 500 / 3.0 | 50 to 500 equivalent; 5000+ hurts badly |
| state machine | 3 / 3 / 6 / 60 | confirm / lost / reacquiring / ended |

**Gate growth was the big lever.** With 4 occlusion lengths (3, 8, 15, 30) and 0.2 false
positives per frame (30 seeds), reacquisition vs growth (cov scale 3):

| growth px/frame | reacq | fp_adopt events | err_p95 (px) |
|---:|---:|---:|---:|
| 0 | 100.0% | 5 | 4.4 |
| 2 | 97.5% | 10 | 5.0 |
| 4 | 96.7% | 16 | 5.4 |
| 8 | 89.2% | 44 | 230.8 |
| 12 (old default) | 91.2% | 32 (20 seeds) | 58.6 |

A large gate lets random false positives be adopted during long occlusions, which corrupts the
Kalman state. The prediction through a gap is already accurate (1-4 px error), so the gate does
not need to grow fast. I chose 2 rather than the sweep's winner of 0 because the simulator is
idealized (pure parabola, no drag, no camera motion); real footage has model mismatch and
needs margin. **Re-tune on real clips.**

Final occlusion x false-positive table (30 seeds per cell, defaults above):

```
occl_len  fp_rate   reacq  id_sw  fp_adopt  occl_err  err_mean  err_p95
       3      0.0  100.0%      0         0       1.3       1.4      3.2
       3      0.1  100.0%      1         1       1.3       1.7      3.2
       3      0.5  100.0%      0         0       1.4       1.3      3.4
       8      0.0  100.0%      0         0       1.6       1.4      3.3
       8      0.1  100.0%      1         1       1.6       1.8      3.3
       8      0.5  100.0%      0         1       1.9       1.4      3.5
      15      0.0  100.0%      0         0       2.2       1.5      3.9
      15      0.1  100.0%      1         1       2.2       1.9      3.8
      15      0.5  100.0%      0         2       3.2       1.6      4.2
      30      0.0  100.0%      0         0       3.8       2.0      6.4
      30      0.1  100.0%      1         2       4.0       2.5      6.6
      30      0.5  100.0%      0         8       8.8       3.3      8.1
```

Under harsher conditions (6 px noise, 3 ms timestamp jitter, 10% dropouts, 0.2 FP/frame)
reacquisition stayed at 100% for every occlusion length up to 30 frames. Pushed much harder, with
up to 2 false positives per frame (about 1 in 8 of the frame area covered each second), a 30-frame
occlusion is still reacquired 96.7% of the time, with the 95th-percentile prediction error at
74 px (the previous tracker managed 53.3% and 541 px).

**Real-footage fix (gate size).** On a 186 px pink gumball being pushed around a table (not
ballistic), both YOLO and the color detector lost the track at frame 113 even though the detector
saw the ball in every frame. The ball changed direction, the prediction lagged 80-100 px, and the
fixed 50 px gate rejected the true detection. The synthetic ball is only 24 px, so the fixed pixel
gate never showed the problem. The gate is now at least `gate_size_scale` object sizes (larger
box side), which holds all 162 frames with no losses. Synthetic sweep results are identical, since
a 24 px ball stays under the 40 px floor.

**Real-footage fix 2 (abrupt motion).** A 637-frame webcam recording of a green ball held in a
hand and flung upward made the tracker lose the ball three times even though the detector saw it
in nearly every frame. The ball dropped, then rose at about 75 px per frame within two frames;
the Kalman velocity still said "falling", so the prediction ran away (it reached y = 2726 on a
720 px frame) and the true ball sat outside the gate until the track ended. Measured against the
largest color blob per frame as reference (the detector is near-perfect on these clips, so this
tests the tracker): hold rate **74.9% -> 99.6%** and one track instead of three. Process noise was
tried first and is not the lever (94% plateau from 500 to 5000, worse above). Letting the second
gate grow by a fixed 10 px per frame also fixed the clip but cost the synthetic false-positive
stress about 6-16 points of reacquisition, because 10 px is 40% of the 24 px synthetic ball but
9% of the 110 px real one; growing by a *fraction of object size* costs about 1-2 points.
The extrapolation gate is only used for the first `observed_gate_max_misses` missed frames: left
on indefinitely it cost the worst synthetic cell (30-frame occlusion, 0.5 FP/frame) 10 points
(83.3% -> 73.3%), because a ballistic ball does not return along a straight line; 10 frames keeps
the real-clip gain (99.6%) and restores the synthetic numbers. Remaining caveat: the clip is a hand moving the ball, and `Gren_Ball_Test001` (ball visible for
only 18 short bursts at about 200 px/frame) is still 44% and is not fixable by gating.

Also found in that audit: the color detector gave a 16 px speck confidence 1.0 and the real ball
0.97, so auto-select could pick the speck. Confidence now scales with blob area
(`confident_area_px`).

**Real-footage fix 3 (long loss, far return).** A second 1158-frame webcam recording (`try2`, ball
carried by a hand, repeatedly leaving and re-entering the frame) scored only 67.7% against the
largest-blob reference and created 13 tracks. Camera and drops were fine (29.9 FPS, 0 of 1158
frames dropped after moving processing off the display thread). The failures were long stretches
(up to 44 frames) where the ball was plainly detected but outside every gate: the ballistic
prediction of a lost track runs away (to y = -9162 in one case) when a hand, not gravity, moves
the ball. A lost track now also adopts a size-consistent detection within
`reacquire_search_size_scale` object sizes of where it vanished once it is seen on two consecutive
frames, restarting the filter on the ball (the single-detection false-positive protection of the
gate is kept: one far detection is ignored). Two further defects surfaced while tracing: the
remembered size shrank to 76 px as the ball left the top of the frame so its 130-220 px return
failed the size gate (now also compared with the largest size over the last 15 hits), and
auto-select could pick a 16 px speck on a confidence tie (ties now go to the larger box).
Result on `try2`: **67.7% -> 97.5%, 13 -> 7 tracks** (the rest are real ends: the ball leaves for
more than 60 frames); `try1` 99.6% and the gumball clip 100% are unchanged. Synthetic results
*improved* (30-frame occlusion at 0.5 FP/frame: 83.3% -> 100%).
A correction worth recording: a first version of the confirmation tolerated gaps between the two
sightings (to cope with a flickering, motion-blurred ball). A stale line meant that tolerance was
silently off while I measured; once actually on, it gained nothing on the real clips (97.5% ->
97.6%) and cost the synthetic stress (30-frame occlusion at 2 FP/frame: 96.7% -> 83.3% for a
window of 5), so the default stays strictly consecutive.

## Design decisions (ambiguities resolved the simple way)

- Package lives at the repo root (`src/edge_vision/`); the repo is "Project Sentinel", the Python
  package is `edge_vision`.
- Frame counts in the state machine are in processed frames, not seconds. `max_frames_before_ended:
  60` is one second at 60 FPS; revisit if the Jetson runs slower than the camera.
- A `DETECTED` (unconfirmed) track ends after `frames_missed_before_lost` misses.
- A single hit in `TEMPORARILY_LOST` / `REACQUIRING` returns the track to `TRACKING` (no
  re-confirmation). Requiring several hits would cut false adoptions further at the cost of
  reacquisition latency; not needed by the sweeps.
- Auto-selection after a track ends creates a **new** ID and counts as an ID switch in metrics.
- `evaluate_tracks.py` evaluates in-process synthetic scenarios rather than reading CSVs: the
  required per-frame CSV schema has no "observed" flag, which false-adoption detection needs.
- Config dataclasses have no defaults; every value must come from YAML (missing/unknown keys raise).
- A constant-acceleration Kalman model assumes no drag or bounce. The simulator matches it, so
  synthetic accuracy is optimistic.

## Not verified yet

- `YoloDetector` has only been tried on a few short phone clips. On a pink gumball it labeled
  the ball "orange" (COCO 49) rather than "sports ball" (32), and on a small, motion-blurred green
  ball it gave too few detections to confirm a track. The color detector exists for such cases.
- No clip yet has a clean throw with the whole arc in frame and a deliberate occlusion. The real
  clips (`try1`, `try2`, gumball) are hand-held or pushed balls that leave and re-enter the frame,
  scored against the largest color blob per frame as reference. That tests the tracker (the
  detector is near-perfect there) but is not hand-reviewed ground truth.
- The click-to-select display window and the `label_clip.py` review window are untested (their
  logic is covered; the interactive UI has not been exercised).
- The live webcam path has been run on one MacBook camera (color detector, 1280x720, 30 FPS:
  600+ frames, p99 latency about 2-6 ms). Not tried with YOLO or other cameras.
- No hand-reviewed ground truth exists yet, so no real accuracy number has been established.
- No real-video tuning: the defaults come from synthetic data only.

## Jetson porting notes

1. **CameraSource**: implement it with GStreamer (`nvarguscamerasrc` for CSI or `v4l2src` for USB)
   through `cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)` with an `appsink drop=true max-buffers=1`
   so latency does not build up. Use the **driver/buffer timestamps** (GStreamer PTS), not
   `time.time()`, for `Frame.timestamp_s`.
2. **TensorRT**: export once on the device: `yolo export model=yolo11n.pt format=engine half=True`,
   then point `detection.model` at the `.engine` file and set `detection.device: 0`. Keep
   `image_size` identical to the export size.
3. Re-measure `detect_latency_ms` / `total_latency_ms` (already in the CSV). If real FPS differs
   from the camera FPS, re-check the frame-count thresholds in `state_machine`.
4. Re-tune gate growth and Kalman noise on real clips before trusting the synthetic defaults.

First task on the Jetson: get a live `CameraSource` producing driver-timestamped frames, run the
TensorRT detector on it, and record a throw-and-occlude clip to tune against.
