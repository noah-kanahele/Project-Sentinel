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
stub detector.

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
it). It shows the whole path so far as a cyan line, filled dots where the ball was **observed**
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
- **Association / reacquisition**: nearest same-class, above-threshold detection to the Kalman
  prediction inside the gate. Gate radius = `base + growth_per_frame_lost * misses +
  covariance_scale * predicted_position_sigma`, capped. A detection inside the gate inherits the
  track ID, which is reacquisition. Detections outside the gate are ignored.

## Chosen defaults and why

Sweeps used 20-30 seeds per cell on the synthetic ball (60 FPS, parabola, 2 px center noise,
2% random dropouts). Final defaults in `configs/default.yaml`:

| parameter | value | note |
|---|---|---|
| `gate_base_radius_px` | 40 | floor; 20 to 60 behaved similarly once growth was small |
| `gate_size_scale` | 1.0 | gate is at least one object size wide; added after real footage (below) |
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
       3      0.5  100.0%      0         1       1.6       1.5      3.7
       8      0.0  100.0%      0         0       1.6       1.4      3.3
       8      0.1  100.0%      1         1       1.6       1.8      3.3
       8      0.5  100.0%      0         2       2.2       1.5      3.9
      15      0.0  100.0%      0         0       2.2       1.5      3.9
      15      0.1  100.0%      1         1       2.2       1.9      3.8
      15      0.5  100.0%      0         3       3.6       1.8      4.6
      30      0.0  100.0%      0         0       3.8       2.0      6.4
      30      0.1   96.7%      2         2       4.0       5.4      7.7
      30      0.5   83.3%      2        13       9.7      26.2    195.6
```

Under harsher conditions (6 px noise, 3 ms timestamp jitter, 10% dropouts, 0.2 FP/frame)
reacquisition stayed at 100% for occlusions up to 15 frames and 93% at 30 frames.
The weak spot is long occlusions with many false positives (30 frames, 0.5 FP/frame: 83%).

**Real-footage fix (gate size).** On a 186 px pink gumball being pushed around a table (not
ballistic), both YOLO and the color detector lost the track at frame 113 even though the detector
saw the ball in every frame. The ball changed direction, the prediction lagged 80-100 px, and the
fixed 50 px gate rejected the true detection. The synthetic ball is only 24 px, so the fixed pixel
gate never showed the problem. The gate is now at least `gate_size_scale` object sizes (larger
box side), which holds all 162 frames with no losses. Synthetic sweep results are identical, since
a 24 px ball stays under the 40 px floor.

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
- No clip yet has a full throw-and-occlude arc with the ball in frame throughout. The gumball clip
  is a rolling/pushed ball with no occlusion, so it tests the gate but not reacquisition.
- The click-to-select display window and the `label_clip.py` review window are untested (their
  logic is covered; the interactive UI has not been exercised).
- The live webcam path has not been run on a real camera.
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
