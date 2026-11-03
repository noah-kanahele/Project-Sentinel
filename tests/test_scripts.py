"""Smoke tests: every script runs end to end and prints what it promises."""
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FPS = 30.0
SIZE = (320, 240)
GREEN = (60, 200, 60)


def run(*args, ok_codes=(0,)):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, *map(str, args)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode in ok_codes, proc.stdout + proc.stderr
    return proc.stdout


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A short mp4 with a green ball in a parabola (readable by the color detector)."""
    path = tmp_path_factory.mktemp("clips") / "ball.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for i in range(45):
        t = i / FPS
        img = np.full((SIZE[1], SIZE[0], 3), 200, np.uint8)
        cv2.circle(img, (int(40 + 150 * t), int(200 - 200 * t + 0.5 * 220 * t * t)), 14, GREEN, -1)
        writer.write(img)
    writer.release()
    return path


def test_run_synthetic_prints_table():
    out = run("scripts/run_synthetic.py", "--seeds", "2", "--lengths", "5", "--fp-rates", "0")
    assert "occl_len" in out and "reacq" in out


def test_evaluate_tracks_prints_report():
    out = run("scripts/evaluate_tracks.py", "--seeds", "2", "--occlusion", "5")
    assert "ID switches" in out and "reacquisition overall" in out


def test_run_video_with_color_detector_writes_csv_and_analysis(clip, tmp_path):
    csv_path, analysis = tmp_path / "out.csv", tmp_path / "analysis.mp4"
    out = run("scripts/run_video.py", clip, "--detector", "color", "--csv", csv_path, "--analysis", analysis)
    assert "45 frames" in out and csv_path.exists() and analysis.exists()
    assert (tmp_path / "out.events.csv").exists()


def test_run_realtime_replay_reports_stats_and_records(clip, tmp_path):
    prefix = tmp_path / "rec"
    out = run("scripts/run_realtime.py", clip, "--detector", "color", "--record", prefix, ok_codes=(0, 1))
    assert "frames captured 45" in out and "latency ms" in out
    assert (tmp_path / "rec.mp4").exists() and (tmp_path / "rec.timestamps.csv").exists()
    assert (tmp_path / "rec_overlay.mp4").exists()


def test_label_then_evaluate_accuracy(clip, tmp_path):
    labels = tmp_path / "ball.labels.csv"
    out = run("scripts/label_clip.py", clip, "--out", labels, "--auto", "color", "--no-review")
    assert "auto-labeled 45 frames" in out and labels.exists()
    out = run("scripts/evaluate_accuracy.py", clip, "--labels", labels, "--detector", "color", ok_codes=(0, 1))
    assert "hold rate" in out and "target hold rate" in out


def test_run_realtime_requires_a_source():
    proc = subprocess.run([sys.executable, "scripts/run_realtime.py"], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "video file or --camera" in proc.stderr
