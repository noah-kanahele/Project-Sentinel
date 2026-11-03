#!/usr/bin/env python3
"""Run synthetic sweeps and print result tables.

  python scripts/run_synthetic.py                 # occlusion length x false-positive rate
  python scripts/run_synthetic.py --sweep gate    # gate base radius x growth per lost frame
  python scripts/run_synthetic.py --sweep kalman  # process noise x measurement noise
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from edge_vision.config import load_config
from edge_vision.sim.runner import evaluate_scenario
from edge_vision.tracking.metrics import TrackMetrics, merge_metrics

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _pct(succ: int, total: int) -> str:
    return f"{100.0 * succ / total:5.1f}%" if total else "  n/a "


def print_table(header: Sequence[str], rows: List[Sequence[Any]]) -> None:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    line = "  ".join(str(h).rjust(w) for h, w in zip(header, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).rjust(w) for c, w in zip(r, widths)))


def summary_cells(m: TrackMetrics) -> List[Any]:
    succ, total = m.reacquisition_rate()
    return [
        _pct(succ, total), m.id_switches, m.fp_adoption_events,
        f"{m.occlusion_pred_err_mean:.1f}", f"{m.pred_err_mean:.1f}", f"{m.pred_err_p95:.1f}",
    ]


SUMMARY_HEADER = ["reacq", "id_sw", "fp_adopt", "occl_err", "err_mean", "err_p95"]


def sweep_occlusion_fp(cfg: Mapping[str, Any], seeds: Sequence[int], lengths: Sequence[int], fp_rates: Sequence[float]) -> None:
    rows = []
    for length, fp in itertools.product(lengths, fp_rates):
        m = evaluate_scenario(
            cfg, seeds, {"synthetic.occlusions": ((40, length),), "synthetic.false_positive_rate": fp}
        )
        rows.append([length, fp] + summary_cells(m))
    print(f"\nOcclusion length x false-positive rate ({len(seeds)} seeds each; errors in px)")
    print_table(["occl_len", "fp_rate"] + SUMMARY_HEADER, rows)


def sweep_params(
    cfg: Mapping[str, Any], seeds: Sequence[int], names: Sequence[str], grids: Sequence[Sequence[float]],
    lengths: Sequence[int], fp_rate: float,
) -> None:
    """Grid over config params; each cell pools several occlusion lengths at a fixed FP rate."""
    rows = []
    for combo in itertools.product(*grids):
        pooled: List[TrackMetrics] = []
        for length in lengths:
            overrides: Dict[str, Any] = dict(zip(names, combo))
            overrides.update({"synthetic.occlusions": ((40, length),), "synthetic.false_positive_rate": fp_rate})
            pooled.append(evaluate_scenario(cfg, seeds, overrides))
        rows.append(list(combo) + summary_cells(merge_metrics(pooled)))
    print(f"\nParam sweep, occlusions {list(lengths)}, fp_rate {fp_rate}, {len(seeds)} seeds each")
    print_table([n.split(".")[1] for n in names] + SUMMARY_HEADER, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--sweep", choices=["occlusion", "gate", "kalman"], default="occlusion")
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds per cell")
    ap.add_argument("--lengths", type=int, nargs="+", default=[3, 8, 15, 30])
    ap.add_argument("--fp-rates", type=float, nargs="+", default=[0.0, 0.1, 0.5])
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = list(range(args.seeds))
    if args.sweep == "occlusion":
        sweep_occlusion_fp(cfg, seeds, args.lengths, args.fp_rates)
    elif args.sweep == "gate":
        sweep_params(
            cfg, seeds, ["tracker.gate_base_radius_px", "tracker.gate_growth_per_frame_lost_px"],
            [[20, 40, 60], [4, 12, 24]], args.lengths, max(args.fp_rates),
        )
    else:
        sweep_params(
            cfg, seeds, ["kalman.process_noise_jerk", "kalman.measurement_noise_px"],
            [[50, 500, 5000, 50000], [1.0, 3.0, 6.0]], args.lengths, max(args.fp_rates),
        )


if __name__ == "__main__":
    main()
