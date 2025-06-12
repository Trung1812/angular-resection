#!/usr/bin/env python3
"""
Usage examples
--------------
# Generate 100 datasets (6 anchors each) into ./out/
$ python src/utils.py --outdir data --n-sim 100 \
      --n-anchors 6 --radius 35 --true-pos 8 5 \
      --sigma-theta 0.5 --anchor-sigma 0.015 --outlier-prob 0.1 --seed 42

# Single dataset, custom prefix
$ python generate_dataset.py --outfile scene.csv --n-anchors 4 --radius 25
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Generator

import numpy as np

# -----------------------------------------------------------------------------
# Core simulation util ---------------------------------------------------------
# -----------------------------------------------------------------------------

def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def simulate_scene(
    *,
    n_anchors: int,
    radius: float,
    true_pos: Tuple[float, float],
    sigma_theta_deg: float,
    anchor_sigma: float,
    outlier_prob: float,
    outlier_scale_deg: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    m = n_anchors
    # place anchors uniformly on circle
    phis = np.linspace(0, 2 * np.pi, m, endpoint=False)
    anchors_true = np.column_stack((radius * np.cos(phis), radius * np.sin(phis)))
    anchors = anchors_true + rng.normal(0, anchor_sigma, (m, 2))
    Sigma = np.repeat((anchor_sigma ** 2 * np.eye(2))[None, :, :], m, axis=0)

    true_pos = np.asarray(true_pos, float)
    theta_true = np.arctan2(anchors[:, 1] - true_pos[1], anchors[:, 0] - true_pos[0])

    sigma_theta = np.deg2rad(sigma_theta_deg)
    bearings = theta_true + rng.normal(0, sigma_theta, m)

    is_outlier = rng.random(m) < outlier_prob
    bearings[is_outlier] += rng.normal(0, np.deg2rad(outlier_scale_deg), is_outlier.sum())
    bearings = _wrap_pi(bearings)

    return {
        "anchors": anchors,
        "Sigma": Sigma,
        "bearings": bearings,
        "is_outlier": is_outlier,
        "sigma_theta": np.full(m, sigma_theta),
        "true_pos": true_pos,
    }

# -----------------------------------------------------------------------------
# CSV writer -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _write_csv(path: Path, data: Dict[str, Any], meta: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#META", json.dumps(meta)])
        writer.writerow([
            "id",
            "anchor_x",
            "anchor_y",
            "bearing",
            "is_outlier",
            "sigma_theta",
            "Sxx",
            "Sxy",
            "Syy",
            "true_x",
            "true_y",
        ])
        m = data["anchors"].shape[0]
        for i in range(m):
            x, y = data["anchors"][i]
            b = data["bearings"][i]
            out = bool(data["is_outlier"][i])
            sig = float(data["sigma_theta"][i])
            Sxx, Sxy, Syy = data["Sigma"][i].flatten()[[0, 1, 3]]
            writer.writerow([i, x, y, b, out, sig, Sxx, Sxy, Syy, *data["true_pos"]])

# -----------------------------------------------------------------------------
# CLI --------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch generator for 2D resection CSV datasets")
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument("--outdir", type=Path, help="directory to hold dataset_###.csv files")
    out.add_argument("--outfile", type=Path, help="single output CSV filename (n-sim must be 1)")

    p.add_argument("--n-sim", type=int, default=1, help="number of simulations to produce [1]")
    p.add_argument("--n-anchors", type=int, default=4)
    p.add_argument("--radius", type=float, default=30.0)
    p.add_argument("--true-pos", nargs=2, type=float, default=(0, 0), metavar=("X", "Y"))
    p.add_argument("--sigma-theta", type=float, default=0.5, help="bearing [deg]")
    p.add_argument("--anchor-sigma", type=float, default=0.02, help="anchor GNSS [m]")
    p.add_argument("--outlier-prob", type=float, default=0.05)
    p.add_argument("--outlier-scale", type=float, default=10.0, help="outlier [deg]")
    p.add_argument("--seed", type=int, default=None, help="base RNG seed")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Main -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.outfile and args.n_sim != 1:
        raise SystemExit("--outfile only allowed with --n-sim 1")

    base_seed = args.seed if args.seed is not None else np.random.SeedSequence().entropy

    for k in range(args.n_sim):
        seed_k = (base_seed + k) % 2**32
        rng = np.random.default_rng(seed_k)
        scene = simulate_scene(
            n_anchors=args.n_anchors,
            radius=args.radius,
            true_pos=tuple(args.true_pos),
            sigma_theta_deg=args.sigma_theta,
            anchor_sigma=args.anchor_sigma,
            outlier_prob=args.outlier_prob,
            outlier_scale_deg=args.outlier_scale,
            rng=rng,
        )
        meta = {
            "n_anchors": args.n_anchors,
            "radius": args.radius,
            "true_pos": list(map(float, args.true_pos)),
            "sigma_theta_deg": args.sigma_theta,
            "anchor_sigma": args.anchor_sigma,
            "outlier_prob": args.outlier_prob,
            "outlier_scale_deg": args.outlier_scale,
            "seed": int(seed_k),
        }

        if args.outdir:
            fname = args.outdir / f"dataset_{k:03d}.csv"
        else:
            fname = args.outfile
        _write_csv(fname, scene, meta)
        print("wrote", fname)


if __name__ == "__main__":
    main()
