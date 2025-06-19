#!/usr/bin/env python3
"""
Usage examples
--------------
# Generate 100 datasets (6 anchors each) into ./out/
$ python src/generate_dataset.py --outdir data --n-sim 100 \
      --n-anchors 6 --radius 35 --true-pos 8 5 \
      --bearing-kappa 100 --anchor-gnss-sigma 3.0 \
      --outlier-prob 0.1 --outlier-kappa 2

# Single dataset, custom prefix
$ python generate_dataset.py --outfile scene.csv --n-anchors 4 --radius 25 \
      --bearing-kappa 200 --anchor-gnss-sigma 1.0
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Generator

import numpy as np
from noise_models import *
# --- Main Simulation Function ---

def simulate_scene(
    *,
    num_anchors: int,
    target_pos: Tuple[float, float],
    radius: float, # Used to derive min/max_dist
    bearing_kappa: float, # Main bearing noise kappa
    anchor_gnss_sigma: float, # Sigma_GNSS
    outlier_prob: float,
    outlier_kappa: float, # Kappa for von Mises outlier
    outlier_scale_deg: float, # Scale for uniform outlier (converted to range)
    marker_size: float = 2.0,
    sigma_map: float = 0.15
) -> Dict[str, Any]:
    
    min_dist = radius * 0.8 # Example: 80% of radius
    max_dist = radius * 1.2 # Example: 120% of radius

    outlier_uniform_range = (-np.radians(outlier_scale_deg), np.radians(outlier_scale_deg))

    anchors_true = generate_anchor_positions(num_anchors, np.array(target_pos), min_dist, max_dist)
    true_pos_np = np.asarray(target_pos, float) # Ensure true_pos is a numpy array

    bearing_true = calculate_true_bearings(target_pos, anchors_true)

    if bearing_kappa > 0:
        sigma_theta_for_solver = 1.0 / np.sqrt(bearing_kappa)
    else:
        sigma_theta_for_solver = 1000.0 # Very large for effectively random bearings

    # --- Generate Noisy Anchors and their Covariance Matrices (Sigma) ---
    anchors = []
    Sigma_for_solver = np.zeros((num_anchors, 2, 2))

    sigma_marker_sq = (marker_size**2) / 12.0
    sigma_map_sq = sigma_map**2
    total_anchor_variance = (anchor_gnss_sigma**2) + sigma_marker_sq + sigma_map_sq

    for i, anchor_true_pos in enumerate(anchors_true):
        noisy_anchor_pos = add_anchor_noise(anchor_true_pos, anchor_gnss_sigma, marker_size, sigma_map)
        anchors.append(noisy_anchor_pos)
        
        Sigma_for_solver[i, 0, 0] = total_anchor_variance # Variance in X
        Sigma_for_solver[i, 1, 1] = total_anchor_variance # Variance in Y
        # Off-diagonal elements are 0 for independent noise
    anchors = np.array(anchors)

    # --- Generate Noisy Bearings and Outlier Flags ---
    bearings = []
    for bt in bearing_true:
        noisy_bearing = add_bearing_noise(
            bt,
            bearing_kappa,
            outlier_prob,
            outlier_kappa,
            outlier_uniform_range,
        )
        bearings.append(noisy_bearing)
    bearings = np.array(bearings)

    return {
        "anchors": anchors,
        "Sigma": Sigma_for_solver,
        "bearings": bearings,
        "sigma_theta": np.full(num_anchors, sigma_theta_for_solver), # Pass as (m,) array as per solver interface
        "true_pos": true_pos_np,
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
            "sigma_theta",
            "Sxx", # Sigma[0,0]
            "Sxy", # Sigma[0,1]
            "Syx", # Sigma[1,0] - Added for completeness, usually 0 if Sxy is 0
            "Syy", # Sigma[1,1]
            "true_x",
            "true_y",
        ])
        m = data["anchors"].shape[0]
        for i in range(m):
            x, y = data["anchors"][i]
            b = data["bearings"][i]
            sig_theta = float(data["sigma_theta"][i]) # Should be uniform now
            # Extract elements from the 2x2 covariance matrix
            Sxx, Sxy = data["Sigma"][i, 0, :]
            Syx, Syy = data["Sigma"][i, 1, :] # Syx will be 0 if Sxy is 0 for diagonal
            writer.writerow([i, x, y, b, sig_theta, Sxx, Sxy, Syx, Syy, *data["true_pos"]])

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
    p.add_argument("--radius", type=float, default=30.0, help="Approximate radius for anchor placement around target")
    p.add_argument("--true-pos", nargs=2, type=float, default=(0, 0), metavar=("X", "Y"), help="True target position (x, y)")

    # Noise parameters (updated to match your models)
    p.add_argument("--bearing-kappa", type=float, default=100.0, help="Von Mises kappa for in-spec bearing noise (higher = less noise)")
    p.add_argument("--anchor-gnss-sigma", type=float, default=3.0, help="GNSS system noise (sigma_GNSS) for anchors [m]")
    p.add_argument("--outlier-prob", type=float, default=0.05, help="Probability of a bearing outlier [0-1]")
    p.add_argument("--outlier-kappa", type=float, default=2.0, help="Von Mises kappa for bearing outlier component (lower = wider spread)")
    p.add_argument("--outlier-scale-deg", type=float, default=180.0, help="Scale (half-range) for uniform bearing outlier component [deg]")
    
    # Fixed parameters from your config_manager
    p.add_argument("--marker-size", type=float, default=2.0, help="Physical size of markers (d_i) [m]")
    p.add_argument("--sigma-map", type=float, default=1.5, help="Map digitalization offset [m]")

    return p.parse_args()

# -----------------------------------------------------------------------------
# Main -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.outfile and args.n_sim != 1:
        raise SystemExit("--outfile only allowed with --n-sim 1")

    for k in range(args.n_sim):
    
        scene = simulate_scene(
            num_anchors=args.n_anchors,
            target_pos=tuple(args.true_pos),
            radius=args.radius,
            bearing_kappa=args.bearing_kappa,
            anchor_gnss_sigma=args.anchor_gnss_sigma,
            outlier_prob=args.outlier_prob,
            outlier_kappa=args.outlier_kappa,
            outlier_scale_deg=args.outlier_scale_deg,
            marker_size=args.marker_size, # Pass fixed parameters
            sigma_map=args.sigma_map,     # Pass fixed parameters
        )
        meta = {
            "n_anchors": args.n_anchors,
            "radius": args.radius,
            "true_pos": list(map(float, args.true_pos)),
            "bearing_kappa": args.bearing_kappa,
            "anchor_gnss_sigma": args.anchor_gnss_sigma,
            "outlier_prob": args.outlier_prob,
            "outlier_kappa": args.outlier_kappa,
            "outlier_scale_deg": args.outlier_scale_deg,
            "marker_size": args.marker_size,
            "sigma_map": args.sigma_map,
        }

        if args.outdir:
            fname = args.outdir / f"dataset_{k:03d}.csv"
        else:
            fname = args.outfile
        _write_csv(fname, scene, meta)
        print("wrote", fname)


if __name__ == "__main__":
    main()