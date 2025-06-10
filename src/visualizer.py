# File: test_noise_model.py

import numpy as np
import matplotlib.pyplot as plt
from noise_models import (
    add_vonmises_noise,
    simulate_bearing_measurements,
    build_anchor_covariance,
    simulate_anchor_positions,
)

def plot_geometry(anchors, bearings, x_est):
    pass

def plot_error_histograms(errors):
    pass

def plot_convergence(metrics_over_time):
    pass

def plot_bearing_distribution(true_bearing_rad, sigma_rad, outlier_prob=0.02, n_samples=10000):
    noisy_bearings = simulate_bearing_measurements(
        [true_bearing_rad] * n_samples, sigma_rad, outlier_prob
    )
    angles = np.array(noisy_bearings)
    
    plt.figure(figsize=(6, 4))
    plt.hist(np.degrees(angles), bins=100, density=True, alpha=0.7, color='steelblue')
    plt.axvline(np.degrees(true_bearing_rad), color='red', linestyle='--', label='True Bearing')
    plt.title("Distribution of Noisy Bearing Measurements")
    plt.xlabel("Bearing (degrees)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_anchor_covariance_and_simulation():
    # Define one anchor's true position and error components
    anchor_true = [10.0, 15.0]  # in meters
    sigma_e = 0.012  # 1.2 cm
    sigma_n = 0.010  # 1.0 cm
    rho = 0.0
    diameter = 0.10  # 10 cm
    sigma_map = 0.05  # 5 cm
    sigma_drift = 0.0

    # Build covariance matrix and simulate 500 positions
    cov = build_anchor_covariance(sigma_e, sigma_n, rho, diameter, sigma_map, sigma_drift)
    noisy_positions = simulate_anchor_positions([anchor_true] * 500, [cov] * 500)
    noisy_positions = np.array(noisy_positions)

    plt.figure(figsize=(5, 5))
    plt.scatter(noisy_positions[:, 0], noisy_positions[:, 1], s=5, alpha=0.6, label="Noisy Samples")
    plt.scatter(*anchor_true, color="red", label="True Anchor", marker="x")
    plt.gca().set_aspect('equal')
    plt.title("Anchor Position Noise (Simulated Samples)")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set true bearing and simulate
    true_bearing_deg = 45
    true_bearing_rad = np.radians(true_bearing_deg)
    sigma_deg = 0.5
    sigma_rad = np.radians(sigma_deg)

    print(f"Testing bearing noise model with σ = {sigma_deg}° and outliers = 2%...")
    plot_bearing_distribution(true_bearing_rad, sigma_rad, outlier_prob=0.02)

    print("Testing anchor noise simulation with full covariance model...")
    test_anchor_covariance_and_simulation()
