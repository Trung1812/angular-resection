import numpy as np
from scipy.stats import vonmises

def normalize_angle(angle_rad):
    """Normalize angle to range [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def add_vonmises_noise(bearing_rad, sigma_rad, outlier_prob=0.02):
    """
    Adds noise to a bearing using a von Mises distribution and an optional outlier model.
    
    Parameters:
        bearing_rad (float): True bearing in radians.
        sigma_rad (float): Standard deviation of in-spec noise in radians.
        outlier_prob (float): Probability of an outlier sample.
    
    Returns:
        float: Noisy bearing in radians.
    """
    kappa_in_spec = 1 / sigma_rad**2
    if np.random.rand() < outlier_prob:
        # Outlier from broader von Mises or uniform distribution
        return normalize_angle(bearing_rad + vonmises.rvs(kappa=2))
    else:
        return normalize_angle(bearing_rad + vonmises.rvs(kappa=kappa_in_spec))

def build_anchor_covariance(sigma_e, sigma_n, rho, diameter, sigma_map, sigma_drift=0.0):
    """
    Constructs the 2x2 covariance matrix for an anchor point.

    Parameters:
        sigma_e (float): GNSS standard deviation in East direction (meters).
        sigma_n (float): GNSS standard deviation in North direction (meters).
        rho (float): GNSS correlation coefficient between E and N.
        diameter (float): Diameter of physical marker (meters).
        sigma_map (float): Standard deviation due to mapping error (meters).
        sigma_drift (float): Long-term drift standard deviation (meters).

    Returns:
        np.ndarray: 2x2 covariance matrix.
    """
    cov_gnss = np.array([
        [sigma_e**2, rho * sigma_e * sigma_n],
        [rho * sigma_e * sigma_n, sigma_n**2]
    ])
    cov_marker = (diameter**2 / 12.0) * np.eye(2)
    cov_map = sigma_map**2 * np.eye(2)
    cov_drift = sigma_drift**2 * np.eye(2)
    
    return cov_gnss + cov_marker + cov_map + cov_drift

def simulate_bearing_measurements(true_bearing_rads, sigma_rad, outlier_prob):
    """
    Simulates noisy bearing measurements for a list of true bearings.
    
    Parameters:
        true_bearing_rads (list): List of true bearings in radians.
        sigma_rad (float): Standard deviation of in-spec noise in radians.
        outlier_prob (float): Probability of a large spike.
    
    Returns:
        list: List of noisy bearing measurements.
    """
    return [add_vonmises_noise(b, sigma_rad, outlier_prob) for b in true_bearing_rads]

def simulate_anchor_positions(true_anchors, cov_matrices):
    """
    Simulates noisy anchor positions by adding multivariate Gaussian noise.
    
    Parameters:
        true_anchors (list of [x, y]): List of true anchor positions.
        cov_matrices (list of np.ndarray): Corresponding 2x2 covariance matrices.
    
    Returns:
        list: List of noisy anchor positions.
    """
    return [np.random.multivariate_normal(mean=anchor, cov=cov) 
            for anchor, cov in zip(true_anchors, cov_matrices)]


