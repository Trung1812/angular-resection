import numpy as np
from scipy.stats import vonmises, uniform

def generate_anchor_positions(num_anchors, target_pos, min_dist=20.0, max_dist=100.0):
    """Simulating anchor positions
    
    Parameters
    ----------
    num_anchors : int 
        number of anchors
    target_pos : Tuple[float, float]
        position of the target
    min_dist: float, default 20
        minimun distance from anchor to target
    max_dist: flow, default 100
        maximun distance from anchor to target
    
    Return
    ------
    anchors : np.ndarray
        the positions of anchor
    """
    anchors = []
    angles = np.linspace(0, 2 * np.pi, num_anchors, endpoint=False)
    for i in range(num_anchors):
        angle = angles[i] + np.random.uniform(-0.1, 0.1)
        distance = np.random.uniform(min_dist, max_dist)
        x = target_pos[0] + distance * np.cos(angle)
        y = target_pos[1] + distance * np.sin(angle)
        anchors.append(np.array([x, y]))
    return np.array(anchors)

def calculate_true_bearings(target_pos, anchor_positions):
    """
    Calculate bearing from anchor position using the geometry formula:

    Parameters
    ----------
    target_pos: Tuple[float, float]
        target position (x, y) coordinate
    anchor_positions: (n, 2) np.ndarray
        positions of anchors
    
    Return
    ------
    true_bearings: np.array
        bearings in radian
    """
    true_bearings = []
    for anchor_pos in anchor_positions:
        delta_x = anchor_pos[0] - target_pos[0]
        delta_y = anchor_pos[1] - target_pos[1]
        bearing = np.arctan2(delta_x, delta_y)
        if bearing < 0:
            bearing += 2 * np.pi
        true_bearings.append(bearing)
    return np.array(true_bearings)

def add_anchor_noise(anchor_true_pos, sigma_gnss, marker_size=0.1, sigma_map=0.05):
    """
    Add noise to anchors (based on noise model)
    Parameters
    ----------
    anchor_true_pos : np.ndarray
        positions of anchor
    sigma_gnss: float
        error from gnss rover (in metre)
    marker_size: np.array
        size of the anchor to calculate error from pointing (in metre)
    sigma_map: floaat
        the error from map digitalization (in meter)

    Return
    ------
    noisy_anchor_pos: np.ndarray
        noise injected anchors' position
    """
    sigma_marker_sq = (marker_size**2) / 12.0
    total_variance_x = sigma_gnss**2 + sigma_marker_sq + sigma_map**2
    total_variance_y = sigma_gnss**2 + sigma_marker_sq + sigma_map**2
    noisy_anchor_pos = anchor_true_pos + np.array([
        np.random.normal(0, np.sqrt(total_variance_x)),
        np.random.normal(0, np.sqrt(total_variance_y))
    ])
    return noisy_anchor_pos

def add_bearing_noise(true_bearing_rad, kappa_main, outlier_prob, kappa_outlier, uniform_outlier_range):
    """
    Injecting noise to bearing
    Parameters
    ----------
    true_bearing_rad : np.array
        ground true bearing
    kappa_main : float
        reciporical of error mean
    outlier_prob : float
        the potion of outliers
    ....
        
    """
    noise_rad = 0.0
    if np.random.rand() < outlier_prob:
        if np.random.rand() < 0.5:
            noise_rad = vonmises.rvs(kappa_outlier, loc=0)
        else:
            noise_rad = uniform.rvs(loc=uniform_outlier_range[0], scale=uniform_outlier_range[1] - uniform_outlier_range[0])
    else:
        noise_rad = vonmises.rvs(kappa_main, loc=0)
    
    noisy_bearing = true_bearing_rad + noise_rad
    noisy_bearing = (noisy_bearing + np.pi) % (2 * np.pi) - np.pi
    return noisy_bearing
