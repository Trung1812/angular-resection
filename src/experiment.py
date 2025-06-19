"""
Running Monte Carlo simulation
Usage Modify the config files in experiments/configs to control parameters of the experiment
"""
import numpy as np
import pandas as pd
from noise_models import (generate_anchor_positions,
                          calculate_true_bearings,
                          add_anchor_noise,
                          add_bearing_noise)
from utils import ExperimentConfig
from solver import solve_resection_odr, solve_resection_ols


# --- Main Simulation Loop ---
def run_simulation(
    num_simulations,
    num_anchors,
    kappa_bearing,
    sigma_gnss_m,
    marker_size,
    sigma_map,
    target_true_pos,
    min_dist, max_dist,
    outlier_prob, kappa_outlier, uniform_outlier_range,
    log_df, # Pass the DataFrame to append results
    experiment_name, # From config
    experiment_description, # From config
    solver = "odr"
):
    print(f"Running experiment '{experiment_name}': {experiment_description}")
    print(f"  Num Anchors: {num_anchors}, Bearing Kappa: {kappa_bearing}, Sigma GNSS: {sigma_gnss_m}m")
    
    errors = []
    
    true_anchor_positions = generate_anchor_positions(
        num_anchors, target_true_pos, min_dist, max_dist
    )
    true_bearings_rad = calculate_true_bearings(target_true_pos, true_anchor_positions)
    
    for _ in range(num_simulations):
        noisy_anchors = []
        for anchor_true_pos in true_anchor_positions:
            noisy_anchors.append(
                add_anchor_noise(anchor_true_pos, sigma_gnss_m, marker_size, sigma_map)
            )
        noisy_anchors = np.array(noisy_anchors)
        
        noisy_bearings = []
        for bearing_rad in true_bearings_rad:
            noisy_bearings.append(
                add_bearing_noise(
                    bearing_rad, kappa_bearing, outlier_prob,
                    kappa_outlier, uniform_outlier_range
                )
            )
        noisy_bearings = np.array(noisy_bearings)

        #compute sigma_theta for solver (using inlier kappa (test the effect of robust Huber loss))
        if kappa_bearing > 0:
            solver_sigma_theta = 1.0 / np.sqrt(kappa_bearing)
        else:
            solver_sigma_theta = np.inf 
        
        #compute sigma_anchor for solver
        Sigma_for_solver = np.zeros((num_anchors, 2, 2))

        sigma_marker_sq = (marker_size**2) / 12.0 # marker_size from config
        sigma_map_sq = sigma_map**2               # sigma_map from config

        # Each anchor has the same noise properties in current model
        # use fixed_sigma_gnss_m (or sigma_g from the loop) for the GNSS component.
        total_variance = (sigma_gnss_m**2) + sigma_marker_sq + sigma_map_sq

        for i in range(num_anchors):
            # Assuming x and y noise are independent and have the same variance
            Sigma_for_solver[i, 0, 0] = total_variance # Variance in X
            Sigma_for_solver[i, 1, 1] = total_variance 
        
        if solver == "odr":
            solver_result = solve_resection_odr(noisy_bearings, noisy_anchors,
                                                solver_sigma_theta, Sigma_for_solver)
        elif solver == "ols":
            solver_result = solve_resection_ols(noisy_bearings, noisy_anchors,
                                                solver_sigma_theta)
        else:
            raise Exception(f"Solver must be in [\"ols\", \"odr\"], get {solver} instead")
        estimated_pos = solver_result["position"]
        error = np.linalg.norm(estimated_pos - target_true_pos)
        errors.append(error)
        
    avg_error = np.mean(errors)
    std_error = np.std(errors)
    
    # Append results to the DataFrame
    log_df.loc[len(log_df)] = {
        'experiment_name': experiment_name,
        'experiment_description': experiment_description,
        'num_anchors': num_anchors,
        'bearing_kappa': kappa_bearing,
        'bearing_noise_std_deg_approx': np.degrees(np.sqrt(1/kappa_bearing)) if kappa_bearing > 0 else np.nan,
        'sigma_gnss_m': sigma_gnss_m,
        'avg_localization_error_m': avg_error,
        'std_localization_error_m': std_error,
        'n_simulations': num_simulations
    }
    
    print(f"  Result: Avg Error: {avg_error:.4f} m, Std Error: {std_error:.4f} m")


if __name__ == "__main__":
    # Load configuration
    config = ExperimentConfig()

    # Initialize DataFrame to store results
    results_df = pd.DataFrame(columns=[
        'experiment_name', 'experiment_description', 'num_anchors', 'bearing_kappa',
        'bearing_noise_std_deg_approx', 'sigma_gnss_m', 'avg_localization_error_m',
        'std_localization_error_m', 'n_simulations', 'solver'
    ])
    
    print("Starting localization simulation based on config file...")
    
    global_settings = config.global_settings

    for exp_cfg in config.get_all_experiments():
        # Iterate over number of anchors based on global or specific experiment config
        num_anchors_to_iterate = global_settings.num_anchors_range
        if hasattr(exp_cfg, 'num_anchors_subset') and exp_cfg.num_anchors_subset:
            num_anchors_to_iterate = exp_cfg.num_anchors_subset

        if exp_cfg.type == "fixed_anchor_vary_bearing":
            for num_anchors in num_anchors_to_iterate:
                for kappa in exp_cfg.bearing_noise_kappas:
                    run_simulation(
                        global_settings.n_simulations,
                        num_anchors,
                        kappa,
                        exp_cfg.fixed_sigma_gnss_m, # Fixed anchor noise from experiment config
                        global_settings.marker_size,
                        global_settings.sigma_map,
                        global_settings.target_true_pos,
                        global_settings.anchor_min_dist_from_target,
                        global_settings.anchor_max_dist_from_target,
                        global_settings.outlier_probability,
                        global_settings.outlier_vonmises_kappa,
                        global_settings.outlier_uniform_range,
                        results_df,
                        exp_cfg.name,
                        exp_cfg.description,
                        global_settings.solver
                    )

        elif exp_cfg.type == "fixed_bearing_vary_anchor":
            for num_anchors in num_anchors_to_iterate:
                for sigma_g in exp_cfg.anchor_gnss_sigmas_m:
                    run_simulation(
                        global_settings.n_simulations,
                        num_anchors,
                        exp_cfg.fixed_bearing_kappa, # Fixed bearing noise from experiment config
                        sigma_g, # Varying anchor noise from experiment config
                        global_settings.marker_size,
                        global_settings.sigma_map,
                        global_settings.target_true_pos,
                        global_settings.anchor_min_dist_from_target,
                        global_settings.anchor_max_dist_from_target,
                        global_settings.outlier_probability,
                        global_settings.outlier_vonmises_kappa,
                        global_settings.outlier_uniform_range,
                        results_df,
                        exp_cfg.name,
                        exp_cfg.description,
                        global_settings.solver
                    )
        
        elif exp_cfg.type == "combined_noise":
            for num_anchors in num_anchors_to_iterate:
                for kappa in exp_cfg.bearing_noise_kappas:
                    for sigma_g in exp_cfg.anchor_gnss_sigmas_m:
                        run_simulation(
                            global_settings.n_simulations,
                            num_anchors,
                            kappa, # Varying bearing noise
                            sigma_g, # Varying anchor noise
                            global_settings.marker_size,
                            global_settings.sigma_map,
                            global_settings.target_true_pos,
                            global_settings.anchor_min_dist_from_target,
                            global_settings.anchor_max_dist_from_target,
                            global_settings.outlier_probability,
                            global_settings.outlier_vonmises_kappa,
                            global_settings.outlier_uniform_range,
                            results_df,
                            exp_cfg.name,
                            exp_cfg.description,
                            global_settings.solver
                        )
        else:
            print(f"Warning: Unknown experiment type '{exp_cfg.type}' for experiment '{exp_cfg.name}'. Skipping.")

    # Save final results
    results_df.to_csv(global_settings.log_file, index=False)
    print(f"\nAll simulations complete. Results saved to {global_settings.log_file}")