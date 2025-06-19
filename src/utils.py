import pandas as pd
import logging
import os
import csv
import yaml
import numpy as np
from types import SimpleNamespace

CONFIG_PATH = '/Users/phamquangtrung/Desktop/localization-viettel-mini-project/experiments/configs/experiment_config.yaml'
def load_data(file_path):
    """
    Load data from a CSV file and return it as a NumPy array.
    Also load the metadata from the CSV file.
    Parameters:
    - file_path: str, path to the CSV file
    
    Returns:
    - data: np.ndarray, loaded data
    """
    df = pd.read_csv(file_path)
    data = df.values
    metadata = df.columns.tolist()
    
    return data, metadata


class ExperimentLogger:
    """
    Logger for experiment runs: logs messages to console/file and appends structured results to CSV.
    """
    def __init__(self, csv_path: str, log_path: str = None):
        self.csv_path = csv_path
        # Ensure CSV directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Initialize CSV with header if new
        self._init_csv()
        
        # Set up Python logger
        self.logger = logging.getLogger("ExperimentLogger")
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)
        
        # File handler, if provided
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
    
    def _init_csv(self):
        if not os.path.isfile(self.csv_path):
            # Placeholder header; will be overwritten on first write
            with open(self.csv_path, 'w', newline='') as f:
                pass
    
    def log(self, data_point_info: dict, results: dict):
        """
        Logs an experiment run.
        
        Args:
            data_point_info (dict): Information about the input data point.
            results (dict): Experiment output metrics to log.
        """
        # Combine info
        row = {**data_point_info, **results}
        
        # Write to CSV; write header if first row
        file_exists = os.path.getsize(self.csv_path) > 0
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        # Log a summary message
        summary = ", ".join(f"{k}={v}" for k, v in row.items())
        self.logger.info(f"Experiment logged: {summary}")

class ExperimentConfig:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self._config = self._load_config()
        self.global_settings = SimpleNamespace(**self._config['global_settings'])
        self.experiments = [SimpleNamespace(**exp) for exp in self._config['experiments']]

        # Convert list to range for num_anchors_range if needed, or handle directly
        if hasattr(self.global_settings, 'num_anchors_range') and isinstance(self.global_settings.num_anchors_range, list):
            # Ensure it's a list for iteration, or convert to a proper range object if strict range behavior is needed
            # For simplicity of iteration, keeping it as a list is fine.
            pass # Already a list, no conversion needed for iteration.

        # Convert tuple-like lists to numpy arrays or tuples where appropriate
        self.global_settings.target_true_pos = np.array(self.global_settings.target_true_pos)
        self.global_settings.outlier_uniform_range = tuple(self.global_settings.outlier_uniform_range)

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_global_setting(self, key):
        return getattr(self.global_settings, key)

    def get_experiment_by_name(self, name):
        for exp in self.experiments:
            if exp.name == name:
                return exp
        return None

    def get_all_experiments(self):
        return self.experiments

    def __str__(self):
        return f"ExperimentConfig loaded from {self.config_path}:\n{yaml.dump(self._config, indent=2)}"

# Example Usage (for testing the config_manager)
if __name__ == "__main__":
    config = ExperimentConfig()
    print(config)

    print(f"\nGlobal Settings:")
    print(f"  N_Simulations: {config.global_settings.n_simulations}")
    print(f"  Target True Pos: {config.global_settings.target_true_pos}")
    print(f"  The solver: {config.global_settings.solver}")
    print(f"\nExperiments:")
    for exp in config.get_all_experiments():
        print(f"  - Name: {exp.name}, Type: {exp.type}")
        if exp.type == "fixed_anchor_vary_bearing":
            print(f"    Fixed Sigma GNSS: {exp.fixed_sigma_gnss_m}")
            print(f"    Bearing Kappas: {exp.bearing_noise_kappas}")