
angular_resection/
├── data/
│   ├── anchors.csv               # Anchor ground-truth (or noisy) positions
│   ├── measurements.csv          # Bearing measurements from unknown X
│   └── experiments/              # Field/test datasets
├── src/
│   ├── __init__.py
│   ├── geometry.py               # Bearing computation, residuals, Jacobians
│   ├── noise_models.py           # Gaussian, wrapped normal, etc.
│   ├── odr_solver.py             # Total/orthogonal distance regression with Huber loss
│   ├── simulator.py              # Monte Carlo simulation of synthetic data
│   ├── utils.py                  # Common math, logging, angle normalization
│   └── visualizer.py             # Plotting results and errors
├── notebooks/
│   ├── 01_problem_formulation.ipynb
│   ├── 02_noise_analysis.ipynb
│   ├── 03_simulation_results.ipynb
│   └── 04_field_test_analysis.ipynb
├── tests/
│   └── test_odr.py               # Unit tests for estimation correctness
├── main.py                       # Entry point for running estimation
├── config.yaml                   # Config for parameters, noise settings
├── requirements.txt              # Dependencies
└── README.md
