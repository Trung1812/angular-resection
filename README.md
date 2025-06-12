# Angular Resection: 2D Bearing-Only Localization

This project provides tools and a web UI for 2D bearing-only resection, enabling the estimation of an unknown position using bearings from known anchors. It includes simulation, robust estimation, and visualization capabilities for both synthetic and field data.

## Environment Setup

You can recreate the environment using either Conda or pip:

### Using Conda (recommended)
```bash
conda env create -f environment.yml
conda activate viettel
```

### Using pip
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Repository Structure

```
angular_resection/
├── data/                # Datasets and anchor/measurement files
│   ├── anchors.csv
│   ├── measurements.csv
│   └── experiments/
├── src/                 # Core library: geometry, noise models, ODR solver, etc.
│   ├── __init__.py
│   ├── geometry.py
│   ├── noise_models.py
│   ├── odr_solver.py
│   ├── simulator.py
│   ├── utils.py
│   └── visualizer.py
├── notebooks/           # Jupyter notebooks for analysis and experiments
│   ├── 01_problem_formulation.ipynb
│   ├── 02_noise_analysis.ipynb
│   ├── 03_simulation_results.ipynb
│   └── 04_field_test_analysis.ipynb
├── tests/               # Unit tests
│   └── test_odr.py
├── ui/                  # Streamlit web UI
│   └── streamlit_app.py
├── main.py              # CLI entry point for estimation
├── config.yaml          # Configuration file
├── requirements.txt     # pip dependencies
├── environment.yml      # Conda environment
└── README.md
```

## Running the Code

### 1. Streamlit Web UI
From the project root, run:
```bash
streamlit run ui/streamlit_app.py
```
This launches an interactive web app for anchor input, bearing entry, and visualization of the estimated position and confidence ellipse.

### 2. Command-Line Interface
You can run estimation or simulations via `main.py`:
```bash
python main.py --help
```

### 3. Notebooks
Explore the `notebooks/` directory for step-by-step analysis and results.

---
For questions or contributions, please open an issue or pull request.
