# TSFM Robustness Benchmark

[English](./README.md) | [中文](./ReadMe-zh.md)

The TSFM Robustness Benchmark is a systematic testing tool designed to evaluate the engineering robustness of Time Series Foundation Models (TSFMs) in edge cases (e.g., frequency mismatch, data contamination, covariate interference). This release includes a systematic evaluation of TimechoAI as the first targeted model. More models will be integrated in subsequent iterations.

## 1. Core Architecture - Layered Architecture
- This project is built on Python 3.12, with core dependencies on `timecho-ai` and `pandas`.

## 2. Directory and File Specifications
- `config/`: Global configuration management module
   - `dataResults.py`: Result Data Processing.
   - `settings.py`: Global environment variable configuration (e.g., `TIMECHO_API_KEY`), etc.
   - `constants.py`: Global constants definition.
- `core/`: Core common component layer (cross-business reuse)
   - `resume.py`: Encapsulates the checkpoint resume mechanism, managing checkpoint states and file persistence.
   - `timecho.py`: Encapsulates TimechoAI API interaction logic.
- `features/`: Business feature implementation layer, containing specific business scenario logic
- `utils/`: Basic utility library, containing stateless pure functions and general entity encapsulations
   - `client.py`: Encapsulates the underlying client connection entity.
   - `files.py`: File operation utilities.
- `run.py`: Unified entry point; bootstraps sys.path and dispatches execution by module name or file path.
- `README.md`: Project documentation, providing an overview, usage instructions, and notes.

## 3. Testing Process
1. Configuration initialization: Reads environment variable from `config/settings.py`.
2. Model initialization: Initializes the TimechoAI model using the provided API key.
3. Testing execution: Executes the specified testing process based on the provided command-line arguments.
4. Result output: Outputs the testing results to the console or specified file.

## 4. Commands and Installation  
- **Create virtual environment** (universal):  
  `python -m venv .venv`

- **Activate the virtual environment** (choose the command based on your OS):  
  - **macOS / Linux**: `source .venv/bin/activate`  
  - **Windows (CMD)**: `.venv\Scripts\activate.bat`  
  - **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`

- **Install dependencies** (**run this the first time after activating the venv**):  
  `python -m pip install timecho-ai pandas`

- **Deactivate the virtual environment** (universal):  
  `deactivate`

>  **Windows PowerShell users**: If you see an error about script execution being disabled, open PowerShell as Administrator and run:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

- **Run**:   
   `python run.py features.futureCovs.conceptDrift.concept_drift_test_v1`  # Concept Drift Test (Simplified Edition)  
   `python run.py features.futureCovs.conceptDrift.concept_drift_test_v2`  # Concept Drift Test (XYZ scenario)  
   `python run.py features.futureCovs.covariant.cov_test`                # Covariate effectiveness  
   `python run.py features.futureCovs.covariant.cov_test_models`         # Covariate Support Test (Iterate All Models)  
   `python ./features/futureCovs/dirtyData/dirty_test.py`                # Dirty data robustness  
   `python ./features/futureCovs/forecastHorizon/forecast_horizon_ablation.py` # C3 Forecast horizon ablation test  
   `python ./features/futureCovs/freqMismatch/frequency_mismatch_test.py`  # C5 Frequency mismatch robustness  
   `python ./features/futureCovs/inputLength/input_length_test.py`    # input_length ablation test  
   `python ./features/futureCovs/irregularSampling/irregular_sampling_test.py`  # Irregular sampling robustness  

## 5. Testing Objectives
- Edge case exploration: Systematically verify the engineering robustness of the model against boundary conditions such as complex queries, replica inconsistencies, and out-of-order time-series writes.  
- Defensive architecture verification: Apply strict engineering standards to test the model's degradation behavior and recovery capabilities under non-ideal inputs.

## 6. Scope of Testing Disclaimer
The test results of this framework are limited by the specific model version, data preprocessing strategy, and runtime environment. This tool aims to provide an objective reference perspective for the engineering defensive architecture design of time-series models, rather than an absolute assertion of the final performance of any commercial product.

