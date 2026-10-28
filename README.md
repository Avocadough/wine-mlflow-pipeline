# Wine Classification with MLflow

A small MLflow workflow for validating scikit-learn's wine dataset, training a classifier and registering a model.

## Workflow

- [`01_data_validation.py`](mlops_pipeline/scripts/01_data_validation.py) loads `sklearn.datasets.load_wine` and records basic checks.
- [`02_data_preprocessing.py`](mlops_pipeline/scripts/02_data_preprocessing.py) creates a stratified train/test split, logs it as an MLflow artifact and prints its run ID.
- [`03_train_evaluate_register.py`](mlops_pipeline/scripts/03_train_evaluate_register.py) retrieves those artifacts, fits a standard scaler + logistic regression pipeline, records test accuracy and registers the model if its configured threshold is met.
- [`04_transition_model.py`](mlops_pipeline/scripts/04_transition_model.py) assigns an alias to the latest registered version.
- [`lab9/04_load_and_predict.py`](lab9/04_load_and_predict.py) is an example of loading a registered model and predicting a sample.

The repository also contains [`mlops_pipeline/requirements.txt`](mlops_pipeline/requirements.txt), a [GitHub Actions workflow](.github/workflows/main.yml), and historical `mlruns/` and `processed_data/` artifacts.

## Run locally

The workflow uses Python 3.10. From the repository root, in a disposable copy:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r mlops_pipeline/requirements.txt
python mlops_pipeline/scripts/01_data_validation.py
python mlops_pipeline/scripts/02_data_preprocessing.py
```

Copy the **preprocessing run ID printed by step 2**, and pass it to step 3. Use the same MLflow tracking location for both steps so that the run's artifacts can be found.

```bash
python mlops_pipeline/scripts/03_train_evaluate_register.py <PREPROCESSING_RUN_ID> 10.0
python mlops_pipeline/scripts/04_transition_model.py wine-classifier-prod Staging
```

Replace the placeholder without angle brackets. These scripts create or overwrite local split files and MLflow runs/registry records; existing artifacts in the repository are historical. The commands have not been rerun for this README.

The alias setter uses MLflow aliases, whereas `lab9/04_load_and_predict.py` constructs a stage-style URI ending in `/staging`. Check and align the model URI and alias or version before using that example. The sample data comes from scikit-learn; `data/wine.data` is present in the repository but the pipeline loads `load_wine` directly.

## Evaluation limits

The training script uses one held-out split and a registration threshold of `0.1` accuracy; that threshold is a workflow demonstration, not a meaningful quality or deployment guarantee. Historical tracked metrics and the configured Actions workflow have not been independently rerun or verified here.
