# Forecasting France’s 10Y OAT Yield Changes (2000–2025)

This project forecasts the **one-month-ahead change** in France’s 10-year government bond yield (OAT) using monthly macroeconomic indicators from 2000 to 2025. 
The goal is both **prediction** (out-of-sample performance) and **interpretation** (which indicators are associated with yield movements).

# Research Question
Which model best predicts **monthly changes** in the 10Y OAT yield:
- a **naïve no-change** baseline (Δ=0),
- an **AR(1)** model on yield changes,
- **multivariate OLS**,
- **Ridge regression** (L2-regularized OLS),
- and a **Random Forest Regressor**

Model performance is evaluated using **time-based splits** and the metrics **MAE**, **MSE**, and **R²**.

## Setup

# Create environment
conda env create -f environment.yml
conda activate oat-project

## Usage

python main.py
Expected output: Performance comparison between 5 models.

## Project Structure

my-oat-forecasting/
├── main.py              # Main entry point
├── src/                 # Source code
│   ├── data_loader.py   # Data loading/preprocessing
│   ├── models.py        # Model training
│   └── graphs_and_metrics.py    # Evaluation metrics
├── results/             # Output plots and metrics
│    ├── tables          # Metrics
│    └── figures         # Png plots
├── robustness_check/    # Metrics archive for robustness check
├── data/                
│    ├── raw             # Original data
│    └── parquet         # Processed panel parquet
├── readme.md            # Read me file
├── proposal.md          # Initial proposal from 24/10/25 
└── environment.yml      # Conda environment specification

## Results

Time split: Train 2000–2018, Validation 2019–2021, Test 2022–2025.  
Metrics: **MAE** in percentage points (pp) of monthly yield change; **MSE** in pp².

Validation (2019–2021):
- Naïve (Δ=0): MAE 0.1067, MSE 0.0157, R² -0.0057
- AR(1): MAE 0.0979, MSE 0.0144, R² 0.0784
- **OLS (final spec): MAE 0.0965, MSE 0.0138, R² 0.1135**
- Ridge (tuned, CV): MAE 0.1002, MSE 0.0164
- Random Forest (tuned, CV): MAE 0.0962, MSE 0.0166

Test (2022–2025):
- **Naïve (Δ=0): MAE 0.1705, MSE 0.0561, R² -0.1028**
- AR(1): MAE 0.1755, MSE 0.0571, R² -0.1229
- OLS (final spec): MAE 0.2063, MSE 0.0674, R² -0.3251
- Ridge (tuned): MAE 0.1964, MSE 0.0636, R² -0.2500
- RF (tuned): MAE 0.1895, MSE 0.0615, R² -0.2081

## Requirements
- Python 3.11
- numpy, scikit-learn, pandas, matplotlib, statsmodels, pandas-datareader, requests, ipykernel, pyarrow