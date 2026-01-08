# src/models.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Iterable, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline


# ----- Baselines -----
def predict_naive(n: int) -> np.ndarray:
    """Δy_{t+1}=0 for all t (random-walk-in-levels baseline)."""
    return np.zeros(n, dtype=float)

def fit_ar1(y_train: pd.Series, X_train_ar: pd.DataFrame) -> LinearRegression:
    """AR(1) on ΔOAT using 'OAT_change' as the only regressor."""
    mdl = LinearRegression().fit(X_train_ar, y_train)
    return mdl

# ----- OLS / Ridge / RF -----
def fit_ols(X_tr, y_tr):
    return LinearRegression().fit(X_tr, y_tr)


def fit_ridge_cv(X, y, index, alphas, val_end):
    def folds(idx, first_val_start="2014-01-01", span=12, step=6, last_val_end=val_end):
        import pandas as pd, numpy as np
        cur = pd.to_datetime(first_val_start)
        while True:
            val_start = cur
            val_end_  = (val_start + pd.DateOffset(months=span - 1)).to_period("M").to_timestamp(how="start")
            if val_end_ > pd.to_datetime(last_val_end):
                break
            tr_mask = (idx < val_start)
            va_mask = (idx >= val_start) & (idx <= val_end_)
            yield np.where(tr_mask)[0], np.where(va_mask)[0]
            cur = cur + pd.DateOffset(months=step)

    best_alpha, best = None, (1e9, 1e9)
    for a in alphas:
        maes, mses = [], []
        for tr, va in folds(index):
            pipe = make_pipeline(StandardScaler(), Ridge(alpha=a, random_state=42))
            pipe.fit(X.iloc[tr], y.iloc[tr])          # fit with DataFrame (names ok)
            yhat = pipe.predict(X.iloc[va])           # predict with DataFrame
            maes.append(mean_absolute_error(y.iloc[va], yhat))
            mses.append(mean_squared_error(y.iloc[va], yhat))
        score = (float(np.mean(maes)), float(np.mean(mses)))
        if score < best:
            best, best_alpha = score, a

# Fit final pipeline on all Train+Val data
    best_pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha, random_state=42))
    best_pipe.fit(X, y)
    return best_pipe, best[0], best[1]

def fit_rf_cv(X: pd.DataFrame, y: pd.Series,
              index: pd.DatetimeIndex,
              grid: Iterable[Dict[str, Any]],
              val_end: str) -> Tuple[RandomForestRegressor, float, float]:
    """Walk-forward CV for RF (no scaling)."""
    
    def folds(idx, first_val_start="2014-01-01", span=12, step=6, last_val_end=val_end):
        cur = pd.to_datetime(first_val_start)
        while True:
            val_start = cur
            val_end_  = (val_start + pd.DateOffset(months=span - 1)).to_period("M").to_timestamp(how="start")
            if val_end_ > pd.to_datetime(last_val_end):
                break
            tr_mask = (idx < val_start)
            va_mask = (idx >= val_start) & (idx <= val_end_)
            yield np.where(tr_mask)[0], np.where(va_mask)[0]
            cur = cur + pd.DateOffset(months=step)

    best_params, best = None, (1e9, 1e9)
    for params in grid:
        maes, mses = [], []
        for tr, va in folds(index):
            mdl = RandomForestRegressor(random_state=42, n_jobs=-1, **params).fit(X.iloc[tr], y.iloc[tr])
            yhat = mdl.predict(X.iloc[va])
            maes.append(mean_absolute_error(y.iloc[va], yhat))
            mses.append(mean_squared_error(y.iloc[va], yhat))
        s = (float(np.mean(maes)), float(np.mean(mses)))
        if s < best:
            best, best_params = s, params

    best_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params).fit(X, y)
    return best_model, best[0], best[1]
