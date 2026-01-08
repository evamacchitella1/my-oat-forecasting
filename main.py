# main.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Robust project paths
# =========================
THIS_FILE = Path(__file__).resolve()

# If main.py sits in the project root, /src exists next to it.
# Otherwise (rare), fall back one level up.
ROOT_DIR = THIS_FILE.parent if (THIS_FILE.parent / "src").exists() else THIS_FILE.parents[1]

FIG_DIR = ROOT_DIR / "results" / "figures"
TAB_DIR = ROOT_DIR / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] __file__  = {THIS_FILE}")
print(f"[paths] ROOT_DIR = {ROOT_DIR}")
print(f"[paths] FIG_DIR  = {FIG_DIR}")
print(f"[paths] TAB_DIR  = {TAB_DIR}")

# =========================
# Imports (after paths)
# =========================
from src.data_loader import build_or_load_panel, make_lag_spec, adf_summary
from src.models import fit_ar1, fit_ols, fit_ridge_cv, fit_rf_cv
from src.graphs_and_metrics import (
    compute_metrics, save_table,
    plot_key_series_overlay,
    plot_test_overlay_all_models, plot_scatter_grid_all_models,
    plot_residuals_basic, plot_band, plot_ols_coefficients, rf_importance_bar
)

# =========================
# Console helpers
# =========================
def banner(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def tick(msg: str):
    print(f"   ✓ {msg}")

def bullet(msg: str):
    print(f"   • {msg}")

def print_table(rows, title):
    banner(title)
    if not rows:
        print("(no rows)")
        return
    df = pd.DataFrame(rows, columns=["Model", "MAE", "MSE", "R²"])
    fmt = {c: (lambda x: "n/a" if pd.isna(x) else f"{x:.4f}") for c in ["MAE", "MSE", "R²"]}
    print(df.to_string(index=False, formatters=fmt))

def add_result_row(name, split, y_true, y_pred, results_list):
    m = compute_metrics(y_true, y_pred)
    results_list.append({"Model": name, "Split": split, **m})
    return [name, m["MAE"], m["MSE"], m["R2"]]

# =========================
# Winsor helpers
# =========================
def fit_winsor_limits(df: pd.DataFrame, cols: list[str], lower=0.01, upper=0.99):
    qs = df[cols].quantile([lower, upper])
    return {c: (float(qs.loc[lower, c]), float(qs.loc[upper, c])) for c in cols}

def apply_winsor(df: pd.DataFrame, limits: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, (lo, hi) in limits.items():
        if c in out.columns:
            out[c] = out[c].clip(lower=lo, upper=hi)
    return out

# =========================
# Constants
# =========================
TRAIN_END = "2018-12-31"
VAL_END   = "2021-12-31"

BASE_PREDICTORS = [
    "EURO_CPI_YOY_lag",
    "EA_3M_INTERBANK_diff_lag",
    "FRANCE_UNEMP_diff_lag",
    "FRANCE_MANUF_lag",
    "OAT_change",
]

DO_WINSOR     = False
WINSOR_MODE   = "train"     # "train" or "global"
WINSOR_TARGET = False
W_LO, W_HI    = 0.01, 0.99

# =========================
# Main
# =========================
def main():
    banner("OAT Forecast — Reproducible Run")
    banner("1) Data loading & preprocessing")

    panel = build_or_load_panel()
    LAG   = 3
    MA_K  = 3  # set to None to disable smoothing

    sup = make_lag_spec(panel, lag=LAG, ma_k=MA_K).loc["2000-01-01":]

    tick(f"Feature spec: lag L={LAG}; MA applied to MANUF, ΔUNEMP, ΔINTERBANK with k={MA_K if MA_K and MA_K>1 else 'none'}; CPI YoY unsmoothed.")
    tick(f"Loaded monthly panel: shape={panel.shape}, {panel.index.min().date()} → {panel.index.max().date()}")
    tick("Computed EURO_CPI_YOY, ΔOAT, and target = ΔOAT_{t+1}")
    tick(f"Built supervised table (Lag={LAG}): rows={len(sup)}, cols={sup.shape[1]}")

    # predictors used in THIS run
    predictors = list(BASE_PREDICTORS)
    if "OAT_change_MA" in sup.columns:
        predictors.append("OAT_change_MA")
    tick(f"Predictors used: {predictors}")

    # ADF
    adf = adf_summary(sup[predictors + ["target"]])
    adf.round(6).to_csv(TAB_DIR / "adf_results.csv", index=False)
    bullet(f"ADF summary saved → {TAB_DIR / 'adf_results.csv'}")

    # Overview plot
    banner("A) Data overview plots")
    plot_key_series_overlay(panel, str(FIG_DIR / "series_overlay_zscore.png"), method="zscore")
    tick(f"Saved → {FIG_DIR / 'series_overlay_zscore.png'}")

    # Split
    banner("2) Time-based split (no look-ahead)")
    idx   = sup.index
    train = sup.loc[idx <= TRAIN_END]
    val   = sup.loc[(idx > TRAIN_END) & (idx <= VAL_END)]
    test  = sup.loc[idx > VAL_END]
    tick(f"Train   ≤ 2018-12: n={len(train)}")
    tick(f"Valid.  2019-01…2021-12: n={len(val)}")
    tick(f"Test    ≥ 2022-01: n={len(test)}")

    # Winsor
    if DO_WINSOR:
        wins_cols = predictors + (["target"] if WINSOR_TARGET else [])
        if WINSOR_MODE == "global":
            wins_limits = fit_winsor_limits(sup, wins_cols, lower=W_LO, upper=W_HI)
            sup_w = apply_winsor(sup, wins_limits)
            train_w = sup_w.loc[idx <= TRAIN_END]
            val_w   = sup_w.loc[(idx > TRAIN_END) & (idx <= VAL_END)]
            test_w  = sup_w.loc[idx > VAL_END]
        else:
            wins_limits = fit_winsor_limits(train, wins_cols, lower=W_LO, upper=W_HI)
            train_w = apply_winsor(train, wins_limits)
            val_w   = apply_winsor(val,   wins_limits)
            test_w  = apply_winsor(test,  wins_limits)

        bullet(f"Winsor: mode={WINSOR_MODE} at [{W_LO:.3f}, {W_HI:.3f}]")
    else:
        train_w, val_w, test_w = train, val, test
        bullet("Winsor: disabled")

    # X/y
    Xtr, ytr = train_w[predictors], (train_w["target"] if WINSOR_TARGET else train["target"])
    Xva, yva = val_w[predictors],   (val_w["target"]   if WINSOR_TARGET else val["target"])
    Xte, yte = test_w[predictors],  (test_w["target"]  if WINSOR_TARGET else test["target"])

    # ACF/PACF (train)
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    series_train = train_w["OAT_change"].dropna()

    plt.figure(figsize=(6.5, 4.5)); plot_acf(series_train, lags=12)
    plt.title("ACF of ΔOAT (Train)"); plt.tight_layout()
    plt.savefig(str(FIG_DIR / "acf_deltaoat_train.png"), dpi=200, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6.5, 4.5)); plot_pacf(series_train, lags=12)
    plt.title("PACF of ΔOAT (Train)"); plt.tight_layout()
    plt.savefig(str(FIG_DIR / "pacf_deltaoat_train.png"), dpi=200, bbox_inches="tight"); plt.close()
    tick("Saved → acf_deltaoat_train.png, pacf_deltaoat_train.png")

    # Train & tune
    banner("3) Training & tuning")
    bullet("Fit AR(1) on ΔOAT using OAT_change")
    ar = fit_ar1(ytr, train_w[["OAT_change"]]); tick("AR(1) fitted")

    bullet("Fit OLS on macro features + AR term")
    ols = fit_ols(Xtr.values, ytr.values); tick("OLS fitted")

    try:
        plot_ols_coefficients(predictors, ols.coef_, str(FIG_DIR / "ols_coefficients.png"), "OLS coefficients (Train)")
        tick(f"Saved → {FIG_DIR / 'ols_coefficients.png'}")
    except Exception as e:
        bullet(f"OLS coefficients plot skipped: {e}")

    bullet("Tune Ridge via walk-forward CV on Train+Val")
    trval_w = pd.concat([train_w, val_w])
    y_trval = trval_w["target"] if WINSOR_TARGET else pd.concat([train["target"], val["target"]])
    X_cv = trval_w[predictors]; y_cv = y_trval

    ridge, ridge_mae, ridge_mse = fit_ridge_cv(
        X_cv, y_cv, X_cv.index,
        alphas=[0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
        val_end=VAL_END
    )
    tick(f"Ridge tuned: CV mean MAE={ridge_mae:.4f}, MSE={ridge_mse:.4f}")

    bullet("Tune Random Forest via walk-forward CV (small grid)")
    rf, rf_mae, rf_mse = fit_rf_cv(
        X_cv, y_cv, X_cv.index,
        grid=[
            {"n_estimators":300,"max_depth":3,"min_samples_leaf":3,"max_features":"sqrt"},
            {"n_estimators":300,"max_depth":5,"min_samples_leaf":3,"max_features":"sqrt"},
            {"n_estimators":600,"max_depth":5,"min_samples_leaf":5,"max_features":"sqrt"},
            {"n_estimators":600,"max_depth":7,"min_samples_leaf":8,"max_features":"sqrt"},
        ],
        val_end=VAL_END
    )
    tick(f"RF tuned: CV mean MAE={rf_mae:.4f}, MSE={rf_mse:.4f}")

    # Evaluation
    banner("4) Evaluation")
    results = []
    val_rows = [
        add_result_row("Naive(Δ=0)", "Validation", yva, np.zeros_like(yva), results),
        add_result_row("AR(1)",      "Validation", yva, ar.predict(val_w[["OAT_change"]]), results),
        add_result_row("OLS",        "Validation", yva, ols.predict(Xva.values), results),
    ]
    print_table(val_rows, "Validation metrics (hold-out 2019–2021)")

    cv_rows = [
        ["Ridge(tuned-CV)", ridge_mae, ridge_mse, np.nan],
        ["RF(tuned-CV)",    rf_mae,    rf_mse,    np.nan],
    ]
    results.append({"Model":"Ridge(tuned)","Split":"Validation_CV","MAE":ridge_mae,"MSE":ridge_mse,"R2":None})
    results.append({"Model":"RF(tuned)","Split":"Validation_CV","MAE":rf_mae,"MSE":rf_mse,"R2":None})
    print_table(cv_rows, "Validation metrics (walk-forward CV means)")

    # Refit AR/OLS on Train+Val
    X_trv = pd.concat([Xtr, Xva]); y_trv = pd.concat([ytr, yva])
    ar_final  = fit_ar1(y_trv, pd.concat([train_w[["OAT_change"]], val_w[["OAT_change"]]]))
    ols_final = fit_ols(X_trv.values, y_trv.values)

    test_rows = [
        add_result_row("Naive(Δ=0)",   "Test(final)", yte, np.zeros_like(yte), results),
        add_result_row("AR(1)",        "Test(final)", yte, ar_final.predict(test_w[["OAT_change"]]), results),
        add_result_row("OLS",          "Test(final)", yte, ols_final.predict(Xte.values), results),
        add_result_row("Ridge(tuned)", "Test(final)", yte, ridge.predict(Xte), results),
        add_result_row("RF(tuned)",    "Test(final)", yte, rf.predict(Xte), results),
    ]
    print_table(test_rows, "Test metrics (refit on Train+Val, 2022–2025)")

    val_mae = {"Naive(Δ=0)": val_rows[0][1], "AR(1)": val_rows[1][1], "OLS": val_rows[2][1],
               "Ridge(tuned-CV)": ridge_mae, "RF(tuned-CV)": rf_mae}
    winner = min(val_mae, key=val_mae.get)
    banner("Winner (by Validation MAE)")
    print(f"{winner}  →  MAE = {val_mae[winner]:.4f}")

    # Figures & diagnostics
    banner("5) Figures & diagnostics")
    save_table(results, str(TAB_DIR / "oat_metrics_simple.csv"))
    tick(f"Metrics table saved → {TAB_DIR / 'oat_metrics_simple.csv'}")

    yhat_te_ar    = ar_final.predict(test_w[["OAT_change"]])
    yhat_te_ols   = ols_final.predict(Xte.values)
    yhat_te_ridge = ridge.predict(Xte)
    yhat_te_rf    = rf.predict(Xte)

    preds_test = {
        "AR(1)": yhat_te_ar,
        "OLS": yhat_te_ols,
        "Ridge (tuned)": yhat_te_ridge,
        "RF (tuned)": yhat_te_rf
    }

    plot_test_overlay_all_models(
        test_w.index, yte.values, preds_test,
        str(FIG_DIR / "test_overlay_all_models.png"),
        title="Test: actual vs predicted ΔOAT (all models)"
    )
    tick(f"Saved → {FIG_DIR / 'test_overlay_all_models.png'}")

    plot_scatter_grid_all_models(
        test_w.index, yte.values, preds_test,
        str(FIG_DIR / "test_scatter_grid_all_models.png"),
        title="Test scatter: predicted vs actual (all models)", fit_line=True
    )
    tick(f"Saved → {FIG_DIR / 'test_scatter_grid_all_models.png'}")

    alias = {"Ridge(tuned-CV)": "Ridge (tuned)", "RF(tuned-CV)": "RF (tuned)"}
    winner_key = alias.get(winner, winner)
    pred_lookup = {
        "Naive(Δ=0)": np.zeros_like(yte.values),
        "AR(1)": yhat_te_ar,
        "OLS": yhat_te_ols,
        "Ridge (tuned)": yhat_te_ridge,
        "RF (tuned)": yhat_te_rf
    }
    if winner_key not in pred_lookup:
        winner_key = "RF (tuned)"
    yhat_winner = pred_lookup[winner_key]

    plot_residuals_basic(
        dates=test_w.index, y_true=yte.values, y_pred=yhat_winner,
        out_prefix=str(FIG_DIR / "diag_winner"),
        model_name=f"{winner_key} (Test)"
    )
    tick("Saved → diag_winner_resid_line.png, diag_winner_resid_acf.png, diag_winner_resid_qq.png")

    trval_full_w = pd.concat([train_w, val_w])
    X_trv2 = trval_full_w[predictors]
    y_trv2 = trval_full_w["target"] if WINSOR_TARGET else pd.concat([train["target"], val["target"]])
    ols_trv = fit_ols(X_trv2.values, y_trv2.values)
    resid_trv_ols = y_trv2.values - ols_trv.predict(X_trv2.values)

    plot_band(
        test_w.index, yte.values, ols_final.predict(Xte.values),
        resid_trv_ols, str(FIG_DIR / "test_ols_ci.png"),
        "OLS – Test Forecast (95% band)"
    )
    tick(f"Saved → {FIG_DIR / 'test_ols_ci.png'}")

    rf_importance_bar(
        predictors, rf.feature_importances_,
        str(FIG_DIR / "rf_feature_importance.png"),
        "Random Forest feature importance (tuned)"
    )
    tick(f"Saved → {FIG_DIR / 'rf_feature_importance.png'}")

    banner("6) Run complete")

if __name__ == "__main__":
    main()
