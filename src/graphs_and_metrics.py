# src/graphs_and_metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MSE":  mean_squared_error(y_true, y_pred),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred)
    }

def save_table(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_band(dates, y_true, y_hat, resid_trainval, out_path: str, title: str):
    sigma = float(np.sqrt(np.mean(np.asarray(resid_trainval, dtype=float)**2)))
    lo, hi = y_hat - 1.96*sigma, y_hat + 1.96*sigma
    plt.figure(figsize=(11,5))
    plt.plot(dates, y_true, label="Actual ΔOAT (t+1)")
    plt.plot(dates, y_hat,  label="Forecast")
    plt.fill_between(dates, lo, hi, alpha=0.2, label="95% band")
    plt.legend(); plt.grid(alpha=0.3); plt.title(title)
    plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()

def rf_importance_bar(names, importances, out_path: str, title: str):
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,5))
    plt.barh(np.array(names)[order][::-1], np.array(importances)[order][::-1])
    plt.xlabel("Importance"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()

# --- EXTRA PLOTS ---

def plot_series(dates, values, out_path: str, title: str, ylabel: str = ""):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    plt.plot(dates, values, linewidth=1.8)
    plt.title(title); plt.xlabel("Date"); plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()


def plot_residuals_basic(dates, y_true, y_pred, out_prefix: str, model_name: str):
    """Saves: residuals over time, residual ACF(12), residual QQ-plot (3 PNGs)."""
    resid = pd.Series((np.asarray(y_true, float) - np.asarray(y_pred, float)), index=dates)
    # 1) Residuals line
    plot_series(dates, resid.values,
                out_prefix + "_resid_line.png",
                f"{model_name}: residuals over time", ylabel="Residual")
    # 2) ACF
    from statsmodels.graphics.tsaplots import plot_acf
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4.5))
    plot_acf(resid.dropna(), lags=12)
    plt.title(f"{model_name}: residual ACF(12)")
    plt.tight_layout(); plt.savefig(out_prefix + "_resid_acf.png", dpi=200, bbox_inches="tight"); plt.close()
    # 3) QQ
    import statsmodels.api as sm
    plt.figure(figsize=(6.0, 6.0))
    sm.ProbPlot(resid.dropna()).qqplot(line="s")
    plt.title(f"{model_name}: residual QQ-plot")
    plt.tight_layout(); plt.savefig(out_prefix + "_resid_qq.png", dpi=200, bbox_inches="tight"); plt.close()

def plot_ols_coefficients(names, coefs, out_path: str, title: str):
    order = np.argsort(np.abs(coefs))
    names = np.array(names)[order]
    coefs = np.array(coefs)[order]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    ypos = np.arange(len(names))
    plt.barh(names, coefs)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(title); plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()

def plot_key_series_overlay(panel: pd.DataFrame, out_path: str, method: str = "zscore"):
    """
    One PNG with all 5 series on a single axis, after normalization.
    method:
      - 'zscore'  → (x - mean)/std
      - 'index'   → index each series to 100 at the start (x / x0 * 100)
    """
    cols = ["FRANCE_10Y_OAT", "EURO_CPI_YOY", "FRANCE_UNEMP", "FRANCE_MANUF", "EA_3M_INTERBANK"]
    labels = {
        "FRANCE_10Y_OAT":   "OAT 10Y (%)",
        "EURO_CPI_YOY":     "CPI YoY (%)",
        "FRANCE_UNEMP":     "Unemployment (%)",
        "FRANCE_MANUF":     "Manufacturing (YoY %)",
        "EA_3M_INTERBANK":  "EA 3M Interbank (%)",
    }
    df = panel[cols].copy()

    if method == "zscore":
        df = (df - df.mean()) / df.std(ddof=0)
        ylab = "z-score"
        title = "Key macro series (normalized: z-score)"
    elif method == "index":
        base = df.iloc[0]
        df = 100 * df / base
        ylab = "Index (100 = first obs)"
        title = "Key macro series (indexed to 100 at start)"
    else:
        raise ValueError("method must be 'zscore' or 'index'")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for c in cols:
        ax.plot(panel.index, df[c].values, label=labels[c], linewidth=1.6)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Date"); ax.set_ylabel(ylab)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---- NEW: unified test overlay + unified scatter grid ----

def plot_test_overlay_all_models(
    dates,
    y_true,
    preds_dict: dict[str, np.ndarray | pd.Series],
    out_path: str,
    title: str = "Test: actual vs predicted ΔOAT (all models)"
):
    """One figure: Actual vs each model's forecast on the same axes."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates, y_true, label="Actual ΔOAT (t+1)", linewidth=2)

    for name, yhat in preds_dict.items():
        yhat = np.asarray(yhat, dtype=float)
        ax.plot(dates, yhat, label=name, linewidth=1.5)

    ax.axhline(0.0, ls="--", lw=1, color="k", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("ΔOAT (pp)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_grid_all_models(
    dates,
    y_true,
    preds_dict: dict[str, np.ndarray | pd.Series],
    out_path: str,
    title: str = "Test scatter: predicted vs actual (all models)",
    fit_line: bool = True
):
    """
    2x2 (or more rows) grid of scatters: x = predicted, y = actual.
    Shows identity line. Optionally overlays calibration fit y = a + b x.
    Annotates MAE / RMSE / r / a / b in each panel.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    import math

    y = np.asarray(y_true, dtype=float)

    # Common axis limits across panels
    xs = [np.asarray(v, dtype=float) for v in preds_dict.values()]
    x_all = np.concatenate(xs)
    xmin = float(np.nanmin([x_all.min(), y.min()]))
    xmax = float(np.nanmax([x_all.max(), y.max()]))
    pad = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
    lim = (xmin - pad, xmax + pad)
    xx = np.linspace(lim[0], lim[1], 200)

    # Layout
    n = len(preds_dict)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows), sharex=True, sharey=True)
    axs = np.atleast_1d(axs).flatten()

    for ax, (name, yhat) in zip(axs, preds_dict.items()):
        x = np.asarray(yhat, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)

        ax.scatter(x[mask], y[mask], s=24, alpha=0.8)
        ax.plot(xx, xx, ls="--", lw=1.5, color="C7")  # identity

        # Metrics (RMSE computed without the 'squared' kwarg for compatibility)
        mae  = mean_absolute_error(y[mask], x[mask])
        rmse = float(np.sqrt(mean_squared_error(y[mask], x[mask])))
        r    = np.corrcoef(x[mask], y[mask])[0, 1] if (np.std(x[mask]) > 0 and np.std(y[mask]) > 0) else np.nan

        # Optional calibration line
        if fit_line and np.std(x[mask]) > 0:
            a, b = np.polyfit(x[mask], y[mask], 1)
            ax.plot(xx, a + b * xx, lw=1.2, color="C1")
        else:
            a, b = np.nan, np.nan

        ax.set_title(name)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xlabel("Predicted ΔOAT (t+1)")
        ax.set_ylabel("Actual ΔOAT (t+1)")
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.98,
                f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nr={r:.2f}\na={a:.3f}, b={b:.2f}",
                transform=ax.transAxes, ha="left", va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3))

    # Remove unused axes if any
    for j in range(len(preds_dict), len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
