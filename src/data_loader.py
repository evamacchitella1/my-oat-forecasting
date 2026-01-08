# src/data_loader.py
from __future__ import annotations
import pandas as pd
from pandas_datareader import data as pdr
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
from datetime import datetime

# ---------- 1) Fetch and align raw monthly series ----------
def get_monthly_fred_data_smart(series_dict: dict[str,str],
                                start_date: datetime,
                                end_date: datetime) -> dict[str, pd.DataFrame]:
    """Fetch each series from FRED/DB.Nomics via pandas_datareader and align to monthly start (MS)."""
    monthly = {}
    for name, code in series_dict.items():
        data = pdr.DataReader(code, "fred", start_date, end_date)  # index = DatetimeIndex
        freq = pd.infer_freq(data.index)
        # Resample only when needed
        if freq in ["D", "W", "QE"]:
            s = data.resample("MS").last()
        else:
            s = data.copy()
        s.columns = [name]
        monthly[name] = s
    return monthly

def calculate_yoy_inflation(series: pd.Series, series_name="EURO_CPI") -> pd.Series:
    """Convert price index to YoY % inflation: (x / x_{t-12} - 1) * 100."""
    yoy = (series / series.shift(12) - 1) * 100
    yoy.name = f"{series_name}_YOY"
    return yoy

# ---------- 2) Build the monthly panel ----------
def build_or_load_panel() -> pd.DataFrame:
    """
    If a processed panel exists at data/processed/panel.parquet, load it.
    Otherwise fetch from FRED (your series), compute CPI YoY, and save it.
    Returns a monthly DataFrame indexed by month-start with columns:
      ['FRANCE_10Y_OAT','FRANCE_UNEMP','FRANCE_MANUF','EURO_CPI','EA_3M_INTERBANK',
       'EURO_CPI_YOY','OAT_change','target']
    """
    out_path = ROOT_DIR / "data" / "processed" / "panel.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return pd.read_parquet(out_path)

    fred_series = {
        "FRANCE_10Y_OAT": "IRLTLT01FRM156N",    # %
        "FRANCE_UNEMP":   "LRHUTTTTFRM156S",    # %
        "FRANCE_MANUF":   "FRAPRMNTO01GYSAM",   # YoY %
        "EURO_CPI":       "CP0000EZCCM086NEST", # index
        "EA_3M_INTERBANK":"IR3TIB01EZM156N",    # %
    }
    start_date = datetime(1999, 1, 1)
    end_date   = datetime(2025, 8, 30)

    monthly = get_monthly_fred_data_smart(fred_series, start_date, end_date)
    df = pd.concat(monthly.values(), axis=1).sort_index()
    raw_path = ROOT_DIR / "data" / "raw" / "fred_monthly_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=True)
    
    # CPI YoY and target (ΔOAT_{t+1})
    df["EURO_CPI_YOY"] = calculate_yoy_inflation(df["EURO_CPI"], "EURO_CPI")
    df["OAT_change"]   = df["FRANCE_10Y_OAT"].diff()
    df["target"]       = df["OAT_change"].shift(-1)  # next month's change

    df.to_parquet(out_path, index=True)
    return df


# ---------- 3) Feature set with lags (your create_lag_specifications, simplified) ----------
def make_lag_spec(df: pd.DataFrame, lag: int , ma_k: int | None = None) -> pd.DataFrame:
    """
    Supervised table with optional trailing MA on:
      - FRANCE_MANUF (level, YoY already)
      - ΔFRANCE_UNEMP (monthly change)
      - ΔEA_3M_INTERBANK (monthly change)
    CPI YoY is *not* smoothed. AR term kept raw. No look-ahead (shift by 'lag').
    """
    spec = pd.DataFrame(index=df.index)

    def MA(s, k):
        return s.rolling(k, min_periods=k).mean() if k and k > 1 else s

    spec["EURO_CPI_YOY_lag"] = df["EURO_CPI_YOY"].shift(lag)
    
    # Interbank: diff → (optional MA) → shift
    interbank_diff = df["EA_3M_INTERBANK"].diff()
    spec["EA_3M_INTERBANK_diff_lag"] = MA(interbank_diff, ma_k).shift(lag)  

    # Unemployment: diff → (optional MA) → shift
    unemp_diff = df["FRANCE_UNEMP"].diff()
    spec["FRANCE_UNEMP_diff_lag"] = MA(unemp_diff, ma_k).shift(lag)         

    spec["FRANCE_MANUF_lag"] = MA(df["FRANCE_MANUF"], ma_k).shift(lag)

    spec["OAT_change"] = df["OAT_change"]

    spec["target"] = df["target"]
    
    return spec.dropna()

# ---------- 4) ADF tests
from statsmodels.tsa.stattools import adfuller

def adf_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        res = adfuller(df[col].dropna())
        rows.append({
            "Variable": col,
            "ADF_Statistic": res[0],
            "p_value": res[1],
            "1%_Critical": res[4]["1%"],
            "5%_Critical": res[4]["5%"],
            "10%_Critical": res[4]["10%"],
            "Stationarity": "STATIONARY" if res[1] <= 0.05 else "NON-STATIONARY",
        })
    return pd.DataFrame(rows)
