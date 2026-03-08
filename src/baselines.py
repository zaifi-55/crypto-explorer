import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def time_aware_splits(df, test_size=0.2, val_size=0.0, sort_by=None):
    """
    Returns: train_df, val_df (or None), test_df
    test_size, val_size: fractions (e.g. 0.2)
    """
    if sort_by is not None:
        df = df.sort_values(sort_by).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    test_n = int(np.ceil(n * test_size))
    val_n = int(np.ceil(n * val_size))
    train_end = n - test_n - val_n
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:train_end + val_n].copy() if val_n > 0 else None
    test = df.iloc[train_end + val_n:].copy()
    return train, val, test


def naive_forecast(train, test):
    """Naive forecast: predict last observed value."""
    return pd.Series(train.iloc[-1], index=test.index)


def moving_average_forecast(train, test, window=5):
    """Moving average forecast."""
    ma_value = train.rolling(window=window).mean().iloc[-1]
    return pd.Series(ma_value, index=test.index)


def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: pandas.Series aligned by index. Will drop NaNs.
    Return dict with MAE, RMSE, MAPE.
    """
    mask = y_true.notna() & y_pred.notna()
    y_t = y_true[mask].astype(float)
    y_p = y_pred[mask].astype(float)

    if len(y_t) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    
    mae = mean_absolute_error(y_t, y_p)
    rmse = mean_squared_error(y_t, y_p, squared=False)

    y_t_safe = y_t.replace(0, np.finfo(float).eps)
    mape = (np.abs((y_t - y_p) / y_t_safe)).mean() * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def evaluate_baselines(
    df,
    target_col="Close",
    ma_windows=[5, 10, 20],
    test_size=0.2,
    val_size=0.1
):
    """
    Args:
        df (pd.DataFrame): Input dataframe with target column.
        target_col (str): Column to forecast on.
        ma_windows (list[int]): Window sizes for moving average.
        test_size (float): Fraction of test set.
        val_size (float): Fraction of validation set (from train).
    
    Returns:
        results_df (pd.DataFrame): Performance metrics
        preds (dict): Predictions for each baseline
        train, val, test (pd.Series): Train/val/test splits
    """
    
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in dataframe. Available: {df.columns.tolist()}")

    series = df[target_col].dropna().copy()

    # Split train/val/test
    n = len(series)
    test_n = int(n * test_size)
    val_n = int((n - test_n) * val_size)

    train = series.iloc[:n - test_n - val_n]
    val = series.iloc[n - test_n - val_n:n - test_n] if val_n > 0 else None
    test = series.iloc[n - test_n:]

    preds = {}

    # --- Naive last-value baseline ---
    preds["naive"] = naive_forecast(train, test)

    # --- Moving averages ---
    for w in ma_windows:
        if len(train) >= w:  # Only calculate if we have enough data
            preds[f"ma_{w}"] = moving_average_forecast(train, test, window=w)
        else:
            # If not enough data for the window, use naive forecast
            preds[f"ma_{w}"] = naive_forecast(train, test)

    # Collect results
    rows = []
    for name, yhat in preds.items():
        try:
            # Ensure alignment and drop NaN values
            mask = test.notna() & yhat.notna()
            y_true = test[mask]
            y_pred = yhat[mask]
            
            if len(y_true) == 0:
                rows.append({"model": name, "mse": np.nan, "mae": np.nan})
                continue

            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

            rows.append({"model": name, "mse": mse, "mae": mae})
        except Exception as e:
            print(f"Error calculating metrics for {name}: {e}")
            rows.append({"model": name, "mse": np.nan, "mae": np.nan})

    results_df = pd.DataFrame(rows)

    return results_df, preds, train, val, test