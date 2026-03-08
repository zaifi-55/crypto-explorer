# src/modeling.py
from __future__ import annotations
import os
import json
import joblib
import platform
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import sklearn

from .config import MODELS_DIR

# ---------- helpers

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

RMSE_SCORER = make_scorer(rmse, greater_is_better=False)
MAE_SCORER = make_scorer(mean_absolute_error, greater_is_better=False)

@dataclass
class CVResult:
    model_name: str
    best_params: Dict[str, Any]
    mean_rmse: float
    mean_mae: float
    mean_r2: float
    std_rmse: float
    std_mae: float
    std_r2: float
    estimator: Any


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Fit a model, make predictions, and return metrics + predictions.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return {"mae": mae, "rmse": rmse}, preds

# ---------- model zoo

def get_candidate_models() -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    """
    Returns a dict:
      name -> (pipeline, param_grid)
    """
    # LinearRegression (no hyperparams, but keep consistent pipeline)
    lin = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression())
    ])
    lin_grid = {}  # no tuning

    # Ridge with small log-spaced alpha grid
    ridge = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge(random_state=42))
    ])
    ridge_grid = {
        "model__alpha": list(np.logspace(-3, 3, 7))  # 1e-3 ... 1e3
    }

    # RandomForestRegressor (small grid)
    rf = Pipeline([
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    rf_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 3]
    }

    return {
        "LinearRegression": (lin, lin_grid),
        "Ridge": (ridge, ridge_grid),
        "RandomForest": (rf, rf_grid),
    }

# ---------- time-series CV

def make_tscv(n_splits: int = 5, test_size: int | None = None, gap: int = 0) -> TimeSeriesSplit:
    """
    Expanding-window CV by default (sklearn's TimeSeriesSplit is expanding).
    If test_size is None, it auto-splits based on n_splits.
    """
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

def split_date_ranges(index: pd.Index, splitter: TimeSeriesSplit) -> List[Dict[str, str]]:
    """
    Build human-readable date ranges for each fold.
    """
    ranges = []
    for train_idx, test_idx in splitter.split(np.arange(len(index))):
        ranges.append({
            "train_start": str(index[int(train_idx[0])]),
            "train_end":   str(index[int(train_idx[-1])]),
            "test_start":  str(index[int(test_idx[0])]),
            "test_end":    str(index[int(test_idx[-1])]),
        })
    return ranges

# ---------- fit/evaluate

def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    tscv: TimeSeriesSplit,
    primary_metric: str = "rmse",
) -> Tuple[List[CVResult], Dict[str, Any]]:
    """
    Trains all candidates with GridSearchCV using TimeSeriesSplit.
    Returns:
      - list of CVResult (one per model)
      - a dict with raw cv_results_ per model for deeper inspection
    """
    assert primary_metric in {"rmse", "mae"}

    scoring = {
        "rmse": RMSE_SCORER,
        "mae": MAE_SCORER,
        "r2":  "r2",
    }
    refit_metric = primary_metric  # choose best by RMSE or MAE

    results: List[CVResult] = []
    raw_by_model: Dict[str, Any] = {}

    for name, (pipe, grid) in get_candidate_models().items():
        # Handle models with no hyperparameters
        if not grid:
            grid = [{}]  # Single empty parameter set
        
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=tscv,
            scoring=scoring,
            refit=refit_metric,
            n_jobs=-1,
            verbose=0,
            return_train_score=False,
        )
        gs.fit(X, y)

        raw_by_model[name] = gs.cv_results_

        # Extract means/stds for the selected (best) row
        i = gs.best_index_
        mean_rmse = -float(gs.cv_results_["mean_test_rmse"][i])
        mean_mae  = -float(gs.cv_results_["mean_test_mae"][i])
        mean_r2   =  float(gs.cv_results_["mean_test_r2"][i])

        std_rmse =  float(gs.cv_results_.get("std_test_rmse", [np.nan]*len(gs.cv_results_["rank_test_rmse"]))[i])
        std_mae  =  float(gs.cv_results_.get("std_test_mae",  [np.nan]*len(gs.cv_results_["rank_test_mae"]))[i])
        std_r2   =  float(gs.cv_results_.get("std_test_r2",   [np.nan]*len(gs.cv_results_["rank_test_r2"]))[i])

        results.append(CVResult(
            model_name=name,
            best_params=gs.best_params_,
            mean_rmse=mean_rmse,
            mean_mae=mean_mae,
            mean_r2=mean_r2,
            std_rmse=std_rmse,
            std_mae=std_mae,
            std_r2=std_r2,
            estimator=gs.best_estimator_,
        ))

    # sort by primary metric ascending
    if primary_metric == "rmse":
        results.sort(key=lambda r: r.mean_rmse)
    else:
        results.sort(key=lambda r: r.mean_mae)

    return results, raw_by_model

# ---------- saving

def save_model_bundle(
    estimator: Any,
    metadata: Dict[str, Any],
    filename_stem: str,
    out_dir: str | None = None
) -> str:
    """
    Saves { 'model': estimator, 'metadata': {...} } via joblib.
    Returns the full path.
    """
    out_dir = out_dir or MODELS_DIR
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{filename_stem}_{ts}.joblib"
    path = os.path.join(out_dir, fname)
    bundle = {
        "model": estimator,
        "metadata": metadata,
    }
    joblib.dump(bundle, path)
    return path

def build_metadata(
    coin: str,
    model_name: str,
    feature_cols: List[str],
    target_col: str,
    cv_date_ranges: List[Dict[str, str]],
    best_params: Dict[str, Any],
    scores: Dict[str, float],
    notes: str | None = None
) -> Dict[str, Any]:
    return {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "coin": coin,
        "model_name": model_name,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "cv_date_ranges": cv_date_ranges,
        "best_params": best_params,
        "scores": scores,  # mean cv scores
        "env": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        },
        "notes": notes or "",
    }

# ---------- complete training pipeline

def train_models_pipeline(
    df: pd.DataFrame, 
    target: str = "target", 
    n_splits: int = 5,
    primary_metric: str = "rmse",
    coin_name: str = "unknown"
) -> pd.DataFrame:
    """
    Full pipeline: prepare data, run cross-validation, and return results DataFrame.
    
    Args:
        df: DataFrame with features and target
        target: name of target column
        n_splits: number of time series CV splits
        primary_metric: metric to optimize ('rmse' or 'mae')
        coin_name: name of the coin for metadata and file naming
    
    Returns:
        DataFrame with model comparison results
    """
    # Prepare features and target
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Assume all columns except target are features
    feature_cols = [col for col in df.columns if col != target]
    X = df[feature_cols]
    y = df[target]
    
    # Remove rows with missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"Training on {len(X)} samples with {len(feature_cols)} features")
    print(f"Target: {target}")
    print(f"Features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
    
    # Set up time series cross-validation
    tscv = make_tscv(n_splits=n_splits)
    
    # Get date ranges for CV folds (assuming index has datetime info)
    try:
        cv_date_ranges = split_date_ranges(X.index, tscv)
    except Exception as e:
        print(f"Warning: Could not extract date ranges from index: {e}")
        # Fallback if index doesn't have datetime info
        cv_date_ranges = [{"fold": i} for i in range(n_splits)]
    
    # Run cross-validation
    print(f"Running {n_splits}-fold time series cross-validation...")
    cv_results, raw_results = cross_validate_models(X, y, tscv, primary_metric)
    
    # Convert results to DataFrame
    results_data = []
    for result in cv_results:
        row = {
            "model_name": result.model_name,
            "mean_rmse": result.mean_rmse,
            "std_rmse": result.std_rmse,
            "mean_mae": result.mean_mae,
            "std_mae": result.std_mae,
            "mean_r2": result.mean_r2,
            "std_r2": result.std_r2,
            "best_params": str(result.best_params)
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    
    # Print results summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    for _, row in results_df.iterrows():
        print(f"{row['model_name']:<15} | RMSE: {row['mean_rmse']:.4f} ± {row['std_rmse']:.4f} | "
              f"MAE: {row['mean_mae']:.4f} ± {row['std_mae']:.4f} | R²: {row['mean_r2']:.4f}")
    
    # Save best model
    best_result = cv_results[0]  # Already sorted by primary metric
    
    metadata = build_metadata(
        coin=coin_name,
        model_name=best_result.model_name,
        feature_cols=feature_cols,
        target_col=target,
        cv_date_ranges=cv_date_ranges,
        best_params=best_result.best_params,
        scores={
            "mean_rmse": best_result.mean_rmse,
            "mean_mae": best_result.mean_mae,
            "mean_r2": best_result.mean_r2
        },
        notes=f"Best model from {len(cv_results)} candidates using {primary_metric}"
    )
    
    model_path = save_model_bundle(
        estimator=best_result.estimator,
        metadata=metadata,
        filename_stem=f"{coin_name}_{best_result.model_name.lower()}",
    )
    
    print(f"\nBest model ({best_result.model_name}) saved to: {model_path}")
    
    return results_df