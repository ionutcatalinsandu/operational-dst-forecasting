import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def train_imputation_models(
    df: pd.DataFrame,
    min_data_pct: float = 70.0,
    exclude_cols: Optional[List[str]] = None,
    xgb_params: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    Train XGBoost regressors for each column to enable imputation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with missing values
    min_data_pct : float
        Minimum percentage of non-NaN values required per row for training
    exclude_cols : list, optional
        Columns to exclude from imputation model training
    xgb_params : dict, optional
        XGBoost hyperparameters (uses sensible defaults if None)
        
    Returns:
    --------
    dict
        {column_name: {'model': XGBRegressor, 'r2': float, 'mse': float}}
    """
    
    if exclude_cols is None:
        exclude_cols = []
    
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
    
    n_cols = len(df.columns)
    min_cols_required = int(n_cols * min_data_pct / 100)
    
    models = {}
    
    for target_col in df.columns:
        if target_col in exclude_cols:
            continue
        
        # Get feature columns (all except target)
        feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]
        
        # Filter rows: target not NaN AND sufficient feature data
        row_data_count = df[feature_cols].notna().sum(axis=1)
        valid_mask = (df[target_col].notna()) & (row_data_count >= min_cols_required)
        
        df_train = df[valid_mask]
        
        if len(df_train) < 100:  # Need minimum training samples
            print(f"Skipping {target_col}: insufficient training data ({len(df_train)} rows)")
            continue
        
        # Prepare training data (XGBoost handles NaN natively)
        X_train = df_train[feature_cols]
        y_train = df_train[target_col]
        
        # Train model
        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_train)
        r2 = r2_score(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)
        
        models[target_col] = {
            'model': model,
            'r2': r2,
            'mse': mse,
            'feature_cols': feature_cols,
            'n_train_samples': len(df_train)
        }
        
        print(f"Trained {target_col}: R²={r2:.3f}, MSE={mse:.3f}, N={len(df_train)}")
    
    return models


def impute_missing_values(
    df: pd.DataFrame,
    models: Dict[str, Dict],
    min_data_pct: float = 70.0,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Impute missing values using trained models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with missing values to impute
    models : dict
        Dictionary of trained models from train_imputation_models()
    min_data_pct : float
        Minimum percentage of non-NaN values required per row for imputation
    inplace : bool
        If True, modify df directly; otherwise return copy
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with imputed values
    """
    
    if not inplace:
        df = df.copy()
    
    n_cols = len(df.columns)
    min_cols_required = int(n_cols * min_data_pct / 100)
    
    imputation_stats = {}
    
    for target_col, model_info in models.items():
        if target_col not in df.columns:
            continue
        
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        
        # Find rows with missing target but sufficient feature data
        row_data_count = df[feature_cols].notna().sum(axis=1)
        impute_mask = (df[target_col].isna()) & (row_data_count >= min_cols_required)
        
        n_to_impute = impute_mask.sum()
        if n_to_impute == 0:
            continue
        
        # Predict missing values
        X_impute = df.loc[impute_mask, feature_cols]
        predictions = model.predict(X_impute)
        
        # Fill missing values
        df.loc[impute_mask, target_col] = predictions
        
        imputation_stats[target_col] = n_to_impute
        print(f"Imputed {n_to_impute} values for {target_col}")
    
    total_imputed = sum(imputation_stats.values())
    print(f"\nTotal imputed values: {total_imputed}")
    
    return df


# # Example usage
# if __name__ == "__main__":
#     # Load your space weather data
#     # df = pd.read_csv('omni_data.csv')
    
#     # Columns to exclude (e.g., timestamp, indices we're predicting)
#     exclude = ['Epoch', 'DST', 'Year', 'DOY']
    
#     # Train imputation models
#     print("Training imputation models...")
#     models = train_imputation_models(
#         df,
#         min_data_pct=70.0,
#         exclude_cols=exclude
#     )
    
#     # Save models if needed
#     # import pickle
#     # with open('imputation_models.pkl', 'wb') as f:
#     #     pickle.dump(models, f)
    
#     # Impute missing values
#     print("\nImputing missing values...")
#     df_imputed = impute_missing_values(
#         df,
#         models,
#         min_data_pct=70.0,
#         inplace=False
#     )
    
#     # Compare before/after
#     print("\nNaN counts before imputation:")
#     print(df.isna().sum())
#     print("\nNaN counts after imputation:")
#     print(df_imputed.isna().sum())