import pandas as pd
from typing import List, Dict
from sklearn.base import BaseEstimator

from data_processing2.constants import *
import data_processing2.features as features

def merge_fims_data(current: pd.DataFrame, fism_data: pd.DataFrame, 
               fism_columns_to_keep: list = None) -> pd.DataFrame:
    """
    Merge current data with FISM data, keeping only specified FISM columns
    
    Parameters:
    - current: Your main dataset (DST data)
    - fism_data: FISM dataset  
    - fism_columns_to_keep: List of FISM columns to keep (if None, keeps all except FULL_DATE_COLUMN)
    """
    
    # If no columns specified, keep all (except the ones we always drop)
    if fism_columns_to_keep is None:
        columns_to_drop = [FULL_DATE_COLUMN]
        fism_subset = fism_data.drop(columns_to_drop, axis=1)
    else:
        # Keep only specified columns + the date column needed for merging
        columns_to_keep = fism_columns_to_keep + [FULL_DATE_DAY_COLUMN]
        fism_subset = fism_data[columns_to_keep]
    
    # Merge
    current_merged = current.merge(
        fism_subset,
        on=FULL_DATE_DAY_COLUMN, 
        how='left'
    )
    
    return current_merged


def add_fism_to_current_data(
        current_data: pd.DataFrame, 
        fism_data: pd.DataFrame, 
        fism_columns_to_keep: List[str],
        fism_transforms: Dict[str, List[BaseEstimator]] 
    )->pd.DataFrame:

    current_data[FULL_DATE_DAY_COLUMN] = current_data[FULL_DATE_COLUMN].dt.date
    current_data = merge_fims_data(current_data, fism_data, fism_columns_to_keep=fism_columns_to_keep)
    current_data, fism_fit_transforms = features.compute_fims_features(current_data, column_transforms=fism_transforms, fit_transform=True)

    for col in fism_columns_to_keep:
        assert len(current_data[current_data[col].isna()]) == 0
    return current_data
