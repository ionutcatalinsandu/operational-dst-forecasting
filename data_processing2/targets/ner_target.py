import pandas as pd
import numpy as np
from typing import Tuple, Dict


def add_storm_boundary_target(df: pd.DataFrame, 
                              source_column: str = 'DST',
                              target_column: str = 'storm_boundary',
                              threshold: float = -50) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Add binary labels for storm detection.
    
    Creates sequence labels identifying storm conditions:
    - O (Outside): Not in storm conditions (label=0)
    - I-STORM (Inside): Storm conditions (label=1)
    
    This simplified version treats storm detection as pure binary classification,
    removing the B-STORM (beginning) distinction. All storm periods are labeled
    uniformly, focusing the model on detecting storm vs. non-storm conditions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with DST index
    source_column : str
        Name of column containing DST values
    target_column : str
        Name for new boundary label column
    threshold : float
        DST threshold defining storm conditions (default: -50 nT)
        
    Returns
    -------
    df : pd.DataFrame
        Original dataframe with new boundary label column added
    label_map : Dict[int, str]
        Mapping from integer labels to string descriptions
        
    Examples
    --------
    DST:    [10,  0, -50, -70, -60, -30,  0]
    Labels: [ 0,  0,   1,   1,   1,   0,  0]
            [O,  O,  I,   I,   I,   O,  O]
    """
    dst_values = df[source_column].values
    
    # Simple binary classification: storm (1) vs non-storm (0)
    labels = (dst_values <= threshold).astype(np.int32)
    
    df[target_column] = labels
    
    label_map = {
        0: 'O',
        1: 'I-STORM'
    }
    
    return df, label_map