import pandas as pd
import numpy as np

def add_storm_duration_target(df: pd.DataFrame, source_column: str='DST', 
                              target_column: str='storm_duration_hours',
                              threshold: str=-50):
    """
    Add storm duration target: hours of upcoming storm conditions.
    
    O(n) implementation using backward pass with consecutive counting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with DST index
    source_column : str
        Name of column containing DST values
    target_column : str
        Name for new duration column
    threshold : float
        DST threshold defining storm conditions (default: -50 nT)
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with new duration column added
        
    Examples
    --------
    DST:    [10, 0, -50, -70, -60, -30, 0, 10]
    Target: [ 0, 3,   2,   1,   0,   0, 0,  0]
    """
    dst_values = df[source_column].values
    n = len(dst_values)
    duration = np.zeros(n, dtype=np.int32)
    
    # Last position always 0 (no future)
    duration[n - 1] = 0
    
    # Scan backwards: count consecutive storm hours ahead
    for i in range(n - 2, -1, -1):
        if dst_values[i + 1] <= threshold:
            # Next hour is in storm: add 1 to its remaining duration
            duration[i] = duration[i + 1] + 1
        else:
            # Next hour not in storm: no storm ahead
            duration[i] = 0
    
    df[target_column] = duration
    return df