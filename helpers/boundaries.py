import numpy as np

def detect_storm_periods(dst_array, min_prominence=15, min_storm_depth=-30, min_duration=6):
    """
    Detect storm periods and return binary array
    
    Parameters:
    - dst_array: numpy array of DST values
    - min_prominence: minimum prominence for peak detection  
    - min_storm_depth: minimum DST value to consider as storm
    - min_duration: minimum storm duration in hours
    
    Returns:
    - binary_array: 0/1 array same length as input (1 = storm period)
    """
    from scipy.signal import find_peaks
    
    # Find local minima (invert signal)
    minima_indices, properties = find_peaks(
        -dst_array,
        prominence=min_prominence,
        distance=12  # min 12 hours between storm centers
    )
    
    # Filter minima by depth
    valid_minima = []
    for idx in minima_indices:
        if dst_array[idx] < min_storm_depth:
            valid_minima.append(idx)
    
    # Initialize binary array
    storm_binary = np.zeros(len(dst_array), dtype=int)
    
    # For each valid minimum, find storm boundaries
    for min_idx in valid_minima:
        onset_idx, recovery_idx = find_storm_boundaries(dst_array, min_idx, min_storm_depth)
        
        # Check minimum duration
        if recovery_idx - onset_idx >= min_duration:
            storm_binary[onset_idx:recovery_idx+1] = 1
    
    return storm_binary

def find_storm_boundaries(dst_array, min_idx, threshold):
    """Simple boundary detection around minimum"""
    # Go backwards from minimum to find onset
    onset_idx = min_idx
    for i in range(min_idx - 1, max(0, min_idx - 72), -1):
        if dst_array[i] > threshold:
            onset_idx = i + 1
            break
    
    # Go forwards from minimum to find recovery
    recovery_idx = min_idx  
    for i in range(min_idx + 1, min(len(dst_array), min_idx + 120)):
        if dst_array[i] > threshold:
            recovery_idx = i - 1
            break
    
    return onset_idx, recovery_idx