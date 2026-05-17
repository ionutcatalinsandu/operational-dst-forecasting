import numpy as np
import pandas as pd
from numba import jit
from scipy.signal import find_peaks
from data_processing2.constants import *


def add_energy_loading_state(df: pd.DataFrame, tau_decay_hours: float = 8.5):
    """
    Computes magnetospheric energy loading state using Wang et al. (2014) epsilon 
    and exponential accumulation.
    
    Physics: 
    - E_in: Wang's improved Akasofu epsilon (Eq. 1)
    - E_load(t) = α·E_in(t) + (1-α)·E_load(t-1) (Eq. 2)
    - α = 1 - exp(-Δt/τ) ≈ 0.117 with Δt=1h, τ=8.5h
    
    Args:
        df: DataFrame with solar wind and IMF parameters
        tau_decay_hours: Characteristic decay timescale (default: 8.5 hours)
    
    Returns:
        DataFrame with 'energy_loading_state' column added
    """
    # Extract required parameters
    n_sw = df[SW_DENSITY_COL].values      # Proton density [N/cm³]
    v_sw = df[SW_SPEED_COL].values        # Solar wind speed [km/s]
    bx = df[BX_COL].values                # IMF Bx [nT]
    by = df[BY_COL].values                # IMF By [nT]
    bz = df[BZ_COL].values                # IMF Bz [nT]
    
    dt_hours = 1.0
    energy_state = compute_energy_loading_numba(
        n_sw, v_sw, bx, by, bz, dt_hours, tau_decay_hours
    )
    
    df_result = df.copy()
    df_result[ENERGY_LOADING_STATE_COL] = energy_state
    
    return df_result


@jit(nopython=True)
def compute_energy_loading_numba(n_sw, v_sw, bx, by, bz, dt_hours, tau_decay_hours):
    """
    Numba-accelerated energy loading computation with numerical stability.
    
    Wang epsilon: E_in = 3.78e7 · n^0.24 · V^1.47 · B_T^0.86 · [sin^2.70(θ/2) + 0.25]
    Accumulation: E_load(t) = α·E_in(t) + (1-α)·E_load(t-1)
    
    Numerical guards prevent edge cases:
    - Density floor: 1e-6 N/cm³ (prevents zero in fractional power)
    - Velocity floor: 1.0 km/s (physically impossible to be lower)
    - B_T floor: 1e-6 nT (prevents zero magnetic field)
    """
    n = len(v_sw)
    energy_loading = np.full(n, np.nan)
    
    # α = 1 - exp(-Δt/τ) controls exponential decay rate
    alpha = 1.0 - np.exp(-dt_hours / tau_decay_hours)
    
    for i in range(n):
        # Skip rows with any missing input parameters
        if (np.isnan(n_sw[i]) or np.isnan(v_sw[i]) or 
            np.isnan(bx[i]) or np.isnan(by[i]) or np.isnan(bz[i])):
            continue
        
        # Numerical guards: floor values to prevent invalid operations
        n_density = max(n_sw[i], 1e-6)  # N/cm³
        v_speed = max(v_sw[i], 1.0)     # km/s
        
        # B_T = sqrt(Bx² + By²) - transverse IMF component
        bt = np.sqrt(bx[i]**2 + by[i]**2)
        bt = max(bt, 1e-6)  # nT
        
        # Clock angle θ = atan2(By, Bz) - IMF orientation in GSM Y-Z plane
        theta = np.arctan2(by[i], bz[i])
        
        # Wang's epsilon formula (Eq. 1 from Wang et al. 2014)
        sin_term = np.sin(theta / 2.0)**2.70 + 0.25
        e_in = 3.78e7 * (n_density**0.24) * (v_speed**1.47) * (bt**0.86) * sin_term
        
        # Exponential accumulation (Eq. 2)
        if i == 0:
            # Cold start: first valid value initializes accumulation
            energy_loading[i] = alpha * e_in
        else:
            prev_energy = energy_loading[i-1]
            if np.isnan(prev_energy):
                # Gap restart: re-initialize after missing data
                energy_loading[i] = alpha * e_in
            else:
                # Standard exponential accumulation
                energy_loading[i] = alpha * e_in + (1.0 - alpha) * prev_energy
    
    return energy_loading


def add_magnetotail_loading_state(df: pd.DataFrame, tau_hours: float = 2.0, bz_threshold: float = 3.0):
    """
    Computes magnetotail magnetic flux loading state for substorm preconditioning.
    
    Physics:
    - φ(t): Flux loading rate (Eq. 1)
    - Φ_tail(t) = α·φ(t) + (1-α)·Φ_tail(t-1) (Eq. 2)
    - α = 1 - exp(-Δt/τ) ≈ 0.393 with τ=2h
    
    Args:
        df: DataFrame with solar wind and IMF parameters
        tau_hours: Accumulation timescale (default: 2 hours for substorm loading)
        bz_threshold: Weakly northward threshold in nT (default: 3 nT)
    
    Returns:
        DataFrame with 'magnetotail_loading_state' column added
    """
    bz = df[BZ_COL].values           # IMF Bz [nT]
    v_sw = df[SW_SPEED_COL].values   # Solar wind speed [km/s]
    rho = df[SW_DENSITY_COL].values  # Proton density [N/cm³]
    
    dt_hours = 1.0
    tail_loading = compute_tail_loading_numba(
        bz, v_sw, rho, dt_hours, tau_hours, bz_threshold
    )
    
    df_result = df.copy()
    df_result[MAGNETOTAIL_LOADING_STATE_COL] = tail_loading
    
    return df_result


@jit(nopython=True)
def compute_tail_loading_numba(bz, v_sw, rho, dt_hours, tau_hours, bz_threshold):
    """
    Numba-accelerated magnetotail loading computation with numerical stability.
    
    Flux loading rate: φ(t) = B_s(t) · v(t) · ρ(t) × 10^-4
    where B_s(t) = max(0, -B_z(t) + threshold)
    Accumulation: Φ_tail(t) = α·φ(t) + (1-α)·Φ_tail(t-1)
    
    Numerical guards prevent edge cases:
    - Velocity floor: 1.0 km/s (physically impossible to be lower)
    - Density floor: 1e-6 N/cm³ (prevents zero in multiplication)
    - B_s naturally bounded [0, ∞) by max() operation
    """
    n = len(bz)
    tail_loading = np.full(n, np.nan)
    
    # α = 1 - exp(-Δt/τ) controls exponential decay rate
    # Typical: α ≈ 0.393 for τ=2h, representing substorm timescale
    alpha = 1.0 - np.exp(-dt_hours / tau_hours)
    
    for i in range(n):
        # Skip rows with any missing input parameters
        if np.isnan(bz[i]) or np.isnan(v_sw[i]) or np.isnan(rho[i]):
            continue
        
        # Numerical guards: floor values to prevent invalid operations
        v_speed = max(v_sw[i], 1.0)      # km/s
        density = max(rho[i], 1e-6)      # N/cm³
        
        # B_s = max(0, -B_z + threshold) - southward component with threshold offset
        # Extends reconnection efficiency to weakly northward conditions
        bs = max(0.0, -bz[i] + bz_threshold)
        
        # Flux loading rate (Eq. 1)
        # Factor 10^-4 converts to appropriate flux units
        phi = bs * v_speed * density * 1e-4
        
        # Exponential accumulation (Eq. 2)
        if i == 0:
            # Cold start: first valid value initializes accumulation
            tail_loading[i] = alpha * phi
        else:
            prev_loading = tail_loading[i-1]
            if np.isnan(prev_loading):
                # Gap restart: re-initialize after missing data
                tail_loading[i] = alpha * phi
            else:
                # Standard exponential accumulation
                tail_loading[i] = alpha * phi + (1.0 - alpha) * prev_loading
    
    return tail_loading


def add_coupling_efficiency_state(df: pd.DataFrame, epsilon_stability: float = 1.0):
    """
    Computes magnetospheric coupling efficiency state with numerical stability.
    
    Physics:
    - η_couple(t) = (|D(t)| + A(t)) / (E_in(t) + ε_0)
    - Ratio of past magnetospheric activity to current solar wind energy input
    - Indicates saturation or memory effects
    
    Args:
        df: DataFrame with Dst, AE, and energy input
        epsilon_stability: Numerical stability constant (default: 1.0)
    
    Returns:
        DataFrame with 'coupling_efficiency_state' column added
    """
    # Extract required parameters
    dst = df[DST_COLUMN].values          # Dst index [nT]
    ae = df[AE_COLUMN].values            # AE index [nT]
    
    # Compute Wang epsilon for energy input
    n_sw = df[SW_DENSITY_COL].values
    v_sw = df[SW_SPEED_COL].values
    bx = df[BX_COL].values
    by = df[BY_COL].values
    bz = df[BZ_COL].values
    
    # Compute coupling efficiency
    coupling_efficiency = np.full(len(df), np.nan)
    
    for i in range(len(df)):
        # Skip rows with any missing input parameters
        if (np.isnan(dst[i]) or np.isnan(ae[i]) or 
            np.isnan(n_sw[i]) or np.isnan(v_sw[i]) or
            np.isnan(bx[i]) or np.isnan(by[i]) or np.isnan(bz[i])):
            continue
        
        # Numerical guards: floor values to prevent invalid operations
        n_density = max(n_sw[i], 1e-6)  # N/cm³
        v_speed = max(v_sw[i], 1.0)     # km/s
        
        # Wang epsilon (energy input rate)
        bt = np.sqrt(bx[i]**2 + by[i]**2)
        bt = max(bt, 1e-6)  # nT
        
        # Clock angle θ = atan2(By, Bz) - IMF orientation in GSM Y-Z plane
        theta = np.arctan2(by[i], bz[i])
        
        # Wang's epsilon formula (same as energy loading computation)
        sin_term = np.abs(np.sin(theta / 2.0))**2.70 + 0.25
        e_in = 3.78e7 * (n_density**0.24) * (v_speed**1.47) * (bt**0.86) * sin_term
        
        # Coupling efficiency ratio (Eq. 1)
        # Numerator: magnetospheric activity (|Dst| + AE)
        # Denominator: solar wind energy input + stability constant
        numerator = np.abs(dst[i]) + ae[i]
        denominator = e_in + epsilon_stability
        
        coupling_efficiency[i] = numerator / denominator
    
    df_result = df.copy()
    df_result[COUPLING_EFFICIENCY_STATE] = coupling_efficiency
    
    return df_result


def add_recovery_state(df: pd.DataFrame, lookback_hours: float = 24.0, 
                      storm_threshold: float = -50.0, 
                      min_prominence: float = 15.0,
                      min_duration_hours: float = 2):
    """
    Computes magnetospheric recovery state using peak detection on inverted DST.
    
    Physics:
    - Ω_recovery(t) = f_time(t) · (1 + r(t))
    - f_time: Normalized time since last storm (0-1)
    - r(t): Active recovery rate over 2-hour window
    
    Args:
        df: DataFrame with DST values
        lookback_hours: Maximum lookback period (T_max, default: 24h)
        storm_threshold: Peak threshold for storm identification (e.g., -50, -100, -150 nT)
        min_prominence: Minimum peak prominence (10-20 nT typical)
        min_duration_hours: Minimum storm duration (1-2 hours typical)
    
    Returns:
        Tuple of (DataFrame with recovery_state column added, column_name)
    """
    dst_values = df[DST_COLUMN].values
    
    # Identify storm end times using peak detection on inverted DST
    storm_end_indices = identify_storm_peaks(
        dst_values, storm_threshold, min_prominence, min_duration_hours
    )
    
    # Compute recovery state
    recovery_state = compute_recovery_state_numba(
        dst_values, storm_end_indices, lookback_hours
    )
    
    df_result = df.copy()
    column_name = f"{RECOVERY_STATE_COL}_lookback_{int(lookback_hours)}_thresh_{int(storm_threshold)}"
    df_result[column_name] = recovery_state
    
    return df_result, column_name


def identify_storm_peaks(dst_values: np.ndarray, threshold: float, 
                        prominence: float, min_duration_hours: float) -> np.ndarray:
    """
    Identify storm periods using peak detection on inverted DST time series.
    
    Args:
        dst_values: DST time series [nT]
        threshold: Storm threshold (e.g., -50 nT for moderate storms)
        prominence: Minimum peak prominence - ensures storm is distinct event (nT)
        min_duration_hours: Minimum storm duration - filters transient fluctuations (hours)
    
    Returns:
        Array of indices where storms end (peak times in inverted DST)
    
    Physical interpretation:
    - Inverted DST transforms depressions into peaks
    - Peak detection identifies distinct storm events
    - Prominence requirement ensures storm is significant vs. background
    """
    # Invert DST so storms (negative excursions) become peaks
    inverted_dst = -dst_values + 1
    
    # Peak height threshold in inverted coordinates
    # For threshold=-50 nT, we look for peaks > 50 in inverted series
    # Using strict > comparison (scipy's find_peaks uses >=)
    peak_height = -threshold
    
    # Minimum width ensures storm persists (not just transient spike)
    min_width_samples = int(min_duration_hours)  # Assumes 1-hour resolution
    
    peaks, _ = find_peaks(
        inverted_dst,
        height=peak_height,
        prominence=prominence,
        width=min_width_samples
    )
    
    return peaks


@jit(nopython=True)
def compute_recovery_state_numba(dst_values: np.ndarray, storm_end_indices: np.ndarray, 
                                lookback_hours: float) -> np.ndarray:
    """
    Numba-accelerated recovery state computation with numerical stability.
    
    Implements:
    - Ω_recovery(t) = f_time(t) · (1 + r(t))
    - f_time(t) = min(T_elapsed / 24, 1)  # Normalized by characteristic recovery time
    - r(t) = (1/2)[max(0, D(t)-D(t-1)) + max(0, D(t-1)-D(t-2))]  # Active recovery rate
    
    Physical interpretation:
    - f_time: Fraction of characteristic recovery complete (0 at storm peak, 1 after 24h)
    - r(t): Rate of DST increase (nT/hour) - positive during recovery
    - Product: Combines time-since-storm with active recovery dynamics
    
    Edge cases:
    - No storm found within lookback: NaN (honest uncertainty)
    - Missing DST data: r(t) = 0 (conservative, assumes no active recovery)
    - First 2 hours of dataset: r(t) = 0 (insufficient history)
    """
    n = len(dst_values)
    recovery_state = np.full(n, np.nan)
    
    for i in range(n):
        if np.isnan(dst_values[i]):
            continue
        
        # Find MOST RECENT storm end time within lookback window
        t_elapsed = -1.0  # Flag: no storm found yet
        
        for storm_idx in storm_end_indices:
            if storm_idx < i:  # Storm ended before current time
                time_diff = i - storm_idx  # Hours since storm ended
                if time_diff <= lookback_hours:
                    # Update to most recent storm (minimum time_diff)
                    if t_elapsed < 0 or time_diff < t_elapsed:
                        t_elapsed = time_diff
        
        # If no storm found within lookback window, skip this row
        if t_elapsed < 0:
            continue  # Leaves recovery_state[i] = NaN
        
        # Time factor (Eq. 2): normalized by 24-hour characteristic timescale
        # Saturates at 1.0 for full recovery (>24h since storm)
        f_time = min(t_elapsed / 24.0, 1.0)
        
        # Recovery rate (Eq. 4): active DST restoration over 2-hour window
        # Measures current rate of DST increase (positive = recovering)
        if i >= 2:
            if not (np.isnan(dst_values[i-1]) or np.isnan(dst_values[i-2])):
                # r(t) = (1/2)[max(0, D(t)-D(t-1)) + max(0, D(t-1)-D(t-2))]
                # Only positive changes count (DST increasing toward 0)
                rate_1 = max(0.0, dst_values[i] - dst_values[i-1])
                rate_2 = max(0.0, dst_values[i-1] - dst_values[i-2])
                r_t = 0.5 * (rate_1 + rate_2)
            else:
                # Missing data: assume no active recovery
                r_t = 0.0
        else:
            # Insufficient history for rate computation
            r_t = 0.0
        
        # Recovery state (Eq. 1)
        # Combined metric: time-weighted active recovery
        recovery_state[i] = f_time * (1.0 + r_t)
    
    return recovery_state