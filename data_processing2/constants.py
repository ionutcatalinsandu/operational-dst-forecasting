
DST_COLUMN = "Dst-index, nT"
AE_COLUMN = "AE-index, nT"
DST_COLUMN_SCORE = "Dst-index, nT_score"
EPSILON_COLUMN = "epsilon24 (W)"
ENOU_24_COLUMN = "enou24 (W)"
FULL_DATE_COLUMN = "full_date"
FULL_DATE_DAY_COLUMN = "full_date_day"

# Column name constants
YEAR_COL = "YEAR"
DOY_COL = "DOY"
HOUR_COL = "Hour"
VECTOR_B_MAG_COL = "Vector B Magnitude,nT"
B_LAT_ANGLE_COL = "Lat. Angle of B (GSE)"
B_LONG_ANGLE_COL = "Long. Angle of B (GSE)"
BX_COL = "BX, nT (GSE, GSM)"
BY_COL = "BY, nT (GSM)"
BZ_COL = "BZ, nT (GSM)"
SCALAR_B_COL = "Scalar B, nT"
SW_SPEED_COL = "SW Plasma Speed, km/s"
SW_DENSITY_COL = "SW Proton Density, N/cm^3"
SW_TEMP_COL = "SW Plasma Temperature, K"
SW_FLOW_LONG_ANGLE_COL = "SW Plasma flow long. angle"
SW_FLOW_LAT_ANGLE_COL = "SW Plasma flow lat. angle"
PLASMA_BETA_COL = "Plasma Beta"
FLOW_PRESSURE_COL = "Flow pressure"
E_FIELD_COL = "E elecrtric field"
ALFVEN_MACH_COL = "Alfen mach number"
MAGNETOSONIC_MACH_COL = "Magnetosonic Much num."
LYMAN_ALPHA_COL = "Lyman_alpha"
SUNSPOT_COL = "R (Sunspot No.)"
ALPHA_PROTON_RATIO_COL = "Alpha/Prot. ratio"

ENERGY_LOADING_STATE_COL = "energy_loading_state"
MAGNETOTAIL_LOADING_STATE_COL = "magnetotail_loading_state"
COUPLING_EFFICIENCY_STATE = "coupling_efficiency_state"
RECOVERY_STATE_COL = "recovery_state"

# new magnetospheric dynamics
SUBSTORM_CYCLE_STATE_COL = 'substorm_cycle_state'
RECONNECTION_EFFICIENCY_COL = 'reconnection_efficiency_state'
MAGNETOSPHERIC_IMPEDANCE_COL = 'magnetospheric_impedance'
CROSS_SCALE_COUPLING_COL = 'cross_scale_coupling'
COHERENT_DRIVING_COL = 'coherent_driving_efficiency'
# end

# cycle
SOLAR_CYCLE = "solar_cycle"
SOLAR_PHASE = "solar_phase"

STORM_THRESHOLD = -50

# targets
UPCOMING_STORM = "upcoming_storm"
STORM_BOUNDARY = "storm_boundary"

# Assuming these are defined elsewhere
# FULL_DATE_COLUMN = "your_date_column"
# DST_COLUMN = "your_dst_column"

CANDIDATE_COLUMNS = [
    # YEAR_COL,
    # DOY_COL,
    # HOUR_COL,
    VECTOR_B_MAG_COL, # 
    B_LAT_ANGLE_COL, # 
    B_LONG_ANGLE_COL, # 
    BX_COL, # 
    BY_COL,
    BZ_COL,
    SCALAR_B_COL, 
    SW_SPEED_COL, 
    SW_DENSITY_COL,
    SW_TEMP_COL,
    # SW_FLOW_LONG_ANGLE_COL,
    # SW_FLOW_LAT_ANGLE_COL,
    PLASMA_BETA_COL,
    FLOW_PRESSURE_COL,
    E_FIELD_COL,
    # ALFVEN_MACH_COL,
    MAGNETOSONIC_MACH_COL, # 
    # LYMAN_ALPHA_COL,
    # SUNSPOT_COL,
    FULL_DATE_COLUMN,
    DST_COLUMN,
    AE_COLUMN,
    ALPHA_PROTON_RATIO_COL,
    SOLAR_CYCLE,
    SOLAR_PHASE,
]


BAD_COLUMNS = [
    "Bartels rotation number",
    "Bartels rotation number",
    "ID for SW Plasma spacecraft",
    "ID for IMF spacecraft",
    "RMS_magnitude, nT",
    "RMS_field_vector, nT",
    "RMS_BX_GSE, nT",
    "RMS_BY_GSE, nT",
    "RMS_BZ_GSE, nT",
    "sigma-T,K",
    "sigma-n, N/cm^3)",
    "sigma-V, km/s",               
    "sigma-phi V, degrees",    
    "sigma-theta V, degrees",      
    "sigma-ratio",
    "Kp index",
    "Flux FLAG",
    "ap_index, nT",     
    "f10.7_index",
    # "AE-index, nT",
    "AL-index, nT",
    "AU-index, nT",
    "pc-index",
    "Proton flux (>1 Mev)",
    "Proton flux (>2 Mev)",
    "Proton flux (>4 Mev)",
    "Proton flux (>30 Mev)",
    "Proton flux (>60 Mev)",
    "# of points in IMF averages",
    "# of points in Plasma averag.",
]