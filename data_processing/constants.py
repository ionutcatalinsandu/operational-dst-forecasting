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
]

COLUMNS_WITH_THRESHOLD = [
    ("Proton flux (>10 Mev)", 9999),
    ("Alpha/Prot. ratio", 9),
    ("Scalar B, nT", 999),
    ("Vector B Magnitude,nT", 999),
    ("Lat. Angle of B (GSE)", 999),
    ("Long. Angle of B (GSE)", 999),
    ("BX, nT (GSE, GSM)", 999),
    ("BY, nT (GSE)", 999),
    ("BZ, nT (GSE)", 999),
    ("BY, nT (GSM)", 999),
    ("BZ, nT (GSM)", 999),
    ("SW Plasma Temperature, K", 9999999),
    ("SW Proton Density, N/cm^3", 999),
    ("SW Plasma Speed, km/s", 9999),
    ("SW Plasma flow long. angle", 999),
    ("SW Plasma flow lat. angle", 999),
    ("Flow pressure", 99.99),
    ("E elecrtric field", 999),
    ("Plasma Beta", 999),
    ("Alfen mach number", 999),
    ("Magnetosonic Much num.", 99),
    ("Quasy-Invariant", 9.9999),
    ("AE-index, nT", 9999)
]

DST_COLUMN = "Dst-index, nT"
EPSILON_COLUMN = "epsilon24 (W)"
ENOU_24_COLUMN = "enou24 (W)"
FULL_DATE_COLUMN = "full_date"

CANDIDATE_COLUMNS = [
    # "YEAR",
    # "DOY",
    # "Hour",
    "Vector B Magnitude,nT",
    "Lat. Angle of B (GSE)",
    "Long. Angle of B (GSE)",
    "BX, nT (GSE, GSM)",
    "BY, nT (GSM)",
    "BZ, nT (GSM)",
    "Scalar B, nT", 
    "SW Plasma Speed, km/s", 
    "SW Proton Density, N/cm^3",
    "SW Plasma Temperature, K",
    "SW Plasma flow long. angle",
    "SW Plasma flow lat. angle",
    "Plasma Beta",
    "Flow pressure",
    "E elecrtric field",
    "Alfen mach number",
    "Magnetosonic Much num.",
    # "Lyman_alpha",
    # "R (Sunspot No.)",
    FULL_DATE_COLUMN,
    DST_COLUMN,
    "Alpha/Prot. ratio",
]

ROBUST_ONLY_COLUMNS = [
    "BX, nT (GSE, GSM)",
    "BY, nT (GSM)", 
    "BZ, nT (GSM)",
]

LOG_ROBUST_COLUMNS = [
    # Magnetic field magnitudes
    "Vector B Magnitude,nT",
    "Scalar B, nT",
    
    # Solar wind bulk properties  
    "SW Plasma Speed, km/s",
    "SW Proton Density, N/cm^3", 
    "SW Plasma Temperature, K",
    
    # Pressure and field ratios
    "Plasma Beta",
    "Flow pressure", 
    "E elecrtric field",
    
    # Mach numbers (ratios)
    "Alfen mach number",
    "Magnetosonic Much num.",
    
    # Composition ratio
    "Alpha/Prot. ratio",
]

ANGULAR_TRIGONOMETRIC = [
    "Lat. Angle of B (GSE)",      # Magnetic elevation  
    "Long. Angle of B (GSE)",     # Magnetic azimuth
]

ANGULAR_ROBUST_ONLY = [
    "SW Plasma flow long. angle",  # Flow deviation (small)
    "SW Plasma flow lat. angle",   # Flow deviation (small)  
]

ANGULAR_COLUMNS = [
    # Magnetic field direction (GSE coordinates)
    "Lat. Angle of B (GSE)",      # Elevation: -90° to +90°
    "Long. Angle of B (GSE)",     # Azimuth: 0° to 360° (or ±180°)
    
    # Solar wind flow direction (small deviations from radial)
    "SW Plasma flow long. angle",  # Azimuthal deviation: typically ±20°
    "SW Plasma flow lat. angle",   # Elevation deviation: typically ±10°
]