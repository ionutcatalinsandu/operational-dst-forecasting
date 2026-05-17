def create_interaction_constraints(feature_names, grouping_type="physics"):
    """
    Create interaction constraints based on different grouping strategies.
    
    Parameters:
    - feature_names: list of all feature column names
    - grouping_type: str, one of ["physics", "engineering", "hybrid", "parameter_only", "magnetospheric"]
    
    Returns:
    - str: interaction constraints string for XGBoost, or None
    """
    
    if grouping_type == "physics":
        return _create_physics_constraints(feature_names)
    elif grouping_type == "engineering":
        return _create_engineering_constraints(feature_names)
    elif grouping_type == "hybrid":
        return _create_hybrid_constraints(feature_names)
    elif grouping_type == "parameter_only":
        return _create_parameter_only_constraints(feature_names)
    elif grouping_type == "magnetospheric":
        return _create_magnetospheric_constraints(feature_names)
    else:
        return None


def _create_physics_constraints(feature_names):
    """Group features by underlying physical parameters, including magnetospheric state."""
    
    # Define physics-based parameter groups
    physics_groups = {
        'magnetic_field': [
            "BZ, nT (GSM)", "Scalar B, nT", "BX, nT (GSE, GSM)", 
            "BY, nT (GSM)", "Vector B Magnitude,nT"
        ],
        'plasma_bulk': [
            "SW Plasma Speed, km/s", "SW Proton Density, N/cm^3", 
            "SW Plasma Temperature, K", "Plasma Beta"
        ],
        'electromagnetic': [
            "E elecrtric field", "Flow pressure"
        ],
        'composition': [
            "Alpha/Prot. ratio"
        ],
        'flow_geometry': [
            "SW Plasma flow long. angle", "SW Plasma flow lat. angle"
        ],
        'magnetic_geometry': [
            "Lat. Angle of B (GSE)", "Long. Angle of B (GSE)"
        ],
        'mach_numbers': [
            "Alfen mach number", "Magnetosonic Much num."
        ],
        'solar_activity': [
            "Lyman_alpha", "R (Sunspot No.)"
        ],
        'temporal': [
            "YEAR", "DOY", "Hour"
        ],
        # NEW: Magnetospheric state features
        'magnetospheric_state': [
            "energy_loading_state", "beta_regime_state", "magnetotail_loading_state",
            "coupling_efficiency_state", "circulation_state", "temporal_coherence_state",
            "recovery_state"
        ]
    }
    
    return _build_constraints_from_groups(feature_names, physics_groups)


def _create_magnetospheric_constraints(feature_names):
    """
    NEW: Group magnetospheric features by their physical processes.
    This creates fine-grained groupings within magnetospheric state features.
    """
    
    magnetospheric_groups = {
        'energy_budget': [
            "energy_loading_state", "coupling_efficiency_state"
        ],
        'plasma_regime': [
            "beta_regime_state", "circulation_state"
        ],
        'temporal_dynamics': [
            "magnetotail_loading_state", "temporal_coherence_state"
        ],
        'historical_context': [
            "recovery_state"
        ],
        # Allow magnetospheric features to interact with core physics
        'magnetic_coupling': [
            "BZ, nT (GSM)", "Scalar B, nT", "BY, nT (GSM)",
            "energy_loading_state", "coupling_efficiency_state", "magnetotail_loading_state"
        ],
        'plasma_coupling': [
            "SW Plasma Speed, km/s", "SW Proton Density, N/cm^3", 
            "SW Plasma Temperature, K", "Plasma Beta", "Flow pressure",
            "beta_regime_state", "circulation_state", "temporal_coherence_state"
        ]
    }
    
    return _build_constraints_from_groups(feature_names, magnetospheric_groups)


def _create_engineering_constraints(feature_names):
    """Group features by engineering method/transformation, including magnetospheric features."""
    
    engineering_groups = {
        'topological': ['_jump_mag', '_path_eff', '_turn_var'],
        'signal_processing': ['_max_jump', '_jump_conc', '_trend_str'],
        'convolution': ['conv_'],
        'lag_history': ['_lag_'],
        'rolling_stats': ['_std_', '_rolling_median_', '_cdf_wass_trend_'],
        'magnetospheric_derived': [
            'energy_loading_state', 'beta_regime_state', 'magnetotail_loading_state',
            'coupling_efficiency_state', 'circulation_state', 'temporal_coherence_state',
            'recovery_state'
        ],
        'raw_parameters': []  # Will be filled with features that don't match any suffix
    }
    
    # Build groups based on suffixes and exact matches
    constraints = []
    
    for group_name, patterns in engineering_groups.items():
        if group_name == 'raw_parameters':
            continue  # Handle separately
            
        group_indices = []
        for i, feature_name in enumerate(feature_names):
            # For magnetospheric_derived, check exact matches
            if group_name == 'magnetospheric_derived':
                if feature_name in patterns:
                    group_indices.append(i)
            else:
                # For other groups, check if any pattern is in feature name
                if any(pattern in feature_name for pattern in patterns):
                    group_indices.append(i)
        
        if len(group_indices) > 1:
            constraints.append(group_indices)
    
    # Add raw parameters that can interact with any engineered features
    raw_indices = []
    all_engineering_patterns = ['_jump_mag', '_path_eff', '_turn_var', '_max_jump', '_jump_conc', 
                               '_trend_str', 'conv_', '_lag_', '_std_', '_rolling_median_', '_cdf_wass_trend_']
    magnetospheric_features = ['energy_loading_state', 'beta_regime_state', 'magnetotail_loading_state',
                              'coupling_efficiency_state', 'circulation_state', 'temporal_coherence_state',
                              'recovery_state']
    
    for i, feature_name in enumerate(feature_names):
        is_engineered = any(pattern in feature_name for pattern in all_engineering_patterns)
        is_magnetospheric = feature_name in magnetospheric_features
        
        if not (is_engineered or is_magnetospheric):
            raw_indices.append(i)
    
    # Allow raw parameters to interact with each engineering group
    for constraint in constraints:
        constraint.extend(raw_indices)
    
    return str(constraints) if constraints else None


def _create_hybrid_constraints(feature_names):
    """Hybrid: Major physics domains + magnetospheric state + engineering types within each domain."""
    
    # Three major physics domains
    magnetic_params = ["BZ, nT (GSM)", "Scalar B, nT", "BX, nT (GSE, GSM)", 
                      "BY, nT (GSM)", "Vector B Magnitude,nT", "E elecrtric field"]
    
    plasma_params = ["SW Plasma Speed, km/s", "SW Proton Density, N/cm^3", 
                    "SW Plasma Temperature, K", "Plasma Beta", "Flow pressure", 
                    "Alpha/Prot. ratio"]
    
    # NEW: Magnetospheric state parameters
    magnetospheric_params = ["energy_loading_state", "beta_regime_state", "magnetotail_loading_state",
                            "coupling_efficiency_state", "circulation_state", "temporal_coherence_state",
                            "recovery_state"]
    
    constraints = []
    
    # Magnetic domain: all magnetic-related features can interact
    magnetic_indices = []
    for i, feature_name in enumerate(feature_names):
        if any(param in feature_name for param in magnetic_params):
            magnetic_indices.append(i)
    
    if len(magnetic_indices) > 1:
        constraints.append(magnetic_indices)
    
    # Plasma domain: all plasma-related features can interact  
    plasma_indices = []
    for i, feature_name in enumerate(feature_names):
        if any(param in feature_name for param in plasma_params):
            plasma_indices.append(i)
    
    if len(plasma_indices) > 1:
        constraints.append(plasma_indices)
    
    # NEW: Magnetospheric domain: all magnetospheric state features can interact
    magnetospheric_indices = []
    for i, feature_name in enumerate(feature_names):
        if feature_name in magnetospheric_params:
            magnetospheric_indices.append(i)
    
    if len(magnetospheric_indices) > 1:
        constraints.append(magnetospheric_indices)
    
    # Cross-domain interactions: magnetospheric with core physics
    # Magnetospheric + Magnetic interactions
    mag_cross_indices = magnetic_indices + magnetospheric_indices
    if len(mag_cross_indices) > len(magnetic_indices) and len(mag_cross_indices) > len(magnetospheric_indices):
        constraints.append(mag_cross_indices)
    
    # Magnetospheric + Plasma interactions  
    plasma_cross_indices = plasma_indices + magnetospheric_indices
    if len(plasma_cross_indices) > len(plasma_indices) and len(plasma_cross_indices) > len(magnetospheric_indices):
        constraints.append(plasma_cross_indices)
    
    return str(constraints) if constraints else None


def _create_parameter_only_constraints(feature_names):
    """Group only by base parameter, ignoring all feature engineering, including magnetospheric."""
    
    base_params = [
        "BZ, nT (GSM)", "Scalar B, nT", "BX, nT (GSE, GSM)", "BY, nT (GSM)",
        "Vector B Magnitude,nT", "SW Plasma Speed, km/s", "SW Proton Density, N/cm^3",
        "SW Plasma Temperature, K", "Plasma Beta", "E elecrtric field", 
        "Flow pressure", "Alpha/Prot. ratio", "SW Plasma flow long. angle",
        "SW Plasma flow lat. angle", "Lat. Angle of B (GSE)", "Long. Angle of B (GSE)",
        "Alfen mach number", "Magnetosonic Much num.", "Lyman_alpha", "R (Sunspot No.)",
        "YEAR", "DOY", "Hour"
    ]
    
    # NEW: Add magnetospheric state features as separate base parameters
    magnetospheric_base_params = [
        "energy_loading_state", "beta_regime_state", "magnetotail_loading_state",
        "coupling_efficiency_state", "circulation_state", "temporal_coherence_state",
        "recovery_state"
    ]
    
    all_base_params = base_params + magnetospheric_base_params
    
    constraints = []
    
    for base_param in all_base_params:
        param_indices = []
        for i, feature_name in enumerate(feature_names):
            if base_param in feature_name:
                param_indices.append(i)
        
        if len(param_indices) > 1:
            constraints.append(param_indices)
    
    return str(constraints) if constraints else None


def _build_constraints_from_groups(feature_names, groups_dict):
    """Helper function to build constraints from parameter groups."""
    
    constraints = []
    
    for group_name, base_params in groups_dict.items():
        group_indices = []
        
        for base_param in base_params:
            for i, feature_name in enumerate(feature_names):
                if base_param in feature_name:
                    group_indices.append(i)
        
        # Only add constraint if group has multiple features
        if len(group_indices) > 1:
            constraints.append(group_indices)
    
    return str(constraints) if constraints else None


def get_all_constraint_options(feature_names):
    """
    Get all available constraint options for Optuna to choose from.
    
    Parameters:
    - feature_names: list of feature column names
    
    Returns:
    - list: all constraint options including None
    """
    
    grouping_types = ["physics", "engineering", "hybrid", "parameter_only", "magnetospheric"]
    options = [None]  # Always include no constraints
    
    for grouping_type in grouping_types:
        constraint = create_interaction_constraints(feature_names, grouping_type)
        if constraint and constraint not in options:
            options.append(constraint)
    
    return options


# Usage in Optuna objective
def add_interaction_constraints_to_optuna_param(trial, feature_names):
    """
    Add interaction constraints parameter to Optuna trial.
    
    Usage:
    param = {
        # ... other parameters ...
    }
    param.update(add_interaction_constraints_to_optuna_param(trial, feature_names))
    """
    
    constraint_options = get_all_constraint_options(feature_names)
    
    return {
        "interaction_constraints": trial.suggest_categorical(
            "interaction_constraints",
            constraint_options
        )
    }