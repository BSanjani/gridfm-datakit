"""
Droop Control Extension for GridFM-DataKit
Adds frequency and voltage droop response to generator data
"""

import numpy as np
import pandas as pd

def add_droop_parameters(gen_data, droop_config):
    """
    Add droop control parameters to generator data
    
    Parameters:
    -----------
    gen_data : DataFrame
        Generator data from GridFM
    droop_config : dict
        Droop configuration from user_config2.yaml
        
    Returns:
    --------
    gen_data : DataFrame
        Generator data with added droop parameters
    """
    
    if not droop_config.get('enable', False):
        print("Droop control is disabled")
        return gen_data
    
    n_gens = len(gen_data)
    
    # Extract droop ranges
    R_p_min, R_p_max = droop_config['R_p_range']
    R_q_min, R_q_max = droop_config['R_q_range']
    
    # Generate droop coefficients
    if droop_config.get('randomize_droop', True):
        # Random droop for each generator
        gen_data['R_p'] = np.random.uniform(R_p_min, R_p_max, n_gens)
        gen_data['R_q'] = np.random.uniform(R_q_min, R_q_max, n_gens)
    else:
        # Same droop for all generators (use midpoint)
        gen_data['R_p'] = (R_p_min + R_p_max) / 2
        gen_data['R_q'] = (R_q_min + R_q_max) / 2
    
    # Add nominal frequency
    gen_data['f_nominal'] = droop_config['f_nominal']
    
    # Add voltage setpoint (assume 1.0 p.u. initially)
    gen_data['v_setpoint'] = 1.0
    
    print(f"Added droop parameters to {n_gens} generators")
    print(f"  R_p range: {gen_data['R_p'].min():.4f} - {gen_data['R_p'].max():.4f}")
    print(f"  R_q range: {gen_data['R_q'].min():.4f} - {gen_data['R_q'].max():.4f}")
    
    return gen_data


def calculate_droop_response(gen_data, bus_data, droop_config):
    """
    Calculate generator response with droop control
    
    Droop equations:
    - P_droop = P_setpoint - (1/R_p) * (f - f_nominal) / f_nominal
    - Q_droop = Q_setpoint - (1/R_q) * (V - V_setpoint)
    
    Parameters:
    -----------
    gen_data : DataFrame
        Generator data with droop parameters
    bus_data : DataFrame
        Bus data with voltage and frequency information
    droop_config : dict
        Droop configuration
        
    Returns:
    --------
    gen_data : DataFrame
        Generator data with droop-adjusted power outputs
    """
    
    if not droop_config.get('enable', False):
        return gen_data
    
    # Make a copy to avoid modifying original
    gen_droop = gen_data.copy()
    
    # Assume system frequency (for now, use nominal)
    # In a real implementation, this would come from the power flow solution
    f_system = droop_config['f_nominal']  # Hz
    
    # For each generator, adjust based on local bus voltage
    # Merge with bus data to get voltages
    gen_droop = gen_droop.merge(
        bus_data[['scenario', 'bus', 'Vm']],
        left_on=['scenario', 'bus'],
        right_on=['scenario', 'bus'],
        how='left'
    )
    
    # Store original setpoints
    gen_droop['p_setpoint'] = gen_droop['p_mw']
    gen_droop['q_setpoint'] = gen_droop['q_mvar']
    
    # Calculate droop response
    droop_type = droop_config.get('type', 'frequency_and_voltage')
    
    if droop_type in ['frequency', 'frequency_and_voltage']:
        # Frequency droop: ΔP = -(1/R_p) * Δf/f_nom
        # For now, assume small frequency deviation (e.g., -0.1 Hz)
        delta_f = -0.1  # Hz (example deviation)
        gen_droop['delta_p_droop'] = -(1/gen_droop['R_p']) * (delta_f / f_system) * gen_droop['p_setpoint']
        gen_droop['p_mw_droop'] = gen_droop['p_setpoint'] + gen_droop['delta_p_droop']
        
        # Clip to limits
        gen_droop['p_mw_droop'] = gen_droop['p_mw_droop'].clip(
            gen_droop['min_p_mw'], 
            gen_droop['max_p_mw']
        )
    else:
        gen_droop['p_mw_droop'] = gen_droop['p_mw']
        gen_droop['delta_p_droop'] = 0
    
    if droop_type in ['voltage', 'frequency_and_voltage']:
        # Voltage droop: ΔQ = -(1/R_q) * ΔV
        delta_v = gen_droop['Vm'] - gen_droop['v_setpoint']
        gen_droop['delta_q_droop'] = -(1/gen_droop['R_q']) * delta_v * 100  # MVAr
        gen_droop['q_mvar_droop'] = gen_droop['q_setpoint'] + gen_droop['delta_q_droop']
        
        # Clip to limits
        gen_droop['q_mvar_droop'] = gen_droop['q_mvar_droop'].clip(
            gen_droop['min_q_mvar'],
            gen_droop['max_q_mvar']
        )
    else:
        gen_droop['q_mvar_droop'] = gen_droop['q_mvar']
        gen_droop['delta_q_droop'] = 0
    
    print(f"Applied droop control ({droop_type})")
    print(f"  Average P adjustment: {gen_droop['delta_p_droop'].mean():.2f} MW")
    print(f"  Average Q adjustment: {gen_droop['delta_q_droop'].mean():.2f} MVAr")
    
    return gen_droop


def save_droop_data(gen_data, output_dir):
    """Save generator data with droop parameters"""
    
    droop_cols = ['scenario', 'idx', 'bus', 'p_setpoint', 'q_setpoint', 
                  'p_mw_droop', 'q_mvar_droop', 'R_p', 'R_q', 
                  'delta_p_droop', 'delta_q_droop', 'f_nominal', 'v_setpoint']
    
    # Select only columns that exist
    available_cols = [col for col in droop_cols if col in gen_data.columns]
    
    droop_data = gen_data[available_cols]
    
    output_file = output_dir / 'gen_data_with_droop.parquet'
    droop_data.to_parquet(output_file)
    
    print(f"Saved droop-controlled generator data to: {output_file}")
    
    return output_file