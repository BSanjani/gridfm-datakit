from juliacall import Main as jl
import os

# Add these two imports
from pfdelta.validation import validate_droop_config
from pfdelta.utils import process_droop_results

# Rest of your existing code...
import os

# Load Julia solver once at module level
_julia_loaded = False

def _ensure_julia_loaded():
    """Load Julia droop solver if not already loaded"""
    global _julia_loaded
    if not _julia_loaded:
        julia_file = os.path.join(os.path.dirname(__file__), 'droop_solver.jl')
        jl.include(julia_file)
        _julia_loaded = True

def solve_ac_pf_droop(network_data, droop_params):
    """
    Solve AC power flow with droop control
    
    Args:
        network_data: Power network dictionary
        droop_params: Dict with omega_0, V_0, droop_buses, mp, mq
    
    Returns:
        result: Same format as regular power flow
    """
    _ensure_julia_loaded()
    result = jl.solve_ac_pf_droop(network_data, droop_params)
    return result


def solve_power_flow(network_data, config):
    """
    Main power flow solver with optional droop control
    
    Args:
        network_data: Power network dictionary
        config: Configuration dictionary from YAML
    
    Returns:
        result: Power flow solution
    """
    from pfdelta.validation import validate_droop_config
    from pfdelta.utils import process_droop_results
    
    # Check if droop is enabled
    if config.get('droop_control', {}).get('enabled', False):
        # Validate configuration
        validate_droop_config(network_data, config['droop_control'])
        
        # Extract droop parameters
        droop_params = {
            'omega_0': config['droop_control']['omega_0'],
            'V_0': config['droop_control']['V_0'],
            'droop_buses': config['droop_control']['droop_buses'],
            'mp': config['droop_control']['mp'],
            'mq': config['droop_control']['mq'],
            'ref_bus': config['droop_control'].get('ref_bus', None)
        }
        
        # Solve with droop
        result = solve_ac_pf_droop(network_data, droop_params)
        
        # Process droop-specific results
        result = process_droop_results(result)
    else:
        # Call your existing standard power flow solver
        # Replace this with whatever your existing solve function is called
        result = solve_standard_ac_pf(network_data)
    
    return result





def solve_standard_ac_pf(network_data):
    """
    Solve standard AC power flow (your existing implementation)
    """
    # Your existing power flow code
    pass

def solve_power_flow(network_data, config):
    """
    Main power flow solver with optional droop control
    
    Args:
        network_data: Power network dictionary
        config: Configuration dictionary from YAML
    
    Returns:
        result: Power flow solution
    """
    # Check if droop is enabled
    if config.get('droop_control', {}).get('enabled', False):
        # Validate configuration
        validate_droop_config(network_data, config['droop_control'])
        
        # Extract droop parameters
        droop_params = {
            'omega_0': config['droop_control']['omega_0'],
            'V_0': config['droop_control']['V_0'],
            'droop_buses': config['droop_control']['droop_buses'],
            'mp': config['droop_control']['mp'],
            'mq': config['droop_control']['mq'],
            'ref_bus': config['droop_control'].get('ref_bus', None)
        }
        
        # Solve with droop
        result = solve_ac_pf_droop(network_data, droop_params)
        
        # Process droop-specific results
        result = process_droop_results(result)
    else:
        # Call standard power flow solver
        result = solve_standard_ac_pf(network_data)
    
    return result