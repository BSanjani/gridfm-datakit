# pfdelta/validation.py

def validate_droop_config(network_data, droop_config):
    """
    Validate that droop configuration is compatible with network
    
    Args:
        network_data: Power network dictionary
        droop_config: Droop control configuration dict
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Get bus IDs from network - convert to both int and str for comparison
    network_bus_keys = set(network_data['bus'].keys())
    # Create a set that contains both string and int versions
    bus_ids = set()
    for key in network_bus_keys:
        bus_ids.add(key)
        try:
            # Try to add integer version if key is string
            bus_ids.add(int(key))
        except (ValueError, TypeError):
            pass
        try:
            # Try to add string version if key is int
            bus_ids.add(str(key))
        except (ValueError, TypeError):
            pass
    
    droop_buses = set(droop_config['droop_buses'])
    
    # Check all droop buses exist in network
    invalid_buses = droop_buses - bus_ids
    if invalid_buses:
        raise ValueError(
            f"Droop buses {invalid_buses} not found in network. "
            f"Available buses: {sorted(network_bus_keys)}"
        )
    
    # Check droop buses have generators
    gen_buses = set()
    for gen in network_data['gen'].values():
        gen_bus = gen['gen_bus']
        gen_buses.add(gen_bus)
        try:
            gen_buses.add(int(gen_bus))
        except (ValueError, TypeError):
            pass
        try:
            gen_buses.add(str(gen_bus))
        except (ValueError, TypeError):
            pass
    
    missing_gens = droop_buses - gen_buses
    if missing_gens:
        raise ValueError(
            f"Droop buses {missing_gens} have no generators attached. "
            f"Buses with generators: {sorted([g['gen_bus'] for g in network_data['gen'].values()])}"
        )
    
    # Validate droop coefficients are positive
    if droop_config['mp'] <= 0:
        raise ValueError(f"Active power droop coefficient mp must be positive, got {droop_config['mp']}")
    
    if droop_config['mq'] <= 0:
        raise ValueError(f"Reactive power droop coefficient mq must be positive, got {droop_config['mq']}")
    
    # Validate reference values
    if droop_config['omega_0'] <= 0:
        raise ValueError(f"Reference frequency omega_0 must be positive, got {droop_config['omega_0']}")
    
    if droop_config['V_0'] <= 0:
        raise ValueError(f"Reference voltage V_0 must be positive, got {droop_config['V_0']}")
    
    # Check reference bus if specified
    if 'ref_bus' in droop_config and droop_config['ref_bus'] is not None:
        ref_bus = droop_config['ref_bus']
        if ref_bus not in bus_ids:
            raise ValueError(f"Reference bus {ref_bus} not found in network")
    
    return True