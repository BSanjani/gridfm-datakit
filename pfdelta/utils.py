# pfdelta/utils.py

def process_droop_results(result):
    """
    Extract and format droop-specific results
    
    Args:
        result: Raw result dictionary from Julia solver
    
    Returns:
        result: Enhanced result dictionary with formatted output
    """
    # Get status - handle both string and Julia enum types
    status = result.get('termination_status')
    status_str = str(status)  # Convert to string for comparison
    
    # Check for successful convergence
    success_statuses = ['LOCALLY_SOLVED', 'ALMOST_LOCALLY_SOLVED']
    is_success = any(s in status_str for s in success_statuses)
    
    if is_success:
        solution = result.get('solution', {})
        
        if 'frequency_deviation' in solution:
            freq_dev = solution['frequency_deviation']
            sys_freq = solution['system_frequency']
            
            # Add human-readable summary
            result['droop_summary'] = {
                'frequency_deviation_pu': freq_dev,
                'system_frequency_pu': sys_freq,
                'frequency_deviation_hz': freq_dev * 60.0,
                'system_frequency_hz': sys_freq * 60.0
            }
            
            print("\n" + "=" * 60)
            print("✓ DROOP POWER FLOW CONVERGED SUCCESSFULLY")
            print("=" * 60)
            print(f"System Frequency: {sys_freq:.6f} pu ({sys_freq*60:.4f} Hz)")
            print(f"Frequency Deviation: {freq_dev:.6f} pu ({freq_dev*60:.4f} Hz)")
            print(f"Frequency Deviation: {freq_dev*60*1000:.2f} mHz")
            print("=" * 60)
        else:
            print("✓ Power flow converged but frequency data not available")
    else:
        print(f"\n✗ Power flow did not converge")
        print(f"   Status: {status_str}")
    
    return result