# test_droop.py

import yaml
from pfdelta.solver import solve_power_flow

def test_droop():
    """Test droop control power flow"""
    
    # Load configuration
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a complete test network with all required PowerModels fields
    # Create a realistic balanced 3-bus test network
    network = {
        'bus': {
            '1': {'bus_type': 3, 'vmax': 1.1, 'vmin': 0.9, 'va': 0.0, 'vm': 1.0, 'base_kv': 230.0, 'zone': 1, 'index': 1},
            '2': {'bus_type': 2, 'vmax': 1.1, 'vmin': 0.9, 'va': 0.0, 'vm': 1.0, 'base_kv': 230.0, 'zone': 1, 'index': 2},
            '3': {'bus_type': 1, 'vmax': 1.1, 'vmin': 0.9, 'va': 0.0, 'vm': 1.0, 'base_kv': 230.0, 'zone': 1, 'index': 3}
        },
        'gen': {
            '1': {
                'index': 1,
                'gen_bus': 1,
                'pg': 1.5,      # Generator 1 produces 1.5 MW
                'qg': 0.3, 
                'pmax': 3.0, 
                'pmin': 0.0, 
                'qmax': 2.0, 
                'qmin': -2.0, 
                'gen_status': 1,
                'vg': 1.0,
                'mbase': 100.0,
                'pc1': 0.0, 'pc2': 0.0,
                'qc1min': 0.0, 'qc1max': 0.0,
                'qc2min': 0.0, 'qc2max': 0.0,
                'ramp_agc': 0.0, 'ramp_10': 0.0,
                'ramp_30': 0.0, 'ramp_q': 0.0, 'apf': 0.0
            },
            '2': {
                'index': 2,
                'gen_bus': 2,
                'pg': 0.8,      # Generator 2 produces 0.8 MW
                'qg': 0.2, 
                'pmax': 2.0, 
                'pmin': 0.0,
                'qmax': 1.5, 
                'qmin': -1.5, 
                'gen_status': 1,
                'vg': 1.0,
                'mbase': 100.0,
                'pc1': 0.0, 'pc2': 0.0,
                'qc1min': 0.0, 'qc1max': 0.0,
                'qc2min': 0.0, 'qc2max': 0.0,
                'ramp_agc': 0.0, 'ramp_10': 0.0,
                'ramp_30': 0.0, 'ramp_q': 0.0, 'apf': 0.0
            }
        },
        'branch': {
            '1': {
                'index': 1,
                'f_bus': 1,
                't_bus': 2,
                'br_r': 0.001,     # Low resistance
                'br_x': 0.01,      # Low reactance
                'br_status': 1,
                'rate_a': 5.0,
                'rate_b': 5.0,
                'rate_c': 5.0,
                'tap': 1.0,
                'shift': 0.0,
                'g_fr': 0.0,
                'g_to': 0.0,
                'b_fr': 0.0,
                'b_to': 0.0,
                'angmin': -60.0,
                'angmax': 60.0
            },
            '2': {
                'index': 2,
                'f_bus': 2,
                't_bus': 3,
                'br_r': 0.001,
                'br_x': 0.01,
                'br_status': 1,
                'rate_a': 5.0,
                'rate_b': 5.0,
                'rate_c': 5.0,
                'tap': 1.0,
                'shift': 0.0,
                'g_fr': 0.0,
                'g_to': 0.0,
                'b_fr': 0.0,
                'b_to': 0.0,
                'angmin': -60.0,
                'angmax': 60.0
            }
        },
        'load': {
            '1': {
                'index': 1,
                'load_bus': 3,
                'pd': 2.2,         # Total load matches total generation (1.5 + 0.8 = 2.3, minus losses)
                'qd': 0.4,         # Reactive load
                'status': 1
            }
        },
        'shunt': {},
        'dcline': {},
        'storage': {},
        'switch': {},
        'baseMVA': 100.0,
        'per_unit': True,
        'name': 'droop_test_case',
        'source_version': '2',
        'source_type': 'matpower'
    }



    print("=" * 60)
    print("Testing Droop Control Power Flow")
    print("=" * 60)
    
    # Enable droop
    config['droop_control']['enabled'] = True
    
    print("\nSolving with droop control...")
    print(f"Droop buses: {config['droop_control']['droop_buses']}")
    print(f"mp (P-droop): {config['droop_control']['mp']}")
    print(f"mq (Q-droop): {config['droop_control']['mq']}")
    
    try:
        result = solve_power_flow(network, config)
        print("\n✓ Test completed successfully!")
        print(f"Status: {result.get('termination_status', 'UNKNOWN')}")
        
        if 'droop_summary' in result:
            print("\nDroop Results:")
            print(f"  Frequency: {result['droop_summary']['system_frequency_hz']:.4f} Hz")
            print(f"  Deviation: {result['droop_summary']['frequency_deviation_hz']:.4f} Hz")
        else:
            print("\n⚠ No droop summary in results")
    
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_droop()