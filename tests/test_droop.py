# test_droop.py

import yaml
from pfdelta.solver import solve_power_flow
# Add your network loading import here
# from pfdelta.network import load_network  # or wherever your load function is

def test_droop():
    """Test droop control power flow"""
    
    # Load configuration
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load your network - REPLACE THIS with your actual network loading
    # network = load_network('your_network_file.mat')  # or however you load it
    
    # For now, create a simple test network dictionary
    network = {
        'bus': {
            1: {'bus_type': 3, 'vmax': 1.1, 'vmin': 0.9},
            2: {'bus_type': 2, 'vmax': 1.1, 'vmin': 0.9},
            5: {'bus_type': 1, 'vmax': 1.1, 'vmin': 0.9}
        },
        'gen': {
            1: {'gen_bus': 1, 'pg': 1.0, 'qg': 0.5},
            2: {'gen_bus': 2, 'pg': 0.8, 'qg': 0.3}
        },
        'branch': {},
        'load': {}
    }
    
    print("=" * 60)
    print("Testing Droop Control Power Flow")
    print("=" * 60)
    
    # Enable droop
    config['droop_control']['enabled'] = True
    
    print("\nSolving with droop control...")
    try:
        result = solve_power_flow(network, config)
        print("\n✓ Test completed successfully!")
        
        if 'droop_summary' in result:
            print("\nDroop Results:")
            print(f"  Frequency: {result['droop_summary']['system_frequency_hz']:.4f} Hz")
            print(f"  Deviation: {result['droop_summary']['frequency_deviation_hz']:.4f} Hz")
    
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_droop()