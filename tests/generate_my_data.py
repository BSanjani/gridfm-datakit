from gridfm_datakit.generate import generate_power_flow_data_distributed
from gridfm_datakit.config import Config
from gridfm_datakit.utils.param_handler import NestedNamespace
import os

# Create a simple configuration
config = NestedNamespace(**{
    'network': {
        'name': 'pglib_opf_case14_ieee',
        'source': 'file',
        'network_dir': 'gridfm_datakit/grids'
    },
    'load': {
        'generator': 'agg_load_profile',
        'agg_profile': 'default',
        'scenarios': 100,  # Small number for testing
        'sigma': 0.2,
        'change_reactive_power': True,
        'global_range': 0.4,
        'max_scaling_factor': 4.0,
        'step_size': 0.1,
        'start_scaling_factor': 1.0
    },
    'topology_perturbation': {
        'type': 'none'  # Disable for simplicity
    },
    'generation_perturbation': {
        'type': 'none'  # Disable for simplicity
    },
    'admittance_perturbation': {
        'type': 'none'  # Disable for simplicity
    },
    'settings': {
        'num_processes': 2,
        'data_dir': './my_output_data',  # This is where your data will be!
        'large_chunk_size': 100,
        'overwrite': True,
        'mode': 'pf',
        'include_dc_res': True,
        'enable_solver_logs': False,
        'pf_fast': True,
        'dcpf_fast': True,
        'max_iter': 200,
        'seed': 42
    }
})

print("Starting data generation...")
print(f"Output directory: {os.path.abspath(config.settings.data_dir)}")
print(f"Network: {config.network.name}")
print(f"Scenarios: {config.load.scenarios}")

try:
    file_paths = generate_power_flow_data_distributed(config)
    print("\n✓ SUCCESS! Data generated at:")
    for fp in file_paths:
        print(f"  - {fp}")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()