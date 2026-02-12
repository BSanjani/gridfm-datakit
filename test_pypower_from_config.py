"""
Test PyPower droop solver using the same config as Julia
"""
import yaml
import numpy as np
from pypower.api import case24_ieee_rts
from pypower_droop_solver import PyPowerDroopSolver
import pandas as pd

# Load config
with open('user_config_droop_1500.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("="*70)
print(" "*15 + "PYPOWER TEST WITH JULIA CONFIG")
print("="*70)

# Load case
ppc = case24_ieee_rts()
print(f"\nLoaded case: IEEE 24-bus RTS")
print(f"  Buses: {ppc['bus'].shape[0]}")
print(f"  Generators: {ppc['gen'].shape[0]}")
print(f"  Base MVA: {ppc['baseMVA']}")

# Extract droop config
droop_cfg = config['droop_control']
droop_config = {
    'enabled': droop_cfg['enabled'],
    'mp': 0.03,  # Use average of range [0.03, 0.05]
    'mq': 0.02,  # Use average of range [0.02, 0.04]
    'V_0': droop_cfg['V_0'],
    'frequency_deadband': droop_cfg['frequency_deadband'],
    'voltage_deadband': droop_cfg['voltage_deadband'],
    'droop_buses': [int(b) for b in droop_cfg['droop_buses']]
}

print(f"\nDroop Configuration:")
print(f"  Enabled: {droop_config['enabled']}")
print(f"  mp: {droop_config['mp']}")
print(f"  mq: {droop_config['mq']}")
print(f"  Frequency deadband: {droop_config['frequency_deadband']} p.u.")
print(f"  Droop buses: {droop_config['droop_buses']}")

# IMPORTANT: Match Julia's load-generation imbalance
# Julia shows: Load = 31.35 p.u., Gen = 31.65-31.75 p.u.
print(f"\nOriginal Power Balance:")
total_load = np.sum(ppc['bus'][:, 2]) / ppc['baseMVA']  # PD column
total_gen = np.sum(ppc['gen'][:, 1]) / ppc['baseMVA']   # PG column
print(f"  Total Load: {total_load:.4f} p.u.")
print(f"  Total Gen:  {total_gen:.4f} p.u.")
print(f"  Imbalance:  {total_gen - total_load:.4f} p.u.")

# Adjust to match Julia (31.35 load, ~31.65 gen)
load_scale = 31.35 / total_load
ppc['bus'][:, 2] *= load_scale  # Scale PD
ppc['bus'][:, 3] *= load_scale  # Scale QD

gen_scale = 31.65 / total_gen
ppc['gen'][:, 1] *= gen_scale  # Scale PG
ppc['gen'][:, 2] *= gen_scale  # Scale QG

total_load_new = np.sum(ppc['bus'][:, 2]) / ppc['baseMVA']
total_gen_new = np.sum(ppc['gen'][:, 1]) / ppc['baseMVA']

print(f"\nAdjusted Power Balance (to match Julia):")
print(f"  Total Load: {total_load_new:.4f} p.u.")
print(f"  Total Gen:  {total_gen_new:.4f} p.u.")
print(f"  Imbalance:  {total_gen_new - total_load_new:.4f} p.u.")

# Run PyPower droop solver
print("\n" + "="*70)
print("Running PyPower Droop Solver...")
print("="*70)

solver = PyPowerDroopSolver(ppc, droop_config)
results = solver.solve(verbose=True)

print("\n" + "="*70)
if results['converged']:
    print("✅ PYPOWER CONVERGED")
    print(f"   Frequency deviation: {results['df']:.6f} p.u.")
    print(f"   Frequency: {60 + results['df']*60:.4f} Hz")
    print(f"   Iterations: {results['iterations']}")
    
    # Compare with Julia results
    print("\n" + "="*70)
    print("COMPARISON WITH JULIA")
    print("="*70)
    
    # Load Julia results
    try:
        julia_df = pd.read_parquet(r'data_out_ieee24_droop_droop_10k\case24_ieee_rts\raw\runtime_data.parquet')
        julia_freq_dev = julia_df['frequency_deviation'].values
        
        print(f"\nJulia frequency deviation:")
        print(f"  Mean: {julia_freq_dev.mean():.6f} p.u.")
        print(f"  Std:  {julia_freq_dev.std():.6f} p.u.")
        print(f"  Range: [{julia_freq_dev.min():.6f}, {julia_freq_dev.max():.6f}]")
        
        print(f"\nPyPower frequency deviation:")
        print(f"  Value: {results['df']:.6f} p.u.")
        
        print(f"\nDifference:")
        diff = abs(results['df'] - julia_freq_dev.mean())
        print(f"  |PyPower - Julia mean|: {diff:.6f} p.u.")
        
        if diff < 0.001:
            print("  ✅ GOOD MATCH (< 0.001 p.u. difference)")
        elif diff < 0.005:
            print("  ⚠️  ACCEPTABLE (< 0.005 p.u. difference)")
        else:
            print("  ❌ LARGE DIFFERENCE (> 0.005 p.u.)")
            
    except FileNotFoundError:
        print("  (Julia results not found - run Julia first)")
        
else:
    print("❌ PYPOWER FAILED TO CONVERGE")

print("="*70)