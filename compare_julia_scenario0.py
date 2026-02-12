"""
Compare PyPower droop solver with Julia scenario 0 - WITH OPF
"""
import yaml
import numpy as np
import pandas as pd
from pypower.api import case24_ieee_rts, runopf, ppoption
from pypower_droop_solver import PyPowerDroopSolver

print("="*70)
print(" "*15 + "PYPOWER vs JULIA SCENARIO 0 COMPARISON (WITH OPF)")
print("="*70)

# Load Julia scenario 0 data (for loads only)
bus_df = pd.read_parquet(r'data_out_ieee24_droop_droop_10k\case24_ieee_rts\raw\bus_data.parquet')
gen_df = pd.read_parquet(r'data_out_ieee24_droop_droop_10k\case24_ieee_rts\raw\gen_data.parquet')
rt_df = pd.read_parquet(r'data_out_ieee24_droop_droop_10k\case24_ieee_rts\raw\runtime_data.parquet')

s0_bus = bus_df[bus_df['scenario'] == 0].copy()
s0_gen = gen_df[gen_df['scenario'] == 0].copy()
s0_rt = rt_df[rt_df['scenario'] == 0].iloc[0]

print(f"\nJulia Scenario 0:")
print(f"  Frequency deviation: {s0_rt['frequency_deviation']:.6f} p.u.")
print(f"  Frequency: {60 + s0_rt['frequency_deviation']*60:.4f} Hz")

# Load PyPower case
ppc = case24_ieee_rts()
baseMVA = ppc['baseMVA']

# Apply ONLY Julia's loads (not generation - we'll get that from OPF)
for _, row in s0_bus.iterrows():
    bus_idx = int(row['bus'])
    ppc['bus'][bus_idx, 2] = row['Pd']  # PD in MW
    ppc['bus'][bus_idx, 3] = row['Qd']  # QD in MVAr

print(f"\nLoad applied from Julia scenario 0")
print(f"  Total Load: {np.sum(ppc['bus'][:, 2]):.2f} MW")

# Step 1: Run OPF to get optimal setpoints (like Julia does)
print("\n" + "="*70)
print("STEP 1: Running OPF for initial setpoints...")
print("="*70)

ppopt = ppoption(VERBOSE=1, OUT_ALL=0)
opf_result = runopf(ppc, ppopt)
# 1. Total Generation (Column 1 of the gen matrix is Pg in MW)
total_gen_opf = np.sum(opf_result['gen'][:, 1])
    
# 2. Total Demand (Column 2 of the bus matrix is Pd in MW)
# Note: Use opf_result['bus'] to get the load used DURING the OPF
total_load_opf = np.sum(opf_result['bus'][:, 2])
    
# 3. System Losses
total_losses_opf = total_gen_opf - total_load_opf

print("\n" + "-"*40)
print("STEP 1: OPF STEADY-STATE BALANCE (60Hz)")
print("-"*40)
print(f"Total Generation: {total_gen_opf:10.2f} MW")
print(f"Total Load:       {total_load_opf:10.2f} MW")
print(f"System Losses:    {total_losses_opf:10.2f} MW")
print(f"Loss Percentage:  {(total_losses_opf/total_gen_opf)*100:10.2f}%")
print("-"*40)

ppc_for_droop = opf_result.copy()
ppc_for_droop['bus'][:, 2] *= 1.022 # Scale loads by 2% to create a mismatch for droop to respond to

if opf_result['success']:
    print(f"✓ OPF converged")
    print(f"  Total Generation: {np.sum(opf_result['gen'][:, 1]):.2f} MW")
    print(f"  Total Load: {np.sum(opf_result['bus'][:, 2]):.2f} MW")
    
    # Use OPF solution as the starting point for droop
    ppc_for_droop = opf_result.copy()
else:
    print("✗ OPF failed - using base case")
    ppc_for_droop = ppc.copy()

# Load config for droop buses
with open('user_config_droop_1500.yaml', 'r') as f:
    config = yaml.safe_load(f)

droop_cfg = config['droop_control']

# Extract per-generator droop coefficients from Julia
gen_droop_map = {}
for _, row in s0_gen.iterrows():
    bus_id = int(row['bus'])
    if bus_id in [int(b) for b in droop_cfg['droop_buses']]:
        gen_idx = int(row['idx'])
        gen_droop_map[gen_idx] = {
            'mp': row['mp_droop'],
            'mq': row['mq_droop']
        }

droop_config = {
    'enabled': True,
    'mp': 0.04,
    'mq': 0.025,
    'V_0': droop_cfg['V_0'],
    'frequency_deadband': droop_cfg['frequency_deadband'],
    'voltage_deadband': droop_cfg['voltage_deadband'],
    'droop_buses': [int(b) for b in droop_cfg['droop_buses']],
    'gen_droop_map': gen_droop_map
}

# Step 2: Run Droop PF using OPF setpoints
print("\n" + "="*70)
print("STEP 2: Running Droop PF with OPF setpoints...")
print("="*70)

solver = PyPowerDroopSolver(ppc_for_droop, droop_config)
results = solver.solve(verbose=True)

print("\n" + "="*70)
if results['converged']:

    # The solver returns the updated ppc in the 'ppc' key
    final_ppc = results['ppc']
    
    # 1. Total Final Generation (Sum of all generators after droop response)
    total_gen_final = np.sum(final_ppc['gen'][:, 1])
    
    # 2. Total Final Load (This includes the 1.02 scaling you applied)
    total_load_final = np.sum(final_ppc['bus'][:, 2])
    
    # 3. Final System Losses at the new frequency/dispatch
    total_losses_final = total_gen_final - total_load_final

    print("\n" + "-"*40)
    print(f"STEP 2: DROOP PF BALANCE (Freq: {60 + results['df']*60:.4f} Hz)")
    print("-"*40)
    print(f"Total Generation: {total_gen_final:10.2f} MW")
    print(f"Total Load:       {total_load_final:10.2f} MW")
    print(f"System Losses:    {total_losses_final:10.2f} MW")
    print(f"Gen Increase:     {total_gen_final - total_gen_opf:10.2f} MW (from droop)")
    print(f"Loss Increase:    {total_losses_final - total_losses_opf:10.2f} MW")
    print("-" * 40)
    
    print("✅ PYPOWER CONVERGED")
    print(f"\nResults:")
    print(f"  Frequency deviation: {results['df']:.6f} p.u.")
    print(f"  Frequency: {60 + results['df']*60:.4f} Hz")
    print(f"  Iterations: {results['iterations']}")
    
    # Compare with Julia
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    julia_df = s0_rt['frequency_deviation']
    pypower_df = results['df']
    
    print(f"\nJulia:   df = {julia_df:.6f} p.u.")
    print(f"PyPower: df = {pypower_df:.6f} p.u.")
    print(f"Difference:  {abs(julia_df - pypower_df):.6f} p.u.")
    
    diff_pct = abs(julia_df - pypower_df) / abs(julia_df) * 100
    print(f"Relative error: {diff_pct:.2f}%")
    
    if abs(julia_df - pypower_df) < 0.0001:
        print("\n✅ EXCELLENT MATCH (< 0.0001 p.u.)")
    elif abs(julia_df - pypower_df) < 0.001:
        print("\n✅ GOOD MATCH (< 0.001 p.u.)")
    elif diff_pct < 5.0:
        print("\n✅ ACCEPTABLE (< 5% error)")
    elif diff_pct < 10.0:
        print("\n⚠️  FAIR (< 10% error)")
    else:
        print("\n❌ LARGE DIFFERENCE (> 10% error)")
        
else:
    print("❌ PYPOWER FAILED TO CONVERGE")

print("="*70)