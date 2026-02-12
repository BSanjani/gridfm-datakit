import pandas as pd
import numpy as np
from pathlib import Path

# Path to the Without Droop data
without_droop_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_Withoutdroop_10k\case24_ieee_rts\raw")

print("="*100)
print(" "*30 + "DIAGNOSING 'WITHOUT DROOP' LOW VOLTAGE ISSUE")
print("="*100)

# Load the bus data
bus_file = without_droop_dir / 'bus_data.parquet'
if not bus_file.exists():
    print(f"\n✗ File not found: {bus_file}")
    exit()

print(f"\n✓ Loading: {bus_file}")
bus_df = pd.read_parquet(bus_file)

print(f"\n[1] BASIC INFO")
print(f"  Shape: {bus_df.shape}")
print(f"  Columns: {list(bus_df.columns)}")
print(f"  Number of unique buses: {bus_df['bus'].nunique() if 'bus' in bus_df.columns else 'N/A'}")
print(f"  Number of unique scenarios: {bus_df['scenario'].nunique() if 'scenario' in bus_df.columns else 'N/A'}")

print(f"\n[2] VOLTAGE STATISTICS")
print(f"  Overall voltage (Vm) statistics:")
print(f"    Mean:   {bus_df['Vm'].mean():.6f} p.u.")
print(f"    Median: {bus_df['Vm'].median():.6f} p.u.")
print(f"    Std:    {bus_df['Vm'].std():.6f} p.u.")
print(f"    Min:    {bus_df['Vm'].min():.6f} p.u.")
print(f"    Max:    {bus_df['Vm'].max():.6f} p.u.")

print(f"\n[3] IDENTIFY PROBLEMATIC SCENARIOS")
# Find scenarios with very low voltage
low_voltage_threshold = 0.8  # p.u.
problematic = bus_df[bus_df['Vm'] < low_voltage_threshold]

print(f"  Number of bus-scenario combinations with Vm < {low_voltage_threshold}: {len(problematic)}")
print(f"  Percentage: {100*len(problematic)/len(bus_df):.2f}%")

if len(problematic) > 0:
    print(f"\n  Scenarios with lowest voltages:")
    worst_scenarios = problematic.nsmallest(20, 'Vm')[['scenario', 'bus', 'Vm', 'Va', 'Pd', 'Qd', 'Pg', 'Qg']]
    print(worst_scenarios.to_string())
    
    # Count how many scenarios have critical low voltage
    critical_scenarios = problematic['scenario'].unique()
    print(f"\n  Total unique scenarios with Vm < {low_voltage_threshold}: {len(critical_scenarios)}")
    print(f"  Out of {bus_df['scenario'].nunique()} total scenarios ({100*len(critical_scenarios)/bus_df['scenario'].nunique():.2f}%)")

print(f"\n[4] VOLTAGE DISTRIBUTION BY BUS")
bus_voltage_stats = bus_df.groupby('bus')['Vm'].agg(['mean', 'std', 'min', 'max', 'count']).round(6)
bus_voltage_stats = bus_voltage_stats.sort_values('min')
print(f"  Bus voltage statistics (sorted by minimum voltage):")
print(bus_voltage_stats.to_string())

# Identify which buses have the lowest minimum voltage
print(f"\n[5] BUSES WITH CRITICAL LOW VOLTAGE")
critical_buses = bus_voltage_stats[bus_voltage_stats['min'] < low_voltage_threshold]
if len(critical_buses) > 0:
    print(f"  Buses with min voltage < {low_voltage_threshold}:")
    print(critical_buses.to_string())
else:
    print(f"  No buses with min voltage < {low_voltage_threshold}")

print(f"\n[6] EXAMINE WORST CASE SCENARIO")
# Find the scenario with the absolute minimum voltage
worst_scenario_idx = bus_df['Vm'].idxmin()
worst_scenario_data = bus_df.loc[worst_scenario_idx]
worst_scenario_num = worst_scenario_data['scenario']

print(f"  Worst voltage occurs at:")
print(f"    Scenario: {worst_scenario_num}")
print(f"    Bus: {worst_scenario_data['bus']}")
print(f"    Voltage: {worst_scenario_data['Vm']:.6f} p.u.")

# Get all bus data for this worst scenario
print(f"\n  All buses in scenario {worst_scenario_num}:")
worst_scenario_all_buses = bus_df[bus_df['scenario'] == worst_scenario_num][['bus', 'Vm', 'Va', 'Pd', 'Qd', 'Pg', 'Qg']].sort_values('Vm')
print(worst_scenario_all_buses.to_string())

print(f"\n[7] VOLTAGE HISTOGRAM")
# Create voltage bins
voltage_bins = [0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 2.0]
voltage_distribution = pd.cut(bus_df['Vm'], bins=voltage_bins, include_lowest=True)
voltage_counts = voltage_distribution.value_counts().sort_index()

print(f"  Voltage distribution:")
for interval, count in voltage_counts.items():
    pct = 100 * count / len(bus_df)
    print(f"    {str(interval):20s}: {count:8d} ({pct:6.2f}%)")

print(f"\n[8] CHECK FOR CONVERGENCE ISSUES")
# Check if there are any NaN or infinite values
nan_count = bus_df['Vm'].isna().sum()
inf_count = np.isinf(bus_df['Vm']).sum()
print(f"  NaN values in Vm: {nan_count}")
print(f"  Infinite values in Vm: {inf_count}")

# Check if voltage is exactly 0 (might indicate failed simulation)
zero_voltage = (bus_df['Vm'] == 0).sum()
print(f"  Zero voltage entries: {zero_voltage}")

print(f"\n[9] LOAD AND GENERATION STATS FOR LOW VOLTAGE SCENARIOS")
if len(problematic) > 0:
    print(f"  For scenarios with Vm < {low_voltage_threshold}:")
    print(f"    Mean Load (Pd):       {problematic['Pd'].mean():.2f} MW")
    print(f"    Mean Generation (Pg): {problematic['Pg'].mean():.2f} MW")
    print(f"    Mean Voltage Angle:   {problematic['Va'].mean():.2f} degrees")
    
    # Compare to normal scenarios
    normal = bus_df[bus_df['Vm'] >= low_voltage_threshold]
    print(f"\n  For scenarios with Vm >= {low_voltage_threshold}:")
    print(f"    Mean Load (Pd):       {normal['Pd'].mean():.2f} MW")
    print(f"    Mean Generation (Pg): {normal['Pg'].mean():.2f} MW")
    print(f"    Mean Voltage Angle:   {normal['Va'].mean():.2f} degrees")

print(f"\n[10] RECOMMENDATION")
if len(problematic) > 0:
    print(f"  ⚠ WARNING: {len(critical_scenarios)} scenarios ({100*len(critical_scenarios)/bus_df['scenario'].nunique():.2f}%) have voltage collapse!")
    print(f"  This suggests:")
    print(f"    1. Without droop control, the system cannot maintain voltage stability")
    print(f"    2. These scenarios may represent failed power flow convergence")
    print(f"    3. The data may need filtering to remove non-converged scenarios")
    print(f"\n  Suggested actions:")
    print(f"    - Filter out scenarios with Vm < 0.8 p.u. (voltage collapse)")
    print(f"    - Check runtime_data.parquet for convergence flags")
    print(f"    - Verify that power flow solver converged for all scenarios")
else:
    print(f"  ✓ No critical voltage issues found.")

print("\n" + "="*100)