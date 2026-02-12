import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("POWER FLOW COMPARISON: DROOP CONTROL VS BASELINE (10,000 SCENARIOS)")
print("="*80)

# Load data
print("\n[1/6] Loading data...")

# Baseline (no droop)
bus_baseline = pd.read_parquet('data_out_baseline_no_droop_10000/case24_ieee_rts/raw/bus_data.parquet')
branch_baseline = pd.read_parquet('data_out_baseline_no_droop_10000/case24_ieee_rts/raw/branch_data.parquet')
gen_baseline = pd.read_parquet('data_out_baseline_no_droop_10000/case24_ieee_rts/raw/gen_data.parquet')

# Droop control
bus_droop = pd.read_parquet('data_out_droop_control_10000/case24_ieee_rts/raw/bus_data.parquet')
branch_droop = pd.read_parquet('data_out_droop_control_10000/case24_ieee_rts/raw/branch_data.parquet')
gen_droop = pd.read_parquet('data_out_droop_control_10000/case24_ieee_rts/raw/gen_data.parquet')

n_baseline = len(bus_baseline['scenario'].unique())
n_droop = len(bus_droop['scenario'].unique())

print(f"Baseline scenarios: {n_baseline}")
print(f"Droop scenarios: {n_droop}")

# Calculate voltage deviation
print("\n[2/6] Calculating voltage metrics...")
bus_baseline['V_deviation'] = abs(bus_baseline['Vm'] - 1.0)
bus_droop['V_deviation'] = abs(bus_droop['Vm'] - 1.0)

# Calculate voltage violations
baseline_violations = ((bus_baseline['Vm'] < 0.95) | (bus_baseline['Vm'] > 1.05)).sum()
droop_violations = ((bus_droop['Vm'] < 0.95) | (bus_droop['Vm'] > 1.05)).sum()

# Summary statistics
print("\n[3/6] Computing statistics...")

summary = {
    'Metric': [],
    'Baseline': [],
    'Droop': [],
    'Improvement': []
}

# Voltage statistics
summary['Metric'].append('Avg Voltage (p.u.)')
summary['Baseline'].append(bus_baseline['Vm'].mean())
summary['Droop'].append(bus_droop['Vm'].mean())
summary['Improvement'].append(f"{((bus_droop['Vm'].mean() - bus_baseline['Vm'].mean())/bus_baseline['Vm'].mean()*100):.3f}%")

summary['Metric'].append('Voltage Std Dev')
summary['Baseline'].append(bus_baseline['Vm'].std())
summary['Droop'].append(bus_droop['Vm'].std())
summary['Improvement'].append(f"{((bus_baseline['Vm'].std() - bus_droop['Vm'].std())/bus_baseline['Vm'].std()*100):.3f}%")

summary['Metric'].append('Avg V Deviation')
summary['Baseline'].append(bus_baseline['V_deviation'].mean())
summary['Droop'].append(bus_droop['V_deviation'].mean())
summary['Improvement'].append(f"{((bus_baseline['V_deviation'].mean() - bus_droop['V_deviation'].mean())/bus_baseline['V_deviation'].mean()*100):.3f}%")

summary['Metric'].append('Avg Gen P (MW)')
summary['Baseline'].append(gen_baseline['p_mw'].mean())
summary['Droop'].append(gen_droop['p_mw'].mean())
summary['Improvement'].append(f"{((gen_droop['p_mw'].mean() - gen_baseline['p_mw'].mean())/gen_baseline['p_mw'].mean()*100):.3f}%")

summary['Metric'].append('Avg Gen Q (MVAr)')
summary['Baseline'].append(gen_baseline['q_mvar'].mean())
summary['Droop'].append(gen_droop['q_mvar'].mean())
summary['Improvement'].append(f"{((gen_droop['q_mvar'].mean() - gen_baseline['q_mvar'].mean())/gen_baseline['q_mvar'].mean()*100):.3f}%")

summary['Metric'].append('Avg Branch Flow (MW)')
summary['Baseline'].append(abs(branch_baseline['pf']).mean())
summary['Droop'].append(abs(branch_droop['pf']).mean())
summary['Improvement'].append(f"{((abs(branch_baseline['pf']).mean() - abs(branch_droop['pf']).mean())/abs(branch_baseline['pf']).mean()*100):.3f}%")

summary['Metric'].append('Max Branch Flow (MW)')
summary['Baseline'].append(abs(branch_baseline['pf']).max())
summary['Droop'].append(abs(branch_droop['pf']).max())
summary['Improvement'].append(f"{((abs(branch_baseline['pf']).max() - abs(branch_droop['pf']).max())/abs(branch_baseline['pf']).max()*100):.3f}%")

df_summary = pd.DataFrame(summary)

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print(df_summary.to_string(index=False))
print("="*80)

# Save to file
df_summary.to_csv('comparison_summary_10k.csv', index=False)
print("\nSummary saved to: comparison_summary_10k.csv")

print("\n[4/6] Creating visualizations...")

# Bus-by-bus voltage comparison
voltage_by_bus_baseline = bus_baseline.groupby('bus')['Vm'].mean()
voltage_by_bus_droop = bus_droop.groupby('bus')['Vm'].mean()

plt.figure(figsize=(12, 6))
plt.plot(voltage_by_bus_baseline.index, voltage_by_bus_baseline.values, 'o-', linewidth=2, markersize=6, label='Baseline (No Droop)', color='red')
plt.plot(voltage_by_bus_droop.index, voltage_by_bus_droop.values, 's-', linewidth=2, markersize=6, label='Droop Control', color='blue')
plt.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='Rated Voltage (1.0 p.u.)')
plt.xlabel('Bus Number', fontsize=12)
plt.ylabel('Voltage Magnitude (p.u.)', fontsize=12)
plt.title('Voltage Profile Comparison (10,000 Scenarios)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('voltage_comparison_10k.png', dpi=300, bbox_inches='tight')
print("Saved: voltage_comparison_10k.png")

# Distribution comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(bus_baseline['Vm'], bins=50, alpha=0.7, label='Baseline', edgecolor='black', color='red')
axes[0, 0].hist(bus_droop['Vm'], bins=50, alpha=0.7, label='Droop', edgecolor='black', color='blue')
axes[0, 0].set_xlabel('Voltage Magnitude (p.u.)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Voltage Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(gen_baseline['p_mw'], bins=50, alpha=0.7, label='Baseline', edgecolor='black', color='red')
axes[0, 1].hist(gen_droop['p_mw'], bins=50, alpha=0.7, label='Droop', edgecolor='black', color='blue')
axes[0, 1].set_xlabel('Generator Output (MW)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Active Power Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(branch_baseline['pf'], bins=50, alpha=0.7, label='Baseline', edgecolor='black', color='red')
axes[1, 0].hist(branch_droop['pf'], bins=50, alpha=0.7, label='Droop', edgecolor='black', color='blue')
axes[1, 0].set_xlabel('Branch Power Flow (MW)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Branch Flow Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(gen_baseline['q_mvar'], bins=50, alpha=0.7, label='Baseline', edgecolor='black', color='red')
axes[1, 1].hist(gen_droop['q_mvar'], bins=50, alpha=0.7, label='Droop', edgecolor='black', color='blue')
axes[1, 1].set_xlabel('Reactive Power (MVAr)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Reactive Power Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distributions_comparison_10k.png', dpi=300, bbox_inches='tight')
print("Saved: distributions_comparison_10k.png")

plt.close('all')

print("\n[5/6] Generating detailed report...")

report = f"""
{'='*80}
POWER FLOW COMPARISON REPORT: DROOP CONTROL VS BASELINE
{'='*80}
Dataset: IEEE 24-Bus Reliability Test System
Scenarios: 10,000 each (Baseline and Droop Control)
{'='*80}

VOLTAGE PERFORMANCE
{'='*80}

Mean Voltage:
  Baseline:   {bus_baseline['Vm'].mean():.6f} p.u.
  Droop:      {bus_droop['Vm'].mean():.6f} p.u.

Voltage Std Dev:
  Baseline:   {bus_baseline['Vm'].std():.6f} p.u.
  Droop:      {bus_droop['Vm'].std():.6f} p.u.
  Improvement: {((bus_baseline['Vm'].std() - bus_droop['Vm'].std())/bus_baseline['Vm'].std()*100):.2f}%

{'='*80}
GENERATOR PERFORMANCE
{'='*80}

Reactive Power:
  Baseline:   {gen_baseline['q_mvar'].mean():.3f} MVAr
  Droop:      {gen_droop['q_mvar'].mean():.3f} MVAr
  Change:     {((gen_droop['q_mvar'].mean() - gen_baseline['q_mvar'].mean())/gen_baseline['q_mvar'].mean()*100):.2f}%

{'='*80}
BRANCH FLOW PERFORMANCE
{'='*80}

Maximum Branch Flow:
  Baseline:   {abs(branch_baseline['pf']).max():.3f} MW
  Droop:      {abs(branch_droop['pf']).max():.3f} MW
  Reduction:   {((abs(branch_baseline['pf']).max() - abs(branch_droop['pf']).max())/abs(branch_baseline['pf']).max()*100):.2f}%

{'='*80}
"""

with open('comparison_report_10k.txt', 'w') as f:
    f.write(report)

print(report)
print("\n[6/6] Complete!")
print("="*80)
