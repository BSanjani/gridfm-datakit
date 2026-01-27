"""
Comprehensive Results Generator for GridFM Power Flow Analysis
Generates all publication-ready outputs, tables, and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import json

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path('results_output')
output_dir.mkdir(exist_ok=True)

print("="*70)
print("COMPREHENSIVE RESULTS GENERATOR FOR GRIDFM POWER FLOW ANALYSIS")
print("="*70)

# ============================================================================
# 1. LOAD DATA AND MODEL
# ============================================================================

print("\n[1/8] Loading data and model...")

# Load data
bus_data = pd.read_parquet('data_out/case24_ieee_rts/raw/bus_data.parquet')
gen_data = pd.read_parquet('data_out/case24_ieee_rts/raw/gen_data.parquet')
branch_data = pd.read_parquet('data_out/case24_ieee_rts/raw/branch_data.parquet')
runtime_data = pd.read_parquet('data_out/case24_ieee_rts/raw/runtime_data.parquet')

# Model architecture
class PowerFlowNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128]):
        super(PowerFlowNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Load model
model = PowerFlowNN(input_dim=48, output_dim=48)
model.load_state_dict(torch.load('best_power_flow_model.pth'))
model.eval()

# ============================================================================
# 2. DATASET STATISTICS SUMMARY
# ============================================================================

print("[2/8] Generating dataset statistics...")

stats_summary = {
    'Total Samples': len(bus_data),
    'Number of Scenarios': bus_data['scenario'].nunique(),
    'Number of Buses': bus_data['bus'].nunique(),
    'Number of Generators': gen_data['idx'].nunique(),
    'Number of Branches': branch_data['idx'].nunique(),
    'Load Scenarios': bus_data['load_scenario_idx'].nunique(),
    'Topology Variants per Load': bus_data['scenario'].nunique() // bus_data['load_scenario_idx'].nunique() if bus_data['load_scenario_idx'].nunique() > 0 else 0,
}

# Save to JSON
with open(output_dir / 'dataset_statistics.json', 'w') as f:
    json.dump(stats_summary, f, indent=2)

# Create summary table
summary_df = pd.DataFrame.from_dict(stats_summary, orient='index', columns=['Value'])
summary_df.to_csv(output_dir / 'dataset_statistics.csv')

print(f"   Dataset contains {stats_summary['Total Samples']:,} samples")
print(f"   Saved to: {output_dir / 'dataset_statistics.csv'}")

# ============================================================================
# 3. VOLTAGE STATISTICS BY BUS
# ============================================================================

print("[3/8] Analyzing voltage statistics by bus...")

voltage_stats = bus_data.groupby('bus').agg({
    'Vm': ['mean', 'std', 'min', 'max', 'median'],
    'Va': ['mean', 'std', 'min', 'max', 'median']
}).round(4)

voltage_stats.columns = ['_'.join(col) for col in voltage_stats.columns]

# Add violation statistics
voltage_stats['Vm_violations_%'] = bus_data.groupby('bus')['Vm'].apply(
    lambda x: ((x > 1.05) | (x < 0.95)).sum() / len(x) * 100
).round(2)

voltage_stats.to_csv(output_dir / 'voltage_statistics_by_bus.csv')
print(f"   Saved to: {output_dir / 'voltage_statistics_by_bus.csv'}")

# ============================================================================
# 4. POWER FLOW STATISTICS
# ============================================================================

print("[4/8] Analyzing power flow statistics...")

# Bus-level statistics
bus_power_stats = bus_data.groupby('bus').agg({
    'Pd': ['mean', 'std', 'min', 'max'],
    'Qd': ['mean', 'std', 'min', 'max'],
    'Pg': ['mean', 'std', 'min', 'max'],
    'Qg': ['mean', 'std', 'min', 'max']
}).round(2)

bus_power_stats.columns = ['_'.join(col) for col in bus_power_stats.columns]
bus_power_stats.to_csv(output_dir / 'power_statistics_by_bus.csv')

# Generator statistics
gen_stats = gen_data.groupby('idx').agg({
    'p_mw': ['mean', 'std', 'min', 'max'],
    'q_mvar': ['mean', 'std', 'min', 'max']
}).round(2)

gen_stats.columns = ['_'.join(col) for col in gen_stats.columns]
gen_stats.to_csv(output_dir / 'generator_statistics.csv')

# Branch statistics
branch_stats = branch_data.groupby('idx').agg({
    'pf': ['mean', 'std', 'min', 'max'],
    'qf': ['mean', 'std', 'min', 'max'],
    'pt': ['mean', 'std', 'min', 'max'],
    'qt': ['mean', 'std', 'min', 'max']
}).round(2)

branch_stats.columns = ['_'.join(col) for col in branch_stats.columns]
branch_stats.to_csv(output_dir / 'branch_statistics.csv')

print(f"   Saved power statistics to {output_dir}/")

# ============================================================================
# 5. SOLVER RUNTIME ANALYSIS
# ============================================================================

print("[5/8] Analyzing solver runtime...")

runtime_summary = {
    'AC Power Flow': {
        'Mean (s)': runtime_data['ac'].mean(),
        'Median (s)': runtime_data['ac'].median(),
        'Std (s)': runtime_data['ac'].std(),
        'Min (s)': runtime_data['ac'].min(),
        'Max (s)': runtime_data['ac'].max()
    }
}

if 'dc' in runtime_data.columns:
    runtime_summary['DC Power Flow'] = {
        'Mean (s)': runtime_data['dc'].mean(),
        'Median (s)': runtime_data['dc'].median(),
        'Std (s)': runtime_data['dc'].std(),
        'Min (s)': runtime_data['dc'].min(),
        'Max (s)': runtime_data['dc'].max()
    }

runtime_df = pd.DataFrame(runtime_summary).T
runtime_df.to_csv(output_dir / 'runtime_statistics.csv')
print(f"   Average AC-PF solve time: {runtime_data['ac'].mean():.4f} seconds")
print(f"   Saved to: {output_dir / 'runtime_statistics.csv'}")

# ============================================================================
# 6. COMPREHENSIVE VISUALIZATIONS
# ============================================================================

print("[6/8] Creating comprehensive visualizations...")

# Figure 1: Voltage Distribution Across All Buses
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Voltage magnitude violin plot
sns.violinplot(data=bus_data, x='bus', y='Vm', ax=axes[0], color='skyblue')
axes[0].axhline(y=1.05, color='r', linestyle='--', linewidth=2, label='Upper Limit')
axes[0].axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='Lower Limit')
axes[0].set_xlabel('Bus Number', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Voltage Magnitude (p.u.)', fontweight='bold', fontsize=12)
axes[0].set_title('Voltage Magnitude Distribution Across All Buses', fontweight='bold', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Voltage angle box plot
# sns.boxplot(data=bus_data, x='bus', y='Va', ax=axes[1], palette='Set2')
sns.boxplot(data=bus_data, x='bus', y='Va', ax=axes[1])
axes[1].set_xlabel('Bus Number', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Voltage Angle (degrees)', fontweight='bold', fontsize=12)
axes[1].set_title('Voltage Angle Distribution Across All Buses', fontweight='bold', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'voltage_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'voltage_distributions.pdf', bbox_inches='tight')
plt.close()

# Figure 2: Power Flow Distributions
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Active power demand
axes[0, 0].hist(bus_data['Pd'], bins=100, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].set_xlabel('Active Power Demand (MW)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Active Power Demand Distribution', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Reactive power demand
axes[0, 1].hist(bus_data['Qd'], bins=100, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].set_xlabel('Reactive Power Demand (MVAr)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Reactive Power Demand Distribution', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Active power generation
gen_pg = gen_data[gen_data['p_mw'] > 0]['p_mw']
axes[1, 0].hist(gen_pg, bins=100, alpha=0.7, color='orange', edgecolor='black')
axes[1, 0].set_xlabel('Active Power Generation (MW)', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Active Power Generation Distribution', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Reactive power generation
gen_qg = gen_data[gen_data['q_mvar'] != 0]['q_mvar']
axes[1, 1].hist(gen_qg, bins=100, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].set_xlabel('Reactive Power Generation (MVAr)', fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontweight='bold')
axes[1, 1].set_title('Reactive Power Generation Distribution', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'power_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'power_distributions.pdf', bbox_inches='tight')
plt.close()

# Figure 3: Branch Loading
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(branch_data['pf'], bins=100, alpha=0.7, color='teal', edgecolor='black')
axes[0].set_xlabel('Active Power Flow (MW)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('Branch Active Power Flow Distribution', fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].hist(branch_data['qf'], bins=100, alpha=0.7, color='coral', edgecolor='black')
axes[1].set_xlabel('Reactive Power Flow (MVAr)', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('Branch Reactive Power Flow Distribution', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'branch_loading.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'branch_loading.pdf', bbox_inches='tight')
plt.close()

print(f"   Created 3 comprehensive visualization sets")

# ============================================================================
# 7. ML MODEL PERFORMANCE EVALUATION
# ============================================================================

print("[7/8] Evaluating ML model performance...")

# Prepare test data (sample for speed)
n_test = min(5000, bus_data['scenario'].nunique())
test_scenarios = bus_data['scenario'].unique()[-n_test:]
test_data = bus_data[bus_data['scenario'].isin(test_scenarios)]

features_list = []
targets_list = []

for scenario in test_scenarios:
    scenario_data = test_data[test_data['scenario'] == scenario].sort_values('bus')
    features = np.concatenate([scenario_data['Pd'].values, scenario_data['Qd'].values])
    targets = np.concatenate([scenario_data['Vm'].values, scenario_data['Va'].values])
    features_list.append(features)
    targets_list.append(targets)

features = np.array(features_list)
targets = np.array(targets_list)

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_test = scaler_X.fit_transform(features)
y_test_scaled = scaler_y.fit_transform(targets)

# Predict
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test)
    predictions_scaled = model(X_test_tensor).numpy()

predictions = scaler_y.inverse_transform(predictions_scaled)

# Calculate errors
n_buses = 24
vm_pred = predictions[:, :n_buses]
va_pred = predictions[:, n_buses:]
vm_true = targets[:, :n_buses]
va_true = targets[:, n_buses:]

errors = {
    'Voltage_Magnitude': {
        'MAE': np.mean(np.abs(vm_pred - vm_true)),
        'RMSE': np.sqrt(np.mean((vm_pred - vm_true)**2)),
        'Max_Error': np.max(np.abs(vm_pred - vm_true)),
        'Median_Error': np.median(np.abs(vm_pred - vm_true)),
        '95th_Percentile': np.percentile(np.abs(vm_pred - vm_true), 95)
    },
    'Voltage_Angle': {
        'MAE': np.mean(np.abs(va_pred - va_true)),
        'RMSE': np.sqrt(np.mean((va_pred - va_true)**2)),
        'Max_Error': np.max(np.abs(va_pred - va_true)),
        'Median_Error': np.median(np.abs(va_pred - va_true)),
        '95th_Percentile': np.percentile(np.abs(va_pred - va_true), 95)
    }
}

error_df = pd.DataFrame(errors).T
error_df.to_csv(output_dir / 'ml_model_errors.csv')

# Measure inference time
import time
start = time.time()
with torch.no_grad():
    for _ in range(1000):
        _ = model(X_test_tensor[:1])
inference_time = (time.time() - start) / 1000

performance_metrics = {
    'Inference_Time_ms': inference_time * 1000,
    'Speedup_vs_AC_PF': runtime_data['ac'].mean() / inference_time,
    'Total_Parameters': sum(p.numel() for p in model.parameters())
}

with open(output_dir / 'ml_performance_metrics.json', 'w') as f:
    json.dump(performance_metrics, f, indent=2)

print(f"   ML inference time: {inference_time*1000:.4f} ms")
print(f"   Speedup vs AC-PF: {performance_metrics['Speedup_vs_AC_PF']:.0f}x")

# ============================================================================
# 8. GENERATE SUMMARY REPORT
# ============================================================================

print("[8/8] Generating summary report...")

report = f"""
{'='*70}
GRIDFM POWER FLOW ANALYSIS - SUMMARY REPORT
{'='*70}

DATASET OVERVIEW
{'-'*70}
Total Samples:              {stats_summary['Total Samples']:,}
Number of Scenarios:        {stats_summary['Number of Scenarios']:,}
Number of Buses:            {stats_summary['Number of Buses']}
Number of Generators:       {stats_summary['Number of Generators']}
Number of Branches:         {stats_summary['Number of Branches']}

VOLTAGE STATISTICS
{'-'*70}
Mean Voltage Magnitude:     {bus_data['Vm'].mean():.4f} p.u.
Std Voltage Magnitude:      {bus_data['Vm'].std():.4f} p.u.
Voltage Violations:         {((bus_data['Vm'] > 1.05) | (bus_data['Vm'] < 0.95)).sum() / len(bus_data) * 100:.2f}%

POWER STATISTICS
{'-'*70}
Total Active Load (avg):    {bus_data.groupby('scenario')['Pd'].sum().mean():.2f} MW
Total Reactive Load (avg):  {bus_data.groupby('scenario')['Qd'].sum().mean():.2f} MVAr
Total Generation (avg):     {gen_data.groupby('scenario')['p_mw'].sum().mean():.2f} MW

SOLVER PERFORMANCE
{'-'*70}
AC-PF Mean Runtime:         {runtime_data['ac'].mean():.4f} seconds
AC-PF Median Runtime:       {runtime_data['ac'].median():.4f} seconds

ML MODEL PERFORMANCE
{'-'*70}
Voltage Magnitude MAE:      {errors['Voltage_Magnitude']['MAE']:.6f} p.u. ({errors['Voltage_Magnitude']['MAE']*100:.3f}%)
Voltage Angle MAE:          {errors['Voltage_Angle']['MAE']:.4f} degrees
ML Inference Time:          {inference_time*1000:.4f} ms
Speedup vs Traditional:     {performance_metrics['Speedup_vs_AC_PF']:.0f}x

OUTPUT FILES GENERATED
{'-'*70}
All results saved to: {output_dir}/

CSV Files:
  - dataset_statistics.csv
  - voltage_statistics_by_bus.csv
  - power_statistics_by_bus.csv
  - generator_statistics.csv
  - branch_statistics.csv
  - runtime_statistics.csv
  - ml_model_errors.csv

Plots (PNG and PDF):
  - voltage_distributions
  - power_distributions
  - branch_loading

JSON Files:
  - dataset_statistics.json
  - ml_performance_metrics.json

{'='*70}
"""

with open(output_dir / 'SUMMARY_REPORT.txt', 'w') as f:
    f.write(report)

print(report)
print(f"\n✅ All results generated successfully!")
print(f"📁 Results saved to: {output_dir.absolute()}/")
print(f"\nKey files for your paper:")
print(f"  - SUMMARY_REPORT.txt (overview)")
print(f"  - voltage_distributions.pdf (Figure 1)")
print(f"  - power_distributions.pdf (Figure 2)")
print(f"  - ml_model_errors.csv (Table with model accuracy)")