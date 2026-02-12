"""
COMPREHENSIVE RESULTS ANALYSIS & VISUALIZATION
===============================================
Generates publication-quality plots for:
1. Voltage profiles across scenarios
2. Droop control parameter effects
3. AGC/Secondary control analysis
4. ML model performance comparison
5. Control sensitivity analysis
6. Frequency analysis
7. Error analysis by bus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*80)
print("COMPREHENSIVE RESULTS ANALYSIS")
print("Droop + AGC + ML Performance Visualization")
print("="*80)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# ============================================================================
# LOAD ALL DATA
# ============================================================================
print("\n[1/10] Loading data...")

data_dir = Path('./data_out_droop_secondary_varied_100/case24_ieee_rts/raw')

bus_data = pd.read_parquet(data_dir / 'bus_data.parquet')
gen_data = pd.read_parquet(data_dir / 'gen_data.parquet')
runtime_data = pd.read_parquet(data_dir / 'runtime_data.parquet')

n_scenarios = bus_data['load_scenario_idx'].nunique()
n_buses = bus_data['bus'].nunique()

print(f"✓ Loaded {n_scenarios} scenarios, {n_buses} buses")

# Extract control parameters
droop_params = gen_data.groupby('load_scenario_idx')[['mp_droop', 'mq_droop', 'K_I_secondary']].first()

# ============================================================================
# LOAD ML MODEL AND MAKE PREDICTIONS
# ============================================================================
print("\n[2/10] Loading ML model and making predictions...")

# Load scalers
with open('scaler_X_control_aware.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scaler_y_control_aware.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Prepare data
pd_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Pd', aggfunc='first')
qd_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Qd', aggfunc='first')
vm_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Vm', aggfunc='first')
va_pivot = bus_data.pivot_table(index='load_scenario_idx', columns='bus', values='Va', aggfunc='first')

control_params = gen_data.groupby('load_scenario_idx')[['mp_droop', 'mq_droop', 'K_I_secondary']].first()
control_params = control_params.loc[pd_pivot.index]

X = np.concatenate([
    pd_pivot.values,
    qd_pivot.values,
    control_params['mp_droop'].values.reshape(-1, 1),
    control_params['mq_droop'].values.reshape(-1, 1),
    control_params['K_I_secondary'].values.reshape(-1, 1)
], axis=1)

y = np.concatenate([vm_pivot.values, va_pivot.values], axis=1)

# Scale and predict
X_scaled = scaler_X.transform(X)

# Load model
device = torch.device('cpu')

class ControlAwarePowerFlowNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ControlAwarePowerFlowNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, output_size)
        )
    def forward(self, x):
        return self.network(x)

model = ControlAwarePowerFlowNN(51, 48).to(device)
checkpoint = torch.load('best_model_control_aware.pth', weights_only=False, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    predictions_scaled = model(X_tensor).numpy()

predictions = scaler_y.inverse_transform(predictions_scaled)

pred_vm = predictions[:, :24]
pred_va = predictions[:, 24:]
true_vm = y[:, :24]
true_va = y[:, 24:]

print(f"✓ ML predictions computed for {len(predictions)} scenarios")

# ============================================================================
# CREATE OUTPUT DIRECTORY
# ============================================================================
output_dir = Path('./comprehensive_analysis_results')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# PLOT 1: VOLTAGE MAGNITUDE ANALYSIS
# ============================================================================
print("\n[3/10] Creating voltage magnitude plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1: Voltage histogram
axes[0, 0].hist(bus_data['Vm'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(0.95, color='red', linestyle='--', linewidth=2, label='Lower limit (0.95 pu)')
axes[0, 0].axvline(1.05, color='red', linestyle='--', linewidth=2, label='Upper limit (1.05 pu)')
axes[0, 0].set_xlabel('Voltage Magnitude (pu)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('A) Voltage Distribution Across All Buses & Scenarios', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2: Per-bus voltage boxplot
bus_nums = sorted(bus_data['bus'].unique())
bus_voltage_data = [bus_data[bus_data['bus'] == b]['Vm'].values for b in bus_nums]
bp = axes[0, 1].boxplot(bus_voltage_data, labels=bus_nums, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
axes[0, 1].axhline(0.95, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
axes[0, 1].axhline(1.05, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
axes[0, 1].set_xlabel('Bus Number', fontweight='bold')
axes[0, 1].set_ylabel('Voltage Magnitude (pu)', fontweight='bold')
axes[0, 1].set_title('B) Voltage Range by Bus', fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 1.3: Voltage heatmap (scenarios vs buses)
vm_matrix = vm_pivot.values[:50]  # First 50 scenarios for visibility
im = axes[1, 0].imshow(vm_matrix, cmap='RdYlGn', aspect='auto', vmin=0.95, vmax=1.05)
axes[1, 0].set_xlabel('Bus Number', fontweight='bold')
axes[1, 0].set_ylabel('Scenario Index', fontweight='bold')
axes[1, 0].set_title('C) Voltage Heatmap (First 50 Scenarios)', fontweight='bold')
plt.colorbar(im, ax=axes[1, 0], label='Voltage (pu)')

# 1.4: Voltage vs Load correlation
system_load = bus_data.groupby('load_scenario_idx')['Pd'].sum()
mean_voltage = bus_data.groupby('load_scenario_idx')['Vm'].mean()
axes[1, 1].scatter(system_load, mean_voltage, s=30, alpha=0.6, color='darkblue')
axes[1, 1].set_xlabel('Total System Load (MW)', fontweight='bold')
axes[1, 1].set_ylabel('Mean Voltage (pu)', fontweight='bold')
axes[1, 1].set_title('D) System Voltage vs Load', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Add correlation coefficient
corr = np.corrcoef(system_load, mean_voltage)[0, 1]
axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=axes[1, 1].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / '01_voltage_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_voltage_analysis.png")

# ============================================================================
# PLOT 2: DROOP CONTROL PARAMETERS
# ============================================================================
print("\n[4/10] Creating droop control parameter plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 2.1: mp_droop distribution
axes[0, 0].hist(droop_params['mp_droop'], bins=25, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Active Power Droop (mp)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('A) Primary Droop - Active Power (mp)', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].axvline(droop_params['mp_droop'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0, 0].legend()

# 2.2: mq_droop distribution
axes[0, 1].hist(droop_params['mq_droop'], bins=25, color='darkorange', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Reactive Power Droop (mq)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('B) Primary Droop - Reactive Power (mq)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].axvline(droop_params['mq_droop'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0, 1].legend()

# 2.3: K_I_secondary distribution
axes[0, 2].hist(droop_params['K_I_secondary'], bins=25, color='green', edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('AGC Integral Gain (K_I)', fontweight='bold')
axes[0, 2].set_ylabel('Frequency', fontweight='bold')
axes[0, 2].set_title('C) Secondary Control (AGC)', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='y')
axes[0, 2].axvline(droop_params['K_I_secondary'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0, 2].legend()

# 2.4: mp vs Voltage
merged_mp = mean_voltage.to_frame().merge(droop_params['mp_droop'], left_index=True, right_index=True)
axes[1, 0].scatter(merged_mp['mp_droop'], merged_mp['Vm'], s=30, alpha=0.6, color='steelblue')
axes[1, 0].set_xlabel('mp_droop', fontweight='bold')
axes[1, 0].set_ylabel('Mean Voltage (pu)', fontweight='bold')
axes[1, 0].set_title('D) Voltage Sensitivity to mp', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
corr_mp = np.corrcoef(merged_mp['mp_droop'], merged_mp['Vm'])[0, 1]
axes[1, 0].text(0.05, 0.95, f'Corr: {corr_mp:.3f}', transform=axes[1, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2.5: mq vs Voltage
merged_mq = mean_voltage.to_frame().merge(droop_params['mq_droop'], left_index=True, right_index=True)
axes[1, 1].scatter(merged_mq['mq_droop'], merged_mq['Vm'], s=30, alpha=0.6, color='darkorange')
axes[1, 1].set_xlabel('mq_droop', fontweight='bold')
axes[1, 1].set_ylabel('Mean Voltage (pu)', fontweight='bold')
axes[1, 1].set_title('E) Voltage Sensitivity to mq', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
corr_mq = np.corrcoef(merged_mq['mq_droop'], merged_mq['Vm'])[0, 1]
axes[1, 1].text(0.05, 0.95, f'Corr: {corr_mq:.3f}', transform=axes[1, 1].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2.6: K_I vs Voltage
merged_ki = mean_voltage.to_frame().merge(droop_params['K_I_secondary'], left_index=True, right_index=True)
axes[1, 2].scatter(merged_ki['K_I_secondary'], merged_ki['Vm'], s=30, alpha=0.6, color='green')
axes[1, 2].set_xlabel('K_I_secondary', fontweight='bold')
axes[1, 2].set_ylabel('Mean Voltage (pu)', fontweight='bold')
axes[1, 2].set_title('F) Voltage Sensitivity to AGC', fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)
corr_ki = np.corrcoef(merged_ki['K_I_secondary'], merged_ki['Vm'])[0, 1]
axes[1, 2].text(0.05, 0.95, f'Corr: {corr_ki:.3f}', transform=axes[1, 2].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / '02_droop_control_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_droop_control_analysis.png")

# ============================================================================
# PLOT 3: ML MODEL PERFORMANCE
# ============================================================================
print("\n[5/10] Creating ML performance plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 3.1: Voltage prediction scatter
axes[0, 0].scatter(true_vm.flatten(), pred_vm.flatten(), s=2, alpha=0.5, color='steelblue')
axes[0, 0].plot([true_vm.min(), true_vm.max()], [true_vm.min(), true_vm.max()],
                'r--', linewidth=2, label='Perfect')
axes[0, 0].set_xlabel('True Voltage (pu)', fontweight='bold')
axes[0, 0].set_ylabel('Predicted Voltage (pu)', fontweight='bold')
mae = mean_absolute_error(true_vm.flatten(), pred_vm.flatten())
r2 = r2_score(true_vm.flatten(), pred_vm.flatten())
axes[0, 0].set_title(f'A) Voltage Predictions\n(MAE: {mae:.6f} pu, R²: {r2:.4f})', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 3.2: Voltage angle prediction scatter
axes[0, 1].scatter(true_va.flatten(), pred_va.flatten(), s=2, alpha=0.5, color='darkorange')
axes[0, 1].plot([true_va.min(), true_va.max()], [true_va.min(), true_va.max()],
                'r--', linewidth=2, label='Perfect')
axes[0, 1].set_xlabel('True Angle (deg)', fontweight='bold')
axes[0, 1].set_ylabel('Predicted Angle (deg)', fontweight='bold')
mae_va = mean_absolute_error(true_va.flatten(), pred_va.flatten())
r2_va = r2_score(true_va.flatten(), pred_va.flatten())
axes[0, 1].set_title(f'B) Angle Predictions\n(MAE: {mae_va:.4f}°, R²: {r2_va:.4f})', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3.3: Voltage error distribution
vm_errors = (pred_vm - true_vm).flatten() * 1000
axes[0, 2].hist(vm_errors, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('Voltage Error (×10⁻³ pu)', fontweight='bold')
axes[0, 2].set_ylabel('Frequency', fontweight='bold')
axes[0, 2].set_title(f'C) Voltage Error Distribution\n(μ={vm_errors.mean():.2f}, σ={vm_errors.std():.2f})', 
                     fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# 3.4: Per-bus prediction accuracy
bus_mae = []
for bus_idx in range(24):
    bus_mae.append(mean_absolute_error(true_vm[:, bus_idx], pred_vm[:, bus_idx]))
axes[1, 0].bar(range(24), np.array(bus_mae)*1000, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Bus Number', fontweight='bold')
axes[1, 0].set_ylabel('MAE (×10⁻³ pu)', fontweight='bold')
axes[1, 0].set_title('D) Prediction Error by Bus', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 3.5: Error vs Load
total_load_per_scenario = pd_pivot.sum(axis=1).values
mean_error_per_scenario = np.abs(pred_vm - true_vm).mean(axis=1)
axes[1, 1].scatter(total_load_per_scenario, mean_error_per_scenario*1000, s=20, alpha=0.6, color='purple')
axes[1, 1].set_xlabel('Total Load (MW)', fontweight='bold')
axes[1, 1].set_ylabel('Mean Absolute Error (×10⁻³ pu)', fontweight='bold')
axes[1, 1].set_title('E) Error vs Load Condition', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# 3.6: Error vs Control Parameters
mean_mp = control_params['mp_droop'].values
axes[1, 2].scatter(mean_mp, mean_error_per_scenario*1000, s=20, alpha=0.6, color='green')
axes[1, 2].set_xlabel('mp_droop', fontweight='bold')
axes[1, 2].set_ylabel('Mean Absolute Error (×10⁻³ pu)', fontweight='bold')
axes[1, 2].set_title('F) Error vs Droop Parameter', fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '03_ml_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_ml_performance.png")

# ============================================================================
# PLOT 4: CONTROL SENSITIVITY HEATMAPS
# ============================================================================
print("\n[6/10] Creating control sensitivity heatmaps...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Create 2D analysis
from scipy.stats import binned_statistic_2d

# 4.1: Load vs mp effect on voltage
load_vals = system_load.values
mp_vals = droop_params.loc[system_load.index, 'mp_droop'].values
vm_vals = mean_voltage.loc[system_load.index].values

stat, x_edge, y_edge, _ = binned_statistic_2d(load_vals, mp_vals, vm_vals, statistic='mean', bins=10)
im1 = axes[0].imshow(stat.T, origin='lower', cmap='viridis', aspect='auto')
axes[0].set_xlabel('Load Bins', fontweight='bold')
axes[0].set_ylabel('mp_droop Bins', fontweight='bold')
axes[0].set_title('A) Voltage Sensitivity: Load vs mp', fontweight='bold')
plt.colorbar(im1, ax=axes[0], label='Mean Voltage (pu)')

# 4.2: Load vs mq effect on voltage
mq_vals = droop_params.loc[system_load.index, 'mq_droop'].values
stat2, x_edge2, y_edge2, _ = binned_statistic_2d(load_vals, mq_vals, vm_vals, statistic='mean', bins=10)
im2 = axes[1].imshow(stat2.T, origin='lower', cmap='plasma', aspect='auto')
axes[1].set_xlabel('Load Bins', fontweight='bold')
axes[1].set_ylabel('mq_droop Bins', fontweight='bold')
axes[1].set_title('B) Voltage Sensitivity: Load vs mq', fontweight='bold')
plt.colorbar(im2, ax=axes[1], label='Mean Voltage (pu)')

# 4.3: Load vs K_I effect on voltage
ki_vals = droop_params.loc[system_load.index, 'K_I_secondary'].values
stat3, x_edge3, y_edge3, _ = binned_statistic_2d(load_vals, ki_vals, vm_vals, statistic='mean', bins=10)
im3 = axes[2].imshow(stat3.T, origin='lower', cmap='coolwarm', aspect='auto')
axes[2].set_xlabel('Load Bins', fontweight='bold')
axes[2].set_ylabel('K_I Bins', fontweight='bold')
axes[2].set_title('C) Voltage Sensitivity: Load vs AGC', fontweight='bold')
plt.colorbar(im3, ax=axes[2], label='Mean Voltage (pu)')

plt.tight_layout()
plt.savefig(output_dir / '04_control_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_control_sensitivity.png")

# ============================================================================
# PLOT 5: GENERATION-LOAD BALANCE
# ============================================================================
print("\n[7/10] Creating generation-load balance plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

system_gen = gen_data.groupby('load_scenario_idx')['p_mw'].sum()
system_losses = system_gen - system_load
loss_pct = (system_losses / system_gen) * 100

# 5.1: Gen vs Load
axes[0, 0].scatter(system_load, system_gen, s=30, alpha=0.6, color='green')
axes[0, 0].plot([system_load.min(), system_load.max()], 
                [system_load.min(), system_load.max()],
                'r--', linewidth=2, label='Perfect balance')
axes[0, 0].set_xlabel('Load (MW)', fontweight='bold')
axes[0, 0].set_ylabel('Generation (MW)', fontweight='bold')
axes[0, 0].set_title('A) Generation vs Load', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 5.2: System losses
axes[0, 1].hist(system_losses, bins=30, color='orange', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('System Losses (MW)', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title(f'B) System Losses\n(Mean: {system_losses.mean():.2f} MW)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 5.3: Loss percentage
axes[1, 0].hist(loss_pct, bins=30, color='red', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Loss Percentage (%)', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title(f'C) Loss Percentage\n(Mean: {loss_pct.mean():.2f}%)', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 5.4: Losses vs Load
axes[1, 1].scatter(system_load, loss_pct, s=30, alpha=0.6, color='purple')
axes[1, 1].set_xlabel('System Load (MW)', fontweight='bold')
axes[1, 1].set_ylabel('Loss Percentage (%)', fontweight='bold')
axes[1, 1].set_title('D) Losses vs System Load', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '05_generation_load_balance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_generation_load_balance.png")

# ============================================================================
# PLOT 6: FREQUENCY ANALYSIS (Estimated)
# ============================================================================
print("\n[8/10] Creating frequency analysis plots...")

# Estimate frequency deviation from droop equation: Δf ∝ -(ΔP) * mp
# Use generation deviation as proxy
gen_deviation = system_gen - system_gen.mean()
mp_per_scenario = droop_params.loc[system_gen.index, 'mp_droop']
freq_deviation_estimate = -gen_deviation * mp_per_scenario * 0.001  # Scale to Hz

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1: Frequency deviation distribution
axes[0, 0].hist(freq_deviation_estimate, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Estimated Frequency Deviation (Hz)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('A) Frequency Deviation Distribution', fontweight='bold')
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Nominal')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 6.2: Frequency vs Load
axes[0, 1].scatter(system_load, 60 + freq_deviation_estimate, s=30, alpha=0.6, color='darkblue')
axes[0, 1].axhline(60, color='red', linestyle='--', linewidth=2, label='Nominal (60 Hz)')
axes[0, 1].set_xlabel('System Load (MW)', fontweight='bold')
axes[0, 1].set_ylabel('Estimated System Frequency (Hz)', fontweight='bold')
axes[0, 1].set_title('B) Frequency vs Load (Droop Response)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 6.3: Frequency deviation vs mp
axes[1, 0].scatter(mp_per_scenario, freq_deviation_estimate, s=30, alpha=0.6, color='green')
axes[1, 0].set_xlabel('mp_droop', fontweight='bold')
axes[1, 0].set_ylabel('Frequency Deviation (mHz)', fontweight='bold')
axes[1, 0].set_title('C) Frequency Sensitivity to Droop', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 6.4: AGC impact (K_I vs frequency stability)
ki_per_scenario = droop_params.loc[system_gen.index, 'K_I_secondary']
freq_std = freq_deviation_estimate.abs()
axes[1, 1].scatter(ki_per_scenario, freq_std, s=30, alpha=0.6, color='purple')
axes[1, 1].set_xlabel('K_I_secondary (AGC Gain)', fontweight='bold')
axes[1, 1].set_ylabel('|Frequency Deviation| (mHz)', fontweight='bold')
axes[1, 1].set_title('D) AGC Effect on Frequency Stability', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '06_frequency_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_frequency_analysis.png")

# ============================================================================
# PLOT 7: COMPARATIVE ANALYSIS
# ============================================================================
print("\n[9/10] Creating comparative analysis plots...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Select 3 representative scenarios with different control params
sorted_by_mp = droop_params.sort_values('mp_droop')
low_mp_idx = sorted_by_mp.index[0]
mid_mp_idx = sorted_by_mp.index[len(sorted_by_mp)//2]
high_mp_idx = sorted_by_mp.index[-1]

scenarios_to_compare = [low_mp_idx, mid_mp_idx, high_mp_idx]
colors = ['blue', 'orange', 'green']
labels = [f'Low mp ({droop_params.loc[low_mp_idx, "mp_droop"]:.3f})',
          f'Med mp ({droop_params.loc[mid_mp_idx, "mp_droop"]:.3f})',
          f'High mp ({droop_params.loc[high_mp_idx, "mp_droop"]:.3f})']

# Voltage profiles
ax1 = fig.add_subplot(gs[0, :])
for idx, (scenario, color, label) in enumerate(zip(scenarios_to_compare, colors, labels)):
    scenario_idx = np.where(vm_pivot.index == scenario)[0][0]
    ax1.plot(range(24), true_vm[scenario_idx], '-o', color=color, label=label, linewidth=2, markersize=6)
ax1.set_xlabel('Bus Number', fontweight='bold')
ax1.set_ylabel('Voltage Magnitude (pu)', fontweight='bold')
ax1.set_title('Voltage Profiles for Different Droop Settings', fontweight='bold', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Load profiles
ax2 = fig.add_subplot(gs[1, :])
for idx, (scenario, color, label) in enumerate(zip(scenarios_to_compare, colors, labels)):
    scenario_idx = np.where(pd_pivot.index == scenario)[0][0]
    ax2.bar(np.arange(24) + idx*0.25, pd_pivot.iloc[scenario_idx], width=0.25, 
            color=color, alpha=0.7, label=label)
ax2.set_xlabel('Bus Number', fontweight='bold')
ax2.set_ylabel('Active Load (MW)', fontweight='bold')
ax2.set_title('Load Profiles for Compared Scenarios', fontweight='bold', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Control parameter comparison
ax3 = fig.add_subplot(gs[2, 0])
params_data = droop_params.loc[scenarios_to_compare, ['mp_droop', 'mq_droop', 'K_I_secondary']]
x_pos = np.arange(3)
width = 0.25
ax3.bar(x_pos - width, params_data['mp_droop']*100, width, label='mp×100', color='steelblue')
ax3.bar(x_pos, params_data['mq_droop']*100, width, label='mq×100', color='darkorange')
ax3.bar(x_pos + width, params_data['K_I_secondary'], width, label='K_I', color='green')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['Low', 'Med', 'High'])
ax3.set_ylabel('Parameter Value', fontweight='bold')
ax3.set_title('Control Parameters', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Generation comparison
ax4 = fig.add_subplot(gs[2, 1])
gen_vals = system_gen.loc[scenarios_to_compare]
ax4.bar(range(3), gen_vals, color=colors, alpha=0.7)
ax4.set_xticks(range(3))
ax4.set_xticklabels(['Low', 'Med', 'High'])
ax4.set_ylabel('Total Generation (MW)', fontweight='bold')
ax4.set_title('Total Generation', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Voltage statistics
ax5 = fig.add_subplot(gs[2, 2])
vm_stats = []
for scenario in scenarios_to_compare:
    scenario_idx = np.where(vm_pivot.index == scenario)[0][0]
    vm_stats.append([true_vm[scenario_idx].mean(), true_vm[scenario_idx].std(), 
                     true_vm[scenario_idx].min(), true_vm[scenario_idx].max()])
vm_stats = np.array(vm_stats)
x_pos = np.arange(3)
width = 0.2
ax5.bar(x_pos - width*1.5, vm_stats[:, 0], width, label='Mean', color='blue')
ax5.bar(x_pos - width*0.5, vm_stats[:, 1]*100, width, label='Std×100', color='orange')
ax5.bar(x_pos + width*0.5, vm_stats[:, 2], width, label='Min', color='red')
ax5.bar(x_pos + width*1.5, vm_stats[:, 3], width, label='Max', color='green')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(['Low', 'Med', 'High'])
ax5.set_ylabel('Voltage (pu)', fontweight='bold')
ax5.set_title('Voltage Statistics', fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis='y')

plt.savefig(output_dir / '07_comparative_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_comparative_analysis.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n[10/10] Creating summary report...")

summary_text = f"""
COMPREHENSIVE ANALYSIS SUMMARY
==============================

DATASET:
  • Scenarios: {n_scenarios}
  • Buses: {n_buses}
  • Total samples: {len(bus_data)}

CONTROL PARAMETERS:
  • mp_droop range: {droop_params['mp_droop'].min():.4f} - {droop_params['mp_droop'].max():.4f}
  • mq_droop range: {droop_params['mq_droop'].min():.4f} - {droop_params['mq_droop'].max():.4f}
  • K_I_secondary range: {droop_params['K_I_secondary'].min():.4f} - {droop_params['K_I_secondary'].max():.4f}

VOLTAGE PERFORMANCE:
  • Mean: {bus_data['Vm'].mean():.4f} pu
  • Std: {bus_data['Vm'].std():.4f} pu
  • Range: [{bus_data['Vm'].min():.4f}, {bus_data['Vm'].max():.4f}] pu

SYSTEM OPERATION:
  • Load range: [{system_load.min():.1f}, {system_load.max():.1f}] MW
  • Generation range: [{system_gen.min():.1f}, {system_gen.max():.1f}] MW
  • Mean losses: {system_losses.mean():.2f} MW ({loss_pct.mean():.2f}%)

ML MODEL PERFORMANCE:
  • Voltage MAE: {mae:.6f} pu ({mae*100:.4f}%)
  • Voltage R²: {r2:.6f}
  • Angle MAE: {mae_va:.4f}°
  • Angle R²: {r2_va:.6f}

GENERATED PLOTS:
  1. 01_voltage_analysis.png - Voltage distributions and correlations
  2. 02_droop_control_analysis.png - Control parameter effects
  3. 03_ml_performance.png - ML model accuracy
  4. 04_control_sensitivity.png - Sensitivity heatmaps
  5. 05_generation_load_balance.png - Power balance analysis
  6. 06_frequency_analysis.png - Frequency deviation analysis
  7. 07_comparative_analysis.png - Scenario comparisons
"""

with open(output_dir / 'ANALYSIS_SUMMARY.txt', 'w') as f:
    f.write(summary_text)

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {output_dir.absolute()}")
print("\n📊 Generated 7 comprehensive analysis plots:")
print("   ✓ 01_voltage_analysis.png")
print("   ✓ 02_droop_control_analysis.png")
print("   ✓ 03_ml_performance.png")
print("   ✓ 04_control_sensitivity.png")
print("   ✓ 05_generation_load_balance.png")
print("   ✓ 06_frequency_analysis.png")
print("   ✓ 07_comparative_analysis.png")
print("\n📄 Summary report: ANALYSIS_SUMMARY.txt")
print("\n" + "="*80)
print("✓ READY FOR PUBLICATION!")
print("="*80)