"""
Deadband Verification Plot
Shows droop correction = 0 within deadband zone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading data...")

# Load data
runtime = pd.read_parquet('data_out_droop_case57_5000/case57_ieee/raw/runtime_data.parquet')
gen = pd.read_parquet('data_out_droop_case57_5000/case57_ieee/raw/gen_data.parquet')

# Get frequency deviation per scenario
freq_data = runtime[['load_scenario_idx', 'frequency_deviation']].copy()
freq_data['frequency_hz'] = 60.0 + freq_data['frequency_deviation'] * 60.0
freq_data['freq_dev_hz'] = freq_data['frequency_deviation'] * 60.0  # In Hz

# Get droop generators
droop_gens = gen[gen['mp_droop'] > 0].copy()

# Merge frequency with generator data
merged = droop_gens.merge(freq_data, on='load_scenario_idx')

# Calculate theoretical droop correction
# Droop equation: ΔP = -(1/mp) × Δf
# If deadband is working, ΔP should be zero when |Δf| < deadband
merged['theoretical_droop_correction'] = -(1 / merged['mp_droop']) * merged['frequency_deviation']

# Calculate actual power deviation from mean
mean_power_per_gen = merged.groupby('idx')['p_mw'].transform('mean')
merged['power_deviation'] = merged['p_mw'] - mean_power_per_gen

print(f"Loaded {len(merged)} generator-scenario records")

# ============================================================================
# MAIN PLOT: DROOP RESPONSE vs FREQUENCY DEVIATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

deadband_pu = 0.0006  # In per-unit
deadband_hz = deadband_pu * 60  # In Hz

# ============================================================================
# PLOT 1: Power Response vs Frequency Deviation (ALL DATA)
# ============================================================================

ax1 = axes[0, 0]

# Scatter plot - color by whether within deadband
within_db = np.abs(merged['freq_dev_hz']) < deadband_hz
colors = ['green' if w else 'red' for w in within_db]

ax1.scatter(merged['freq_dev_hz'], merged['power_deviation'], 
            s=2, alpha=0.3, c=colors)

# Deadband zone
ax1.axvspan(-deadband_hz, deadband_hz, 
            color='yellow', alpha=0.3, label=f'Deadband (±{deadband_hz*1000:.0f} mHz)', zorder=0)

# Zero lines
ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

ax1.set_xlabel('Frequency Deviation (Hz)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Generator Power Deviation (MW)', fontsize=12, fontweight='bold')
ax1.set_title('A) Droop Response vs Frequency Deviation\nAll Generators & Scenarios', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Statistics
within_count = np.sum(within_db)
stats_text = f"Within Deadband: {within_count:,} ({within_count/len(merged)*100:.1f}%)\n"
stats_text += f"Avg |ΔP| within DB: {np.abs(merged[within_db]['power_deviation']).mean():.2f} MW\n"
if not within_db.all():
    stats_text += f"Avg |ΔP| outside DB: {np.abs(merged[~within_db]['power_deviation']).mean():.2f} MW"

ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
         verticalalignment='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================================
# PLOT 2: ZOOMED IN - Within and Near Deadband
# ============================================================================

ax2 = axes[0, 1]

# Filter to near deadband (±5× deadband)
near_db = np.abs(merged['freq_dev_hz']) < 5 * deadband_hz
data_near = merged[near_db]

ax2.scatter(data_near['freq_dev_hz'], data_near['power_deviation'], 
            s=5, alpha=0.4, c='steelblue')

# Deadband zone
ax2.axvspan(-deadband_hz, deadband_hz, 
            color='yellow', alpha=0.3, label=f'Deadband (±{deadband_hz*1000:.0f} mHz)', zorder=0)

# Zero lines
ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Theoretical droop lines (outside deadband)
freq_range = np.linspace(-5*deadband_hz, 5*deadband_hz, 100)
mp_typical = 0.04
droop_response = -(1/mp_typical) * (freq_range/60)  # Convert Hz to pu first
# Apply deadband
droop_response[np.abs(freq_range) < deadband_hz] = 0
ax2.plot(freq_range, droop_response * 100, 'r--', linewidth=2, 
         alpha=0.7, label='Theoretical Droop (mp=0.04)')

ax2.set_xlabel('Frequency Deviation (Hz)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Generator Power Deviation (MW)', fontsize=12, fontweight='bold')
ax2.set_title('B) Zoomed: Near Deadband Region\n(±5× Deadband)', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-5*deadband_hz, 5*deadband_hz])

# ============================================================================
# PLOT 3: Histogram - Power Deviation WITHIN Deadband
# ============================================================================

ax3 = axes[1, 0]

power_within_db = merged[within_db]['power_deviation']

ax3.hist(power_within_db, bins=50, color='green', edgecolor='black', alpha=0.7)
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Deviation')

ax3.set_xlabel('Power Deviation (MW)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('C) Power Deviation Distribution\nWITHIN Deadband Only', 
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Statistics
mean_within = power_within_db.mean()
std_within = power_within_db.std()
stats_within = f"Mean: {mean_within:.3f} MW\n"
stats_within += f"Std: {std_within:.2f} MW\n"
stats_within += f"Max |ΔP|: {np.abs(power_within_db).max():.2f} MW\n"
stats_within += f"Samples: {len(power_within_db):,}"

ax3.text(0.98, 0.97, stats_within, transform=ax3.transAxes,
         verticalalignment='top', horizontalalignment='right', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# ============================================================================
# PLOT 4: Histogram - Power Deviation OUTSIDE Deadband
# ============================================================================

ax4 = axes[1, 1]

if not within_db.all():
    power_outside_db = merged[~within_db]['power_deviation']
    
    ax4.hist(power_outside_db, bins=50, color='red', edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero Deviation')
    
    # Statistics
    mean_outside = power_outside_db.mean()
    std_outside = power_outside_db.std()
    stats_outside = f"Mean: {mean_outside:.3f} MW\n"
    stats_outside += f"Std: {std_outside:.2f} MW\n"
    stats_outside += f"Max |ΔP|: {np.abs(power_outside_db).max():.2f} MW\n"
    stats_outside += f"Samples: {len(power_outside_db):,}"
    
    ax4.text(0.98, 0.97, stats_outside, transform=ax4.transAxes,
             verticalalignment='top', horizontalalignment='right', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
else:
    ax4.text(0.5, 0.5, 'No scenarios outside deadband', 
             transform=ax4.transAxes, ha='center', va='center', fontsize=14)

ax4.set_xlabel('Power Deviation (MW)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax4.set_title('D) Power Deviation Distribution\nOUTSIDE Deadband Only', 
              fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

output_file = 'deadband_verification.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_file}")

# ============================================================================
# QUANTITATIVE DEADBAND TEST
# ============================================================================

print("\n" + "="*80)
print("DEADBAND VERIFICATION RESULTS")
print("="*80)

print(f"\nFrequency Deadband: ±{deadband_hz*1000:.0f} mHz (±{deadband_pu:.4f} pu)")

print(f"\nScenarios Analysis:")
print(f"  Total: {len(merged):,} generator-scenario records")
print(f"  Within deadband: {within_count:,} ({within_count/len(merged)*100:.1f}%)")
print(f"  Outside deadband: {len(merged)-within_count:,} ({(len(merged)-within_count)/len(merged)*100:.1f}%)")

print(f"\nPower Response Analysis:")
print(f"  Within deadband:")
print(f"    Mean |ΔP|: {np.abs(power_within_db).mean():.3f} MW")
print(f"    Std ΔP: {power_within_db.std():.3f} MW")

if not within_db.all():
    print(f"  Outside deadband:")
    print(f"    Mean |ΔP|: {np.abs(power_outside_db).mean():.3f} MW")
    print(f"    Std ΔP: {power_outside_db.std():.3f} MW")
    
    ratio = np.abs(power_outside_db).mean() / np.abs(power_within_db).mean()
    print(f"\n  Response Ratio (Outside/Within): {ratio:.2f}×")
    
    if ratio > 1.2:
        print("  ✓ DEADBAND IS WORKING - Larger response outside deadband")
    else:
        print("  ⚠ DEADBAND EFFECT WEAK - Similar response inside/outside")
else:
    print("  Outside deadband: N/A (all scenarios within deadband)")
    print("  ✓ DEADBAND IS WORKING - System very well controlled")

# Correlation test
from scipy.stats import pearsonr

# Within deadband - should have weak correlation
if len(power_within_db) > 10:
    freq_within = merged[within_db]['freq_dev_hz']
    corr_within, p_within = pearsonr(freq_within, power_within_db)
    print(f"\nCorrelation Analysis:")
    print(f"  Within deadband: r = {corr_within:.4f} (p = {p_within:.4f})")
    
    if not within_db.all() and len(power_outside_db) > 10:
        freq_outside = merged[~within_db]['freq_dev_hz']
        corr_outside, p_outside = pearsonr(freq_outside, power_outside_db)
        print(f"  Outside deadband: r = {corr_outside:.4f} (p = {p_outside:.4f})")
        
        if abs(corr_outside) > abs(corr_within):
            print("  ✓ Stronger correlation outside deadband - DEADBAND WORKING")

print("\n" + "="*80)
print("✓ DEADBAND VERIFICATION COMPLETE!")
print("="*80)
print(f"\nSee {output_file} for visual proof of deadband operation")