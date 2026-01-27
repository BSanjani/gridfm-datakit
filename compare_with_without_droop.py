"""
Side-by-Side Comparison: With vs Without Droop Control
Clear visualizations showing the impact of droop control
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*70)
print("SIDE-BY-SIDE COMPARISON: WITH vs WITHOUT DROOP")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/3] Loading data...")

# Original data (without droop)
bus_original = pd.read_parquet('data_out/case24_ieee_rts/raw/bus_data.parquet')
gen_original = pd.read_parquet('data_out/case24_ieee_rts/raw/gen_data.parquet')

# Droop data
gen_droop_response = pd.read_parquet('data_out/case24_ieee_rts/droop_control/gen_data_droop_response_sample.parquet')

# Get matching scenarios
sample_scenarios = gen_droop_response['scenario'].unique()
gen_original_sample = gen_original[gen_original['scenario'].isin(sample_scenarios)]
bus_sample = bus_original[bus_original['scenario'].isin(sample_scenarios)]

print(f"Comparing {len(sample_scenarios)} scenarios")

output_dir = Path('data_out/case24_ieee_rts/droop_control')

# ============================================================================
# FIGURE 1: ACTIVE POWER DISTRIBUTION
# ============================================================================

print("\n[2/3] Creating Figure 1: Active Power Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# WITHOUT DROOP
axes[0].hist(gen_original_sample['p_mw'], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_xlabel('Active Power (MW)', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[0].set_title('WITHOUT Droop Control', fontweight='bold', fontsize=14, color='darkblue')
axes[0].grid(True, alpha=0.3)
mean_p_original = gen_original_sample['p_mw'].mean()
axes[0].axvline(mean_p_original, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_p_original:.2f} MW')
axes[0].legend(fontsize=11)

# WITH DROOP
axes[1].hist(gen_droop_response['p_mw_droop'], bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
axes[1].set_xlabel('Active Power (MW)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[1].set_title('WITH Droop Control', fontweight='bold', fontsize=14, color='darkgreen')
axes[1].grid(True, alpha=0.3)
mean_p_droop = gen_droop_response['p_mw_droop'].mean()
axes[1].axvline(mean_p_droop, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_p_droop:.2f} MW')
axes[1].legend(fontsize=11)

plt.suptitle('Active Power Distribution: With vs Without Droop', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'comparison_active_power.png', dpi=300, bbox_inches='tight')
print(f"Saved: comparison_active_power.png")
plt.show()

# ============================================================================
# FIGURE 2: REACTIVE POWER DISTRIBUTION
# ============================================================================

print("\n[2/3] Creating Figure 2: Reactive Power Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# WITHOUT DROOP
axes[0].hist(gen_original_sample['q_mvar'], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_xlabel('Reactive Power (MVAr)', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[0].set_title('WITHOUT Droop Control', fontweight='bold', fontsize=14, color='darkblue')
axes[0].grid(True, alpha=0.3)
mean_q_original = gen_original_sample['q_mvar'].mean()
axes[0].axvline(mean_q_original, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_q_original:.2f} MVAr')
axes[0].legend(fontsize=11)

# WITH DROOP
axes[1].hist(gen_droop_response['q_mvar_droop'], bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
axes[1].set_xlabel('Reactive Power (MVAr)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[1].set_title('WITH Droop Control', fontweight='bold', fontsize=14, color='darkgreen')
axes[1].grid(True, alpha=0.3)
mean_q_droop = gen_droop_response['q_mvar_droop'].mean()
axes[1].axvline(mean_q_droop, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_q_droop:.2f} MVAr')
axes[1].legend(fontsize=11)

plt.suptitle('Reactive Power Distribution: With vs Without Droop', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'comparison_reactive_power.png', dpi=300, bbox_inches='tight')
print(f"Saved: comparison_reactive_power.png")
plt.show()

# ============================================================================
# FIGURE 3: TOTAL GENERATION PER SCENARIO
# ============================================================================

print("\n[2/3] Creating Figure 3: Total Generation per Scenario...")

# Calculate total generation per scenario
total_original = gen_original_sample.groupby('scenario').agg({
    'p_mw': 'sum',
    'q_mvar': 'sum'
}).reset_index()

total_droop = gen_droop_response.groupby('scenario').agg({
    'p_mw_droop': 'sum',
    'q_mvar_droop': 'sum'
}).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ACTIVE POWER - WITHOUT DROOP
axes[0, 0].plot(total_original['scenario'], total_original['p_mw'], 
               'b-', linewidth=1.5, alpha=0.7, label='Without Droop')
axes[0, 0].set_xlabel('Scenario', fontweight='bold', fontsize=11)
axes[0, 0].set_ylabel('Total Active Power (MW)', fontweight='bold', fontsize=11)
axes[0, 0].set_title('WITHOUT Droop Control', fontweight='bold', fontsize=13, color='darkblue')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)

# ACTIVE POWER - WITH DROOP
axes[0, 1].plot(total_droop['scenario'], total_droop['p_mw_droop'], 
               'g-', linewidth=1.5, alpha=0.7, label='With Droop')
axes[0, 1].set_xlabel('Scenario', fontweight='bold', fontsize=11)
axes[0, 1].set_ylabel('Total Active Power (MW)', fontweight='bold', fontsize=11)
axes[0, 1].set_title('WITH Droop Control', fontweight='bold', fontsize=13, color='darkgreen')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=10)

# REACTIVE POWER - WITHOUT DROOP
axes[1, 0].plot(total_original['scenario'], total_original['q_mvar'], 
               'b-', linewidth=1.5, alpha=0.7, label='Without Droop')
axes[1, 0].set_xlabel('Scenario', fontweight='bold', fontsize=11)
axes[1, 0].set_ylabel('Total Reactive Power (MVAr)', fontweight='bold', fontsize=11)
axes[1, 0].set_title('WITHOUT Droop Control', fontweight='bold', fontsize=13, color='darkblue')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)

# REACTIVE POWER - WITH DROOP
axes[1, 1].plot(total_droop['scenario'], total_droop['q_mvar_droop'], 
               'g-', linewidth=1.5, alpha=0.7, label='With Droop')
axes[1, 1].set_xlabel('Scenario', fontweight='bold', fontsize=11)
axes[1, 1].set_ylabel('Total Reactive Power (MVAr)', fontweight='bold', fontsize=11)
axes[1, 1].set_title('WITH Droop Control', fontweight='bold', fontsize=13, color='darkgreen')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=10)

plt.suptitle('Total Generation Across Scenarios: With vs Without Droop', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'comparison_total_generation.png', dpi=300, bbox_inches='tight')
print(f"Saved: comparison_total_generation.png")
plt.show()

# ============================================================================
# FIGURE 4: BOX PLOTS BY GENERATOR
# ============================================================================

print("\n[2/3] Creating Figure 4: Per-Generator Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# ACTIVE POWER BY GENERATOR
sns.boxplot(data=gen_original_sample, x='idx', y='p_mw', ax=axes[0], color='lightblue')
axes[0].set_xlabel('Generator ID', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Active Power (MW)', fontweight='bold', fontsize=12)
axes[0].set_title('WITHOUT Droop Control', fontweight='bold', fontsize=14, color='darkblue')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(data=gen_droop_response, x='idx', y='p_mw_droop', ax=axes[1], color='lightgreen')
axes[1].set_xlabel('Generator ID', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Active Power (MW)', fontweight='bold', fontsize=12)
axes[1].set_title('WITH Droop Control', fontweight='bold', fontsize=14, color='darkgreen')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('Active Power by Generator: With vs Without Droop', 
            fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / 'comparison_by_generator.png', dpi=300, bbox_inches='tight')
print(f"Saved: comparison_by_generator.png")
plt.show()

# ============================================================================
# FIGURE 5: OVERLAID COMPARISON
# ============================================================================

print("\n[3/3] Creating Figure 5: Overlaid Distributions...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ACTIVE POWER OVERLAY
axes[0].hist(gen_original_sample['p_mw'], bins=50, alpha=0.5, color='blue', 
            edgecolor='black', label='Without Droop')
axes[0].hist(gen_droop_response['p_mw_droop'], bins=50, alpha=0.5, color='green', 
            edgecolor='black', label='With Droop')
axes[0].set_xlabel('Active Power (MW)', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[0].set_title('Active Power Distribution Overlay', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# REACTIVE POWER OVERLAY
axes[1].hist(gen_original_sample['q_mvar'], bins=50, alpha=0.5, color='blue', 
            edgecolor='black', label='Without Droop')
axes[1].hist(gen_droop_response['q_mvar_droop'], bins=50, alpha=0.5, color='green', 
            edgecolor='black', label='With Droop')
axes[1].set_xlabel('Reactive Power (MVAr)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=12)
axes[1].set_title('Reactive Power Distribution Overlay', fontweight='bold', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Distribution Overlay: Direct Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'comparison_overlay.png', dpi=300, bbox_inches='tight')
print(f"Saved: comparison_overlay.png")
plt.show()

# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

summary = pd.DataFrame({
    'Metric': [
        'Active Power Mean (MW)',
        'Active Power Std (MW)',
        'Active Power Min (MW)',
        'Active Power Max (MW)',
        '',
        'Reactive Power Mean (MVAr)',
        'Reactive Power Std (MVAr)',
        'Reactive Power Min (MVAr)',
        'Reactive Power Max (MVAr)'
    ],
    'Without Droop': [
        gen_original_sample['p_mw'].mean(),
        gen_original_sample['p_mw'].std(),
        gen_original_sample['p_mw'].min(),
        gen_original_sample['p_mw'].max(),
        np.nan,
        gen_original_sample['q_mvar'].mean(),
        gen_original_sample['q_mvar'].std(),
        gen_original_sample['q_mvar'].min(),
        gen_original_sample['q_mvar'].max()
    ],
    'With Droop': [
        gen_droop_response['p_mw_droop'].mean(),
        gen_droop_response['p_mw_droop'].std(),
        gen_droop_response['p_mw_droop'].min(),
        gen_droop_response['p_mw_droop'].max(),
        np.nan,
        gen_droop_response['q_mvar_droop'].mean(),
        gen_droop_response['q_mvar_droop'].std(),
        gen_droop_response['q_mvar_droop'].min(),
        gen_droop_response['q_mvar_droop'].max()
    ]
})

summary['Difference'] = summary['With Droop'] - summary['Without Droop']
summary['% Change'] = (summary['Difference'] / summary['Without Droop'] * 100).fillna(0)

print(summary.to_string(index=False))

summary.to_csv(output_dir / 'summary_statistics_comparison.csv', index=False)

print("\n" + "="*70)
print("✅ ALL COMPARISONS COMPLETE!")
print("="*70)
print(f"\nGenerated files:")
print(f"  - comparison_active_power.png")
print(f"  - comparison_reactive_power.png")
print(f"  - comparison_total_generation.png")
print(f"  - comparison_by_generator.png")
print(f"  - comparison_overlay.png")
print(f"  - summary_statistics_comparison.csv")
print(f"\nAll saved to: {output_dir.absolute()}/")