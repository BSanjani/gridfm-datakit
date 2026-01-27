"""
Apply Droop Control to Existing GridFM Data
This script adds droop control parameters and calculations to already-generated power flow data
"""

import pandas as pd
import yaml
from pathlib import Path
from droop_extension import add_droop_parameters, calculate_droop_response, save_droop_data

print("="*70)
print("APPLYING DROOP CONTROL TO GRIDFM DATA")
print("="*70)

# ============================================================================
# 1. LOAD CONFIGURATION
# ============================================================================

print("\n[1/5] Loading configuration...")
with open('user_config2.yaml', 'r') as f:
    config = yaml.safe_load(f)

droop_config = config['droop_control']
print(f"Droop control enabled: {droop_config['enable']}")
print(f"Droop type: {droop_config['type']}")
print(f"R_p range: {droop_config['R_p_range']}")
print(f"R_q range: {droop_config['R_q_range']}")
print(f"Nominal frequency: {droop_config['f_nominal']} Hz")

# ============================================================================
# 2. LOAD EXISTING DATA
# ============================================================================

print("\n[2/5] Loading existing power flow data...")
data_dir = Path('data_out/case24_ieee_rts/raw')

bus_data = pd.read_parquet(data_dir / 'bus_data.parquet')
gen_data = pd.read_parquet(data_dir / 'gen_data.parquet')

print(f"Loaded {len(bus_data):,} bus records")
print(f"Loaded {len(gen_data):,} generator records")
print(f"Number of scenarios: {bus_data['scenario'].nunique():,}")
print(f"Number of generators: {gen_data['idx'].nunique()}")

# ============================================================================
# 3. ADD DROOP PARAMETERS
# ============================================================================

print("\n[3/5] Adding droop parameters to generators...")
gen_data_with_droop = add_droop_parameters(gen_data, droop_config)

# Show sample
print("\nSample of generator data with droop:")
sample_cols = ['idx', 'bus', 'p_mw', 'q_mvar', 'R_p', 'R_q', 'f_nominal']
print(gen_data_with_droop[sample_cols].head(10))

# ============================================================================
# 4. CALCULATE DROOP RESPONSE
# ============================================================================

print("\n[4/5] Calculating droop response...")
print("Note: Using sample scenarios for demonstration (first 100)")

# Use sample of scenarios for speed
sample_scenarios = bus_data['scenario'].unique()[:100]
bus_sample = bus_data[bus_data['scenario'].isin(sample_scenarios)]
gen_sample = gen_data_with_droop[gen_data_with_droop['scenario'].isin(sample_scenarios)]

gen_with_droop_response = calculate_droop_response(gen_sample, bus_sample, droop_config)

# Show the effect of droop
print("\nDroop Response Summary:")
print(f"Original P range: {gen_sample['p_mw'].min():.2f} - {gen_sample['p_mw'].max():.2f} MW")
if 'p_mw_droop' in gen_with_droop_response.columns:
    print(f"Droop P range:    {gen_with_droop_response['p_mw_droop'].min():.2f} - {gen_with_droop_response['p_mw_droop'].max():.2f} MW")
print(f"Original Q range: {gen_sample['q_mvar'].min():.2f} - {gen_sample['q_mvar'].max():.2f} MVAr")
if 'q_mvar_droop' in gen_with_droop_response.columns:
    print(f"Droop Q range:    {gen_with_droop_response['q_mvar_droop'].min():.2f} - {gen_with_droop_response['q_mvar_droop'].max():.2f} MVAr")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

print("\n[5/5] Saving results...")

# Create output directory
output_dir = Path('data_out/case24_ieee_rts/droop_control')
output_dir.mkdir(exist_ok=True)

# Save full generator data with droop parameters
gen_data_with_droop.to_parquet(output_dir / 'gen_data_with_droop_params.parquet')
print(f"Saved: {output_dir / 'gen_data_with_droop_params.parquet'}")

# Save sample with droop response
gen_with_droop_response.to_parquet(output_dir / 'gen_data_droop_response_sample.parquet')
print(f"Saved: {output_dir / 'gen_data_droop_response_sample.parquet'}")

# Save droop statistics
droop_stats = gen_data_with_droop.groupby('idx').agg({
    'R_p': 'first',
    'R_q': 'first',
    'p_mw': ['mean', 'std', 'min', 'max'],
    'q_mvar': ['mean', 'std', 'min', 'max']
}).round(4)

droop_stats.to_csv(output_dir / 'droop_parameters_by_generator.csv')
print(f"Saved: {output_dir / 'droop_parameters_by_generator.csv'}")

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

print("\n[6/5] Creating visualizations...")

import matplotlib.pyplot as plt
import seaborn as sns

# Plot 1: Droop parameter distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(gen_data_with_droop.drop_duplicates('idx')['R_p'], 
             bins=20, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_xlabel('Frequency Droop Coefficient R_p', fontweight='bold')
axes[0].set_ylabel('Number of Generators', fontweight='bold')
axes[0].set_title('Distribution of Frequency Droop Parameters', fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].hist(gen_data_with_droop.drop_duplicates('idx')['R_q'], 
             bins=20, alpha=0.7, color='green', edgecolor='black')
axes[1].set_xlabel('Voltage Droop Coefficient R_q', fontweight='bold')
axes[1].set_ylabel('Number of Generators', fontweight='bold')
axes[1].set_title('Distribution of Voltage Droop Parameters', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'droop_parameters_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'droop_parameters_distribution.png'}")

# Plot 2: Droop response (if calculated)
if 'delta_p_droop' in gen_with_droop_response.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(gen_with_droop_response['p_mw'], 
                   gen_with_droop_response['p_mw_droop'], 
                   alpha=0.5, s=10)
    axes[0].plot([gen_with_droop_response['p_mw'].min(), gen_with_droop_response['p_mw'].max()],
                [gen_with_droop_response['p_mw'].min(), gen_with_droop_response['p_mw'].max()],
                'r--', linewidth=2, label='No Droop')
    axes[0].set_xlabel('Original P (MW)', fontweight='bold')
    axes[0].set_ylabel('P with Droop (MW)', fontweight='bold')
    axes[0].set_title('Active Power: Original vs Droop-Controlled', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(gen_with_droop_response['q_mvar'], 
                   gen_with_droop_response['q_mvar_droop'], 
                   alpha=0.5, s=10)
    axes[1].plot([gen_with_droop_response['q_mvar'].min(), gen_with_droop_response['q_mvar'].max()],
                [gen_with_droop_response['q_mvar'].min(), gen_with_droop_response['q_mvar'].max()],
                'r--', linewidth=2, label='No Droop')
    axes[1].set_xlabel('Original Q (MVAr)', fontweight='bold')
    axes[1].set_ylabel('Q with Droop (MVAr)', fontweight='bold')
    axes[1].set_title('Reactive Power: Original vs Droop-Controlled', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'droop_response_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'droop_response_comparison.png'}")

plt.show()

print("\n" + "="*70)
print("✅ DROOP CONTROL APPLICATION COMPLETE!")
print("="*70)
print(f"\nResults saved to: {output_dir.absolute()}")
print("\nGenerated files:")
print("  - gen_data_with_droop_params.parquet (full data with droop parameters)")
print("  - gen_data_droop_response_sample.parquet (sample with droop response)")
print("  - droop_parameters_by_generator.csv (droop coefficients per generator)")
print("  - droop_parameters_distribution.png (visualization)")
print("  - droop_response_comparison.png (before/after comparison)")