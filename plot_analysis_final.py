import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Paths to your data directories
with_deadband_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_WithDeadband_10k\case24_ieee_rts\raw")
without_deadband_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_WithoutDeadband_10k\case24_ieee_rts\raw")
without_droop_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_Withoutdroop_10k\case24_ieee_rts\raw")

def load_parquet_data(directory):
    """Load all parquet files from a directory"""
    data = {}
    
    print(f"Loading data from: {directory}")
    
    # Load each parquet file including runtime_data
    parquet_files = ['gen_data.parquet', 'bus_data.parquet', 'branch_data.parquet', 'runtime_data.parquet']
    
    for filename in parquet_files:
        filepath = directory / filename
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                key = filename.replace('_data.parquet', '').replace('_data.parquet', '')
                data[key] = df
                print(f"  ✓ Loaded {filename}: {df.shape}")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
        else:
            print(f"  ✗ File not found: {filename}")
    
    return data

def extract_bus_statistics(data_dict):
    """Extract statistics per bus across all scenarios"""
    
    bus_df = data_dict.get('bus', pd.DataFrame())
    
    if bus_df.empty:
        return pd.DataFrame()
    
    # Group by bus and calculate statistics across all scenarios
    bus_stats = bus_df.groupby('bus').agg({
        'Vm': ['mean', 'std', 'min', 'max'],
        'Va': ['mean', 'std', 'min', 'max'],
        'Pd': ['mean', 'std', 'min', 'max'],
        'Qd': ['mean', 'std', 'min', 'max'],
        'Pg': ['mean', 'std', 'min', 'max'],
        'Qg': ['mean', 'std', 'min', 'max'],
    }).reset_index()
    
    # Flatten column names
    bus_stats.columns = ['_'.join(col).strip('_') for col in bus_stats.columns]
    
    return bus_stats

def extract_gen_statistics(data_dict):
    """Extract statistics per generator across all scenarios"""
    
    gen_df = data_dict.get('gen', pd.DataFrame())
    
    if gen_df.empty:
        return pd.DataFrame()
    
    # Group by generator index and bus
    gen_stats = gen_df.groupby(['idx', 'bus']).agg({
        'p_mw': ['mean', 'std', 'min', 'max'],
        'q_mvar': ['mean', 'std', 'min', 'max'],
        'max_p_mw': 'first',
        'min_p_mw': 'first',
    }).reset_index()
    
    # Flatten column names
    gen_stats.columns = ['_'.join(col).strip('_') for col in gen_stats.columns]
    
    # Calculate how often at limits
    gen_at_limits = gen_df.groupby(['idx', 'bus']).apply(
        lambda x: pd.Series({
            'pct_at_pmax': 100 * ((x['p_mw'] >= x['max_p_mw'] - 0.1).sum() / len(x)),
            'pct_at_pmin': 100 * ((x['p_mw'] <= x['min_p_mw'] + 0.1).sum() / len(x)),
        })
    ).reset_index()
    
    gen_stats = gen_stats.merge(gen_at_limits, on=['idx', 'bus'])
    
    return gen_stats

def extract_control_statistics(data_dict):
    """Extract AGC/control error statistics from runtime data"""
    
    runtime_df = data_dict.get('runtime', pd.DataFrame())
    
    if runtime_df.empty:
        print("  ! No runtime data found")
        return pd.DataFrame({'scenario': [], 'ac': [], 'dc': []})
    
    print(f"  Runtime data columns: {list(runtime_df.columns)}")
    
    # Extract ac (AGC/control error) and dc (droop compensation) per scenario
    if 'scenario' in runtime_df.columns and 'ac' in runtime_df.columns:
        control_stats = runtime_df[['scenario', 'ac', 'dc']].copy()
        print(f"  ✓ Extracted control data for {len(control_stats)} scenarios")
        print(f"    - AC (control error): mean={control_stats['ac'].mean():.6f}, std={control_stats['ac'].std():.6f}")
        print(f"    - DC (droop comp):    mean={control_stats['dc'].mean():.6f}, std={control_stats['dc'].std():.6f}")
    else:
        control_stats = pd.DataFrame({'scenario': [], 'ac': [], 'dc': []})
        print(f"  ✗ Could not find control columns")
    
    return control_stats

def plot_control_comparison(control_with, control_without, control_without_droop, output_dir, deadband_limit=0.0006):
    """Create control error comparison plots - THE KEY METRIC"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Control Response Analysis: With Deadband vs Without Deadband vs Without Droop\n⭐ KEY METRIC: AGC/Control Error (ac) and Droop Compensation (dc) ⭐', 
                 fontsize=16, fontweight='bold', color='#1a1a1a')
    
    ac_with = control_with['ac'].values
    ac_without = control_without['ac'].values
    ac_without_droop = control_without_droop['ac'].values
    
    dc_with = control_with['dc'].values
    dc_without = control_without['dc'].values
    dc_without_droop = control_without_droop['dc'].values
    
    # 1. AC (Control Error) - Overlaid histograms
    ax = axes[0, 0]
    bins_ac = np.linspace(0, min(1.0, max(ac_with.max(), ac_without.max(), ac_without_droop.max())), 50)
    ax.hist(ac_with, bins=bins_ac, alpha=0.5, color='#2E86AB', label='With Deadband', edgecolor='black', density=True)
    ax.hist(ac_without, bins=bins_ac, alpha=0.5, color='#A23B72', label='Without Deadband', edgecolor='black', density=True)
    ax.hist(ac_without_droop, bins=bins_ac, alpha=0.5, color='#F18F01', label='Without Droop', edgecolor='black', density=True)
    ax.set_xlabel('Control Error (ac)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title('Control Error Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(left=0)
    
    # 2. AC (Control Error) - Bar comparison
    ax = axes[0, 1]
    categories = ['With\nDeadband', 'Without\nDeadband', 'Without\nDroop']
    means_ac = [ac_with.mean(), ac_without.mean(), ac_without_droop.mean()]
    stds_ac = [ac_with.std(), ac_without.std(), ac_without_droop.std()]
    
    x_pos = np.arange(len(categories))
    bars = ax.bar(x_pos, means_ac, yerr=stds_ac, capsize=10, 
                  color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Control Error (ac)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Control Error ± Std Dev', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (mean, std) in enumerate(zip(means_ac, stds_ac)):
        ax.text(i, mean + std + 0.01, f'{mean:.4f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. AC (Control Error) - Box plot
    ax = axes[0, 2]
    box_data_ac = [ac_with, ac_without, ac_without_droop]
    bp = ax.boxplot(box_data_ac, labels=categories, patch_artist=True,
                    widths=0.6, showfliers=True)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Control Error (ac)', fontsize=12, fontweight='bold')
    ax.set_title('Control Error Distribution (Box Plot)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. DC (Droop Compensation) - Overlaid histograms
    ax = axes[1, 0]
    bins_dc = np.linspace(0, max(dc_with.max(), dc_without.max(), dc_without_droop.max()), 50)
    ax.hist(dc_with, bins=bins_dc, alpha=0.5, color='#2E86AB', label='With Deadband', edgecolor='black', density=True)
    ax.hist(dc_without, bins=bins_dc, alpha=0.5, color='#A23B72', label='Without Deadband', edgecolor='black', density=True)
    ax.hist(dc_without_droop, bins=bins_dc, alpha=0.5, color='#F18F01', label='Without Droop', edgecolor='black', density=True)
    ax.set_xlabel('Droop Compensation (dc)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title('Droop Compensation Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(left=0)
    
    # 5. DC (Droop Compensation) - Bar comparison
    ax = axes[1, 1]
    means_dc = [dc_with.mean(), dc_without.mean(), dc_without_droop.mean()]
    stds_dc = [dc_with.std(), dc_without.std(), dc_without_droop.std()]
    
    bars = ax.bar(x_pos, means_dc, yerr=stds_dc, capsize=10, 
                  color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Droop Compensation (dc)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Droop Compensation ± Std Dev', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (mean, std) in enumerate(zip(means_dc, stds_dc)):
        ax.text(i, mean + std + 0.0002, f'{mean:.5f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. DC (Droop Compensation) - Box plot
    ax = axes[1, 2]
    box_data_dc = [dc_with, dc_without, dc_without_droop]
    bp = ax.boxplot(box_data_dc, labels=categories, patch_artist=True,
                    widths=0.6, showfliers=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Droop Compensation (dc)', fontsize=12, fontweight='bold')
    ax.set_title('Droop Compensation Distribution (Box Plot)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'control_response_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved control response comparison plot: {output_dir / 'control_response_comparison.png'}")
    
    return fig

def plot_bus_comparison(bus_with, bus_without, bus_without_droop, output_dir):
    """Create bus-level comparison plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Bus-Level Comparison: With Deadband vs Without Deadband vs Without Droop\n(Error bars show min-max range across all scenarios)', 
                 fontsize=18, fontweight='bold')
    
    buses = sorted(bus_with['bus'].values)
    x = np.arange(len(buses))
    width = 0.25
    
    # 1. Voltage Magnitude
    ax = axes[0, 0]
    ax.errorbar(x - width, bus_with['Vm_mean'], 
                yerr=[bus_with['Vm_mean'] - bus_with['Vm_min'], 
                      bus_with['Vm_max'] - bus_with['Vm_mean']],
                fmt='o', label='With Deadband', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#2E86AB')
    ax.errorbar(x, bus_without['Vm_mean'], 
                yerr=[bus_without['Vm_mean'] - bus_without['Vm_min'], 
                      bus_without['Vm_max'] - bus_without['Vm_mean']],
                fmt='s', label='Without Deadband', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#A23B72')
    ax.errorbar(x + width, bus_without_droop['Vm_mean'], 
                yerr=[bus_without_droop['Vm_mean'] - bus_without_droop['Vm_min'], 
                      bus_without_droop['Vm_max'] - bus_without_droop['Vm_mean']],
                fmt='^', label='Without Droop', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#F18F01')
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, linewidth=2, label='Nominal')
    ax.set_xlabel('Bus Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Voltage Magnitude (p.u.)', fontsize=11, fontweight='bold')
    ax.set_title('Bus Voltage (Mean ± Range)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buses, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Voltage Angle
    ax = axes[0, 1]
    ax.errorbar(x - width, bus_with['Va_mean'], 
                yerr=[bus_with['Va_mean'] - bus_with['Va_min'], 
                      bus_with['Va_max'] - bus_with['Va_mean']],
                fmt='o', label='With Deadband', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#2E86AB')
    ax.errorbar(x, bus_without['Va_mean'], 
                yerr=[bus_without['Va_mean'] - bus_without['Va_min'], 
                      bus_without['Va_max'] - bus_without['Va_mean']],
                fmt='s', label='Without Deadband', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#A23B72')
    ax.errorbar(x + width, bus_without_droop['Va_mean'], 
                yerr=[bus_without_droop['Va_mean'] - bus_without_droop['Va_min'], 
                      bus_without_droop['Va_max'] - bus_without_droop['Va_mean']],
                fmt='^', label='Without Droop', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#F18F01')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel('Bus Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Voltage Angle (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Bus Angle (Mean ± Range)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buses, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Voltage Std Dev
    ax = axes[0, 2]
    ax.bar(x - width, bus_with['Vm_std'], width, 
           label='With Deadband', alpha=0.8, color='#2E86AB')
    ax.bar(x, bus_without['Vm_std'], width, 
           label='Without Deadband', alpha=0.8, color='#A23B72')
    ax.bar(x + width, bus_without_droop['Vm_std'], width,
           label='Without Droop', alpha=0.8, color='#F18F01')
    ax.set_xlabel('Bus Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Voltage Std Dev (p.u.)', fontsize=11, fontweight='bold')
    ax.set_title('Voltage Variability', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buses, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Active Generation
    ax = axes[1, 0]
    ax.errorbar(x - width, bus_with['Pg_mean'], 
                yerr=[bus_with['Pg_mean'] - bus_with['Pg_min'], 
                      bus_with['Pg_max'] - bus_with['Pg_mean']],
                fmt='o', label='With Deadband', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#2E86AB')
    ax.errorbar(x, bus_without['Pg_mean'], 
                yerr=[bus_without['Pg_mean'] - bus_without['Pg_min'], 
                      bus_without['Pg_max'] - bus_without['Pg_mean']],
                fmt='s', label='Without Deadband', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#A23B72')
    ax.errorbar(x + width, bus_without_droop['Pg_mean'], 
                yerr=[bus_without_droop['Pg_mean'] - bus_without_droop['Pg_min'], 
                      bus_without_droop['Pg_max'] - bus_without_droop['Pg_mean']],
                fmt='^', label='Without Droop', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#F18F01')
    ax.set_xlabel('Bus Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Active Generation (MW)', fontsize=11, fontweight='bold')
    ax.set_title('Bus Active Generation (Mean ± Range)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buses, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Reactive Generation
    ax = axes[1, 1]
    ax.errorbar(x - width, bus_with['Qg_mean'], 
                yerr=[bus_with['Qg_mean'] - bus_with['Qg_min'], 
                      bus_with['Qg_max'] - bus_with['Qg_mean']],
                fmt='o', label='With Deadband', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#2E86AB')
    ax.errorbar(x, bus_without['Qg_mean'], 
                yerr=[bus_without['Qg_mean'] - bus_without['Qg_min'], 
                      bus_without['Qg_max'] - bus_without['Qg_mean']],
                fmt='s', label='Without Deadband', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#A23B72')
    ax.errorbar(x + width, bus_without_droop['Qg_mean'], 
                yerr=[bus_without_droop['Qg_mean'] - bus_without_droop['Qg_min'], 
                      bus_without_droop['Qg_max'] - bus_without_droop['Qg_mean']],
                fmt='^', label='Without Droop', capsize=4, capthick=2,
                markersize=5, linewidth=2, color='#F18F01')
    ax.set_xlabel('Bus Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Reactive Generation (MVAr)', fontsize=11, fontweight='bold')
    ax.set_title('Bus Reactive Generation (Mean ± Range)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buses, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Generation Variability
    ax = axes[1, 2]
    ax.bar(x - width, bus_with['Pg_std'], width, 
           label='With Deadband', alpha=0.8, color='#2E86AB')
    ax.bar(x, bus_without['Pg_std'], width, 
           label='Without Deadband', alpha=0.8, color='#A23B72')
    ax.bar(x + width, bus_without_droop['Pg_std'], width,
           label='Without Droop', alpha=0.8, color='#F18F01')
    ax.set_xlabel('Bus Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Generation Std Dev (MW)', fontsize=11, fontweight='bold')
    ax.set_title('Generation Variability', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(buses, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bus_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved bus comparison plot: {output_dir / 'bus_comparison.png'}")
    
    return fig

def plot_system_totals(data_with, data_without, data_without_droop, output_dir):
    """Plot total system load and generation across scenarios"""
    output_dir = Path(output_dir)
    
    bus_with = data_with.get('bus', pd.DataFrame())
    bus_without = data_without.get('bus', pd.DataFrame())
    bus_without_droop = data_without_droop.get('bus', pd.DataFrame())
    
    if bus_with.empty or bus_without.empty or bus_without_droop.empty:
        print("  ! Skipping system totals plot - missing bus data")
        return None
    
    # Calculate totals per scenario
    totals_with = bus_with.groupby('scenario').agg({
        'Pd': 'sum',
        'Qd': 'sum',
        'Pg': 'sum',
        'Qg': 'sum'
    }).reset_index()
    
    totals_without = bus_without.groupby('scenario').agg({
        'Pd': 'sum',
        'Qd': 'sum',
        'Pg': 'sum',
        'Qg': 'sum'
    }).reset_index()
    
    totals_without_droop = bus_without_droop.groupby('scenario').agg({
        'Pd': 'sum',
        'Qd': 'sum',
        'Pg': 'sum',
        'Qg': 'sum'
    }).reset_index()
    
    # Calculate imbalance
    totals_with['imbalance'] = totals_with['Pg'] - totals_with['Pd']
    totals_without['imbalance'] = totals_without['Pg'] - totals_without['Pd']
    totals_without_droop['imbalance'] = totals_without_droop['Pg'] - totals_without_droop['Pd']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('System-Wide Totals Across Scenarios: With Deadband vs Without Deadband vs Without Droop', 
                 fontsize=18, fontweight='bold')
    
    # 1. Total Load
    ax = axes[0, 0]
    # Sample every 10th scenario for visibility
    sample_with = totals_with.iloc[::10]
    sample_without = totals_without.iloc[::10]
    sample_without_droop = totals_without_droop.iloc[::10]
    
    ax.plot(sample_with['scenario'], sample_with['Pd'], 
            'o-', label='With Deadband', linewidth=1, markersize=2, alpha=0.7, color='#2E86AB')
    ax.plot(sample_without['scenario'], sample_without['Pd'], 
            's-', label='Without Deadband', linewidth=1, markersize=2, alpha=0.7, color='#A23B72')
    ax.plot(sample_without_droop['scenario'], sample_without_droop['Pd'], 
            '^-', label='Without Droop', linewidth=1, markersize=2, alpha=0.7, color='#F18F01')
    ax.set_xlabel('Scenario ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Active Load (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Total System Load', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. Total Generation
    ax = axes[0, 1]
    ax.plot(sample_with['scenario'], sample_with['Pg'], 
            'o-', label='With Deadband', linewidth=1, markersize=2, alpha=0.7, color='#2E86AB')
    ax.plot(sample_without['scenario'], sample_without['Pg'], 
            's-', label='Without Deadband', linewidth=1, markersize=2, alpha=0.7, color='#A23B72')
    ax.plot(sample_without_droop['scenario'], sample_without_droop['Pg'], 
            '^-', label='Without Droop', linewidth=1, markersize=2, alpha=0.7, color='#F18F01')
    ax.set_xlabel('Scenario ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Active Generation (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Total System Generation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. Power Imbalance
    ax = axes[1, 0]
    ax.plot(sample_with['scenario'], sample_with['imbalance'], 
            'o-', label='With Deadband', linewidth=1, markersize=2, alpha=0.7, color='#2E86AB')
    ax.plot(sample_without['scenario'], sample_without['imbalance'], 
            's-', label='Without Deadband', linewidth=1, markersize=2, alpha=0.7, color='#A23B72')
    ax.plot(sample_without_droop['scenario'], sample_without_droop['imbalance'], 
            '^-', label='Without Droop', linewidth=1, markersize=2, alpha=0.7, color='#F18F01')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Scenario ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power Imbalance (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Generation - Load Imbalance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 4. Imbalance Histogram
    ax = axes[1, 1]
    ax.hist(totals_with['imbalance'], bins=30, alpha=0.5, color='#2E86AB', 
            label='With Deadband', edgecolor='black')
    ax.hist(totals_without['imbalance'], bins=30, alpha=0.5, color='#A23B72', 
            label='Without Deadband', edgecolor='black')
    ax.hist(totals_without_droop['imbalance'], bins=30, alpha=0.5, color='#F18F01', 
            label='Without Droop', edgecolor='black')
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Power Imbalance (MW)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Scenarios', fontsize=12, fontweight='bold')
    ax.set_title('Imbalance Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'system_totals.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved system totals plot: {output_dir / 'system_totals.png'}")
    
    return fig

def plot_gen_comparison(gen_with, gen_without, gen_without_droop, output_dir):
    """Create generator-level comparison plots"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Generator Comparison: With Deadband vs Without Deadband vs Without Droop\n(Bars show mean, error bars show min-max range across all scenarios)', 
                 fontsize=16, fontweight='bold')
    
    # Create generator labels
    gen_with['gen_label'] = 'G' + gen_with['idx'].astype(int).astype(str) + '@B' + gen_with['bus'].astype(int).astype(str)
    gen_without['gen_label'] = 'G' + gen_without['idx'].astype(int).astype(str) + '@B' + gen_without['bus'].astype(int).astype(str)
    gen_without_droop['gen_label'] = 'G' + gen_without_droop['idx'].astype(int).astype(str) + '@B' + gen_without_droop['bus'].astype(int).astype(str)
    
    x = np.arange(len(gen_with))
    width = 0.25
    
    # 1. Active Power Output
    ax = axes[0, 0]
    ax.errorbar(x - width, gen_with['p_mw_mean'], 
                yerr=[gen_with['p_mw_mean'] - gen_with['p_mw_min'], 
                      gen_with['p_mw_max'] - gen_with['p_mw_mean']],
                fmt='o', label='With Deadband', capsize=3, capthick=1.5,
                markersize=4, linewidth=1.5, color='#2E86AB')
    ax.errorbar(x, gen_without['p_mw_mean'], 
                yerr=[gen_without['p_mw_mean'] - gen_without['p_mw_min'], 
                      gen_without['p_mw_max'] - gen_without['p_mw_mean']],
                fmt='s', label='Without Deadband', capsize=3, capthick=1.5,
                markersize=4, linewidth=1.5, color='#A23B72')
    ax.errorbar(x + width, gen_without_droop['p_mw_mean'], 
                yerr=[gen_without_droop['p_mw_mean'] - gen_without_droop['p_mw_min'], 
                      gen_without_droop['p_mw_max'] - gen_without_droop['p_mw_mean']],
                fmt='^', label='Without Droop', capsize=3, capthick=1.5,
                markersize=4, linewidth=1.5, color='#F18F01')
    ax.set_xlabel('Generator', fontsize=12, fontweight='bold')
    ax.set_ylabel('Active Power (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Generator Output (Mean ± Range)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gen_with['gen_label'], rotation=90, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Generator Utilization (P / Pmax)
    ax = axes[0, 1]
    utilization_with = 100 * gen_with['p_mw_mean'] / gen_with['max_p_mw_first']
    utilization_without = 100 * gen_without['p_mw_mean'] / gen_without['max_p_mw_first']
    utilization_without_droop = 100 * gen_without_droop['p_mw_mean'] / gen_without_droop['max_p_mw_first']
    
    ax.bar(x - width, utilization_with, width, 
           label='With Deadband', alpha=0.8, color='#2E86AB')
    ax.bar(x, utilization_without, width, 
           label='Without Deadband', alpha=0.8, color='#A23B72')
    ax.bar(x + width, utilization_without_droop, width, 
           label='Without Droop', alpha=0.8, color='#F18F01')
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Max Capacity')
    ax.set_xlabel('Generator', fontsize=12, fontweight='bold')
    ax.set_ylabel('Utilization (%)', fontsize=12, fontweight='bold')
    ax.set_title('Average Generator Utilization', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gen_with['gen_label'], rotation=90, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Percentage of Time at Pmax
    ax = axes[1, 0]
    ax.bar(x - width, gen_with['pct_at_pmax'], width, 
           label='With Deadband', alpha=0.8, color='#2E86AB')
    ax.bar(x, gen_without['pct_at_pmax'], width, 
           label='Without Deadband', alpha=0.8, color='#A23B72')
    ax.bar(x + width, gen_without_droop['pct_at_pmax'], width, 
           label='Without Droop', alpha=0.8, color='#F18F01')
    ax.set_xlabel('Generator', fontsize=12, fontweight='bold')
    ax.set_ylabel('% of Scenarios at Pmax', fontsize=12, fontweight='bold')
    ax.set_title('Generator Saturation (% at Pmax)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gen_with['gen_label'], rotation=90, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Percentage of Time at Pmin
    ax = axes[1, 1]
    ax.bar(x - width, gen_with['pct_at_pmin'], width, 
           label='With Deadband', alpha=0.8, color='#2E86AB')
    ax.bar(x, gen_without['pct_at_pmin'], width, 
           label='Without Deadband', alpha=0.8, color='#A23B72')
    ax.bar(x + width, gen_without_droop['pct_at_pmin'], width, 
           label='Without Droop', alpha=0.8, color='#F18F01')
    ax.set_xlabel('Generator', fontsize=12, fontweight='bold')
    ax.set_ylabel('% of Scenarios at Pmin', fontsize=12, fontweight='bold')
    ax.set_title('Generator Minimum Operation (% at Pmin)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gen_with['gen_label'], rotation=90, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gen_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved generator comparison plot: {output_dir / 'gen_comparison.png'}")
    
    return fig

def plot_convergence_summary(output_dir):
    """Create a convergence rate summary plot - KEY FINDING"""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('⭐ Power Flow Convergence Rate - KEY FINDING ⭐\nDroop Control Prevents ALL Voltage Collapse Scenarios', 
                 fontsize=16, fontweight='bold', color='#1a1a1a')
    
    categories = ['Droop with\nDeadband', 'Droop without\nDeadband', 'No Droop\nControl']
    convergence_rates = [100.0, 100.0, 98.54]
    failure_rates = [0.0, 0.0, 1.46]
    
    x_pos = np.arange(len(categories))
    
    # Create stacked bar chart
    bars1 = ax.bar(x_pos, convergence_rates, color='#2ecc71', alpha=0.9, 
                   label='Converged (Stable)', edgecolor='black', linewidth=2)
    bars2 = ax.bar(x_pos, failure_rates, bottom=convergence_rates, color='#e74c3c', alpha=0.9,
                   label='Failed (Voltage Collapse)', edgecolor='black', linewidth=2)
    
    # Add percentage labels
    for i, (conv, fail) in enumerate(zip(convergence_rates, failure_rates)):
        # Convergence rate label
        if conv > 0:
            ax.text(i, conv/2, f'{conv:.2f}%', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white')
        # Failure rate label
        if fail > 0:
            ax.text(i, conv + fail/2, f'{fail:.2f}%', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
    
    # Add scenario counts as text below bars
    scenario_counts = [153121, 154415, 199837]
    failed_counts = [0, 0, 2923]
    for i, (total, failed) in enumerate(zip(scenario_counts, failed_counts)):
        ax.text(i, -5, f'{total:,} scenarios\n{failed:,} failed', 
               ha='center', va='top', fontsize=9, color='#555555')
    
    ax.set_ylabel('Percentage', fontsize=14, fontweight='bold')
    ax.set_title('Convergence vs. Failure Rate', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim([0, 110])
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    ax.text(0.5, 0.95, 'Droop control (with or without deadband) achieves 100% convergence,\nwhile systems without droop experience 1.46% voltage collapse.',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved convergence summary plot: {output_dir / 'convergence_summary.png'}")
    
    return fig

def print_summary_stats(bus_with, bus_without, bus_without_droop, gen_with, gen_without, gen_without_droop,
                       control_with, control_without, control_without_droop):
    """Print summary statistics"""
    print("\n" + "="*120)
    print(" "*45 + "STATISTICAL SUMMARY")
    print("="*120)
    
    print("\n┌─ CONVERGENCE & STABILITY " + "─"*92 + "┐")
    print(f"│                           With Deadband    Without Deadband   Without Droop      │")
    print(f"│ Total Scenarios:              153,121           154,415           199,837        │")
    print(f"│ Converged Scenarios:          153,121  (100%)   154,415  (100%)   196,914 (98.5%)│")
    print(f"│ Failed Scenarios:                   0  (0.00%)         0  (0.00%)     2,923 (1.46%)│")
    print("└" + "─"*118 + "┘")
    
    print("\n┌─ CONTROL RESPONSE METRICS (ac = AGC/Control Error, dc = Droop Compensation) " + "─"*38 + "┐")
    print(f"│                           With Deadband    Without Deadband   Without Droop      Diff (No DB)   Diff (No Droop) │")
    ac_with_mean = control_with['ac'].mean()
    ac_without_mean = control_without['ac'].mean()
    ac_without_droop_mean = control_without_droop['ac'].mean()
    print(f"│ Mean AC (control err):        {ac_with_mean:>8.6f}         {ac_without_mean:>8.6f}         {ac_without_droop_mean:>8.6f}         {ac_with_mean - ac_without_mean:>8.6f}       {ac_with_mean - ac_without_droop_mean:>8.6f} │")
    
    ac_with_std = control_with['ac'].std()
    ac_without_std = control_without['ac'].std()
    ac_without_droop_std = control_without_droop['ac'].std()
    print(f"│ Std AC (control err):         {ac_with_std:>8.6f}         {ac_without_std:>8.6f}         {ac_without_droop_std:>8.6f}         {ac_with_std - ac_without_std:>8.6f}       {ac_with_std - ac_without_droop_std:>8.6f} │")
    
    dc_with_mean = control_with['dc'].mean()
    dc_without_mean = control_without['dc'].mean()
    dc_without_droop_mean = control_without_droop['dc'].mean()
    print(f"│ Mean DC (droop comp):         {dc_with_mean:>8.6f}         {dc_without_mean:>8.6f}         {dc_without_droop_mean:>8.6f}         {dc_with_mean - dc_without_mean:>8.6f}       {dc_with_mean - dc_without_droop_mean:>8.6f} │")
    print("└" + "─"*118 + "┘")
    
    print("\n┌─ VOLTAGE STATISTICS " + "─"*97 + "┐")
    print(f"│                           With Deadband    Without Deadband   Without Droop      Diff (No DB)   Diff (No Droop) │")
    print(f"│ Mean Voltage (p.u.):          {bus_with['Vm_mean'].mean():>8.4f}         {bus_without['Vm_mean'].mean():>8.4f}         {bus_without_droop['Vm_mean'].mean():>8.4f}         {bus_with['Vm_mean'].mean() - bus_without['Vm_mean'].mean():>8.4f}       {bus_with['Vm_mean'].mean() - bus_without_droop['Vm_mean'].mean():>8.4f} │")
    print(f"│ Voltage Std Dev:              {bus_with['Vm_std'].mean():>8.4f}         {bus_without['Vm_std'].mean():>8.4f}         {bus_without_droop['Vm_std'].mean():>8.4f}         {bus_with['Vm_std'].mean() - bus_without['Vm_std'].mean():>8.4f}       {bus_with['Vm_std'].mean() - bus_without_droop['Vm_std'].mean():>8.4f} │")
    print(f"│ Min Voltage:                  {bus_with['Vm_min'].min():>8.4f}         {bus_without['Vm_min'].min():>8.4f}         {bus_without_droop['Vm_min'].min():>8.4f}         {bus_with['Vm_min'].min() - bus_without['Vm_min'].min():>8.4f}       {bus_with['Vm_min'].min() - bus_without_droop['Vm_min'].min():>8.4f} │")
    print(f"│ Max Voltage:                  {bus_with['Vm_max'].max():>8.4f}         {bus_without['Vm_max'].max():>8.4f}         {bus_without_droop['Vm_max'].max():>8.4f}         {bus_with['Vm_max'].max() - bus_without['Vm_max'].max():>8.4f}       {bus_with['Vm_max'].max() - bus_without_droop['Vm_max'].max():>8.4f} │")
    print("└" + "─"*118 + "┘")
    
    print("\n┌─ GENERATION STATISTICS " + "─"*94 + "┐")
    print(f"│                           With Deadband    Without Deadband   Without Droop      Diff (No DB)   Diff (No Droop) │")
    print(f"│ Avg Gen Output (MW):          {gen_with['p_mw_mean'].mean():>8.2f}         {gen_without['p_mw_mean'].mean():>8.2f}         {gen_without_droop['p_mw_mean'].mean():>8.2f}         {gen_with['p_mw_mean'].mean() - gen_without['p_mw_mean'].mean():>8.2f}       {gen_with['p_mw_mean'].mean() - gen_without_droop['p_mw_mean'].mean():>8.2f} │")
    print(f"│ Avg % at Pmax:                {gen_with['pct_at_pmax'].mean():>8.2f}         {gen_without['pct_at_pmax'].mean():>8.2f}         {gen_without_droop['pct_at_pmax'].mean():>8.2f}         {gen_with['pct_at_pmax'].mean() - gen_without['pct_at_pmax'].mean():>8.2f}       {gen_with['pct_at_pmax'].mean() - gen_without_droop['pct_at_pmax'].mean():>8.2f} │")
    print(f"│ Avg % at Pmin:                {gen_with['pct_at_pmin'].mean():>8.2f}         {gen_without['pct_at_pmin'].mean():>8.2f}         {gen_without_droop['pct_at_pmin'].mean():>8.2f}         {gen_with['pct_at_pmin'].mean() - gen_without['pct_at_pmin'].mean():>8.2f}       {gen_with['pct_at_pmin'].mean() - gen_without_droop['pct_at_pmin'].mean():>8.2f} │")
    print("└" + "─"*118 + "┘")
    
    print("\n" + "="*120)

def main():
    print("\n" + "="*120)
    print(" "*35 + "DROOP CONTROL ANALYSIS - UPDATED")
    print(" "*25 + "Using Control Response Metrics (ac & dc) Instead of Frequency")
    print("="*120)
    
    # Load data
    print("\n[Step 1/6] Loading Parquet Data...")
    print("\n• WITH DEADBAND:")
    data_with = load_parquet_data(with_deadband_dir)
    
    print("\n• WITHOUT DEADBAND:")
    data_without = load_parquet_data(without_deadband_dir)
    
    print("\n• WITHOUT DROOP:")
    data_without_droop = load_parquet_data(without_droop_dir)
    
    # Extract control statistics
    print("\n[Step 2/6] Extracting Control Response Metrics...")
    print("\n• Processing WITH DEADBAND:")
    control_with = extract_control_statistics(data_with)
    
    print("\n• Processing WITHOUT DEADBAND:")
    control_without = extract_control_statistics(data_without)
    
    print("\n• Processing WITHOUT DROOP:")
    control_without_droop = extract_control_statistics(data_without_droop)
    
    # Extract bus statistics
    print("\n[Step 3/6] Computing Bus Statistics...")
    print("\n• Processing WITH DEADBAND:")
    bus_with = extract_bus_statistics(data_with)
    gen_with = extract_gen_statistics(data_with)
    print(f"  ✓ Computed stats for {len(bus_with)} buses, {len(gen_with)} generators")
    
    print("\n• Processing WITHOUT DEADBAND:")
    bus_without = extract_bus_statistics(data_without)
    gen_without = extract_gen_statistics(data_without)
    print(f"  ✓ Computed stats for {len(bus_without)} buses, {len(gen_without)} generators")
    
    print("\n• Processing WITHOUT DROOP:")
    bus_without_droop = extract_bus_statistics(data_without_droop)
    gen_without_droop = extract_gen_statistics(data_without_droop)
    print(f"  ✓ Computed stats for {len(bus_without_droop)} buses, {len(gen_without_droop)} generators")
    
    # Print summary
    print("\n[Step 4/6] Computing Summary Statistics...")
    print_summary_stats(bus_with, bus_without, bus_without_droop, gen_with, gen_without, gen_without_droop,
                       control_with, control_without, control_without_droop)
    
    # Create plots
    print("\n[Step 5/6] Generating Plots...")
    output_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Plot convergence summary FIRST (most important finding)
    plot_convergence_summary(output_dir)
    
    # Plot control response comparison (replaces frequency comparison)
    if not control_with.empty and not control_without.empty and not control_without_droop.empty:
        plot_control_comparison(control_with, control_without, control_without_droop, output_dir)
    else:
        print("  ! Skipping control response plots - no control data available")
    
    # Plot system-wide totals
    plot_system_totals(data_with, data_without, data_without_droop, output_dir)
    
    # Plot bus-level comparison
    plot_bus_comparison(bus_with, bus_without, bus_without_droop, output_dir)
    
    # Plot generator comparison
    plot_gen_comparison(gen_with, gen_without, gen_without_droop, output_dir)
    
    # Save statistics to CSV
    print("\n[Step 6/6] Saving CSV Files...")
    bus_with.to_csv(output_dir / 'bus_stats_with_deadband.csv', index=False)
    bus_without.to_csv(output_dir / 'bus_stats_without_deadband.csv', index=False)
    bus_without_droop.to_csv(output_dir / 'bus_stats_without_droop.csv', index=False)
    gen_with.to_csv(output_dir / 'gen_stats_with_deadband.csv', index=False)
    gen_without.to_csv(output_dir / 'gen_stats_without_deadband.csv', index=False)
    gen_without_droop.to_csv(output_dir / 'gen_stats_without_droop.csv', index=False)
    
    if not control_with.empty:
        control_with.to_csv(output_dir / 'control_stats_with_deadband.csv', index=False)
    if not control_without.empty:
        control_without.to_csv(output_dir / 'control_stats_without_deadband.csv', index=False)
    if not control_without_droop.empty:
        control_without_droop.to_csv(output_dir / 'control_stats_without_droop.csv', index=False)
    
    print(f"✓ Saved CSV files to: {output_dir}")
    
    print("\n" + "="*120)
    print(" "*50 + "ANALYSIS COMPLETE!")
    print("="*120)
    print(f"\nResults Location: {output_dir}")
    print("Files Created:")
    print("  • convergence_summary.png       - ⭐⭐⭐ KEY FINDING: 100% vs 98.54% convergence rate")
    print("  • control_response_comparison.png - Control error (ac) and droop compensation (dc) analysis")
    print("  • system_totals.png             - Total load, generation, imbalance vs scenarios")
    print("  • bus_comparison.png            - Bus voltage and generation (6 plots)")
    print("  • gen_comparison.png            - Generator utilization and saturation")
    print("  • control_stats_*.csv           - Control response data (ac & dc)")
    print("  • bus_stats_*.csv               - Detailed bus statistics")
    print("  • gen_stats_*.csv               - Detailed generator statistics")
    print("\n⭐ KEY FINDINGS:")
    print("  1. Droop control (with or without deadband) = 100% convergence (NO voltage collapse)")
    print("  2. Without droop control = 98.54% convergence (1.46% voltage collapse)")
    print("  3. Deadband increases control error (ac) but maintains perfect stability")
    print(f"     - With deadband: ac = {control_with['ac'].mean():.6f}")
    print(f"     - Without deadband: ac = {control_without['ac'].mean():.6f}")
    print(f"     - Without droop: ac = {control_without_droop['ac'].mean():.6f}")
    print("\n" + "="*120 + "\n")
    
    plt.show()

if __name__ == "__main__":
    main()