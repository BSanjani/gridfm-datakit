import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Paths to your data directories
with_deadband_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_WithDeadband_10k\case24_ieee_rts\raw")
without_deadband_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_WithoutDeadband_10k\case24_ieee_rts\raw")
without_droop_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_Withoutdroop_10k\case24_ieee_rts\raw")

def diagnose_runtime_data(directory):
    """Diagnostic function to examine runtime data structure"""
    print(f"\n{'='*90}")
    print(f"DIAGNOSING RUNTIME DATA: {directory.name}")
    print(f"{'='*90}")
    
    runtime_file = directory / 'runtime_data.parquet'
    if not runtime_file.exists():
        print("✗ runtime_data.parquet not found!")
        return
    
    df = pd.read_parquet(runtime_file)
    print(f"\n✓ File loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Data types:\n{df.dtypes}")
    print(f"\n  First 10 rows:")
    print(df.head(10))
    print(f"\n  Summary statistics:")
    print(df.describe())
    
    # Check for all possible frequency-related columns
    freq_keywords = ['freq', 'df', 'f', 'delta', 'deviation', 'hz']
    freq_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in freq_keywords)]
    
    if freq_cols:
        print(f"\n  Frequency-related columns found: {freq_cols}")
        for col in freq_cols:
            print(f"\n  Column '{col}':")
            print(f"    - Type: {df[col].dtype}")
            print(f"    - Non-null: {df[col].notna().sum()}/{len(df)}")
            print(f"    - Unique values: {df[col].nunique()}")
            print(f"    - Range: [{df[col].min()}, {df[col].max()}]")
            print(f"    - Sample values: {df[col].head(10).tolist()}")
    else:
        print(f"\n  ⚠ No frequency-related columns found!")
        print(f"  All available columns: {list(df.columns)}")
    
    print(f"\n{'='*90}\n")

def load_parquet_data(directory):
    """Load all parquet files from a directory"""
    data = {}
    
    print(f"Loading data from: {directory}")
    
    # Load each parquet file including runtime_data for frequency
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
    gen_df = data_dict.get('gen', pd.DataFrame())
    
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
        'max_p_mw': 'first',  # These should be constant
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

def extract_frequency_statistics(data_dict):
    """Extract frequency deviation statistics from runtime data"""
    
    runtime_df = data_dict.get('runtime', pd.DataFrame())
    
    if runtime_df.empty:
        print("  ! No runtime data found for frequency extraction")
        return pd.DataFrame({'scenario': [], 'df': []})
    
    print(f"  Runtime data shape: {runtime_df.shape}")
    print(f"  Runtime data columns: {list(runtime_df.columns)}")
    
    # Show sample of data to understand structure
    if len(runtime_df) > 0:
        print(f"  First few rows of runtime data:")
        print(runtime_df.head())
        print(f"  Data types: {runtime_df.dtypes.to_dict()}")
    
    # Look for frequency deviation column (might be 'df', 'freq_deviation', 'delta_f', etc.)
    freq_col = None
    for col in ['df', 'freq_deviation', 'delta_f', 'frequency_deviation', 'f_deviation', 'f', 'freq']:
        if col in runtime_df.columns:
            freq_col = col
            print(f"  ✓ Found frequency column: '{freq_col}'")
            # Check if the column has non-zero values
            non_zero = (runtime_df[freq_col] != 0).sum()
            print(f"    - Non-zero values: {non_zero}/{len(runtime_df)}")
            print(f"    - Value range: [{runtime_df[freq_col].min()}, {runtime_df[freq_col].max()}]")
            print(f"    - Mean: {runtime_df[freq_col].mean()}, Std: {runtime_df[freq_col].std()}")
            break
    
    if freq_col is None:
        print("  ! Could not find frequency deviation column in runtime data")
        print(f"  Available columns: {list(runtime_df.columns)}")
        return pd.DataFrame({'scenario': [], 'df': []})
    
    # Extract frequency per scenario
    if 'scenario' in runtime_df.columns:
        freq_stats = runtime_df.groupby('scenario')[freq_col].first().reset_index()
        freq_stats.columns = ['scenario', 'df']
        print(f"  ✓ Extracted frequency deviation for {len(freq_stats)} scenarios")
        print(f"    - Frequency stats: mean={freq_stats['df'].mean():.6f}, std={freq_stats['df'].std():.6f}")
        print(f"    - Range: [{freq_stats['df'].min():.6f}, {freq_stats['df'].max():.6f}]")
    else:
        freq_stats = pd.DataFrame({
            'scenario': range(len(runtime_df)),
            'df': runtime_df[freq_col].values
        })
        print(f"  ✓ Extracted frequency deviation for {len(freq_stats)} entries")
        print(f"    - Frequency stats: mean={freq_stats['df'].mean():.6f}, std={freq_stats['df'].std():.6f}")
    
    return freq_stats

def extract_branch_statistics(data_dict):
    """Extract statistics per branch across all scenarios"""
    
    branch_df = data_dict.get('branch', pd.DataFrame())
    
    if branch_df.empty or 'f_bus' not in branch_df.columns:
        return pd.DataFrame()
    
    # Create branch identifier
    if 'f_bus' in branch_df.columns and 't_bus' in branch_df.columns:
        branch_df['branch_id'] = branch_df['f_bus'].astype(str) + '-' + branch_df['t_bus'].astype(str)
        
        branch_stats = branch_df.groupby('branch_id').agg({
            'pf': ['mean', 'std', 'min', 'max'],
            'pt': ['mean', 'std', 'min', 'max'],
        }).reset_index()
        
        # Flatten column names
        branch_stats.columns = ['_'.join(col).strip('_') for col in branch_stats.columns]
        
        return branch_stats
    
    return pd.DataFrame()

def plot_bus_comparison(bus_with, bus_without, bus_without_droop, output_dir):
    """Create simplified bus-level comparison plots - voltage and generation only"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Bus-Level Comparison: With Deadband vs Without Deadband vs Without Droop\n(Error bars show min-max range across all scenarios)', 
                 fontsize=18, fontweight='bold')
    
    buses = sorted(bus_with['bus'].values)
    x = np.arange(len(buses))
    width = 0.25  # Reduced width to fit three bars
    
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

def plot_frequency_comparison(freq_with, freq_without, freq_without_droop, output_dir, deadband_limit=0.0006):
    """Create frequency deviation comparison plots - THE MOST IMPORTANT PLOTS"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Frequency Deviation Analysis: With Deadband vs Without Deadband vs Without Droop\n⭐ KEY METRIC FOR DROOP CONTROL EVALUATION ⭐', 
                 fontsize=18, fontweight='bold', color='#1a1a1a')
    
    df_with = freq_with['df'].values
    df_without = freq_without['df'].values
    df_without_droop = freq_without_droop['df'].values
    
    # 1. Overlaid histograms for all three
    ax = axes[0, 0]
    bins = np.linspace(min(df_with.min(), df_without.min(), df_without_droop.min()), 
                       max(df_with.max(), df_without.max(), df_without_droop.max()), 50)
    ax.hist(df_with, bins=bins, alpha=0.5, color='#2E86AB', label='With Deadband', edgecolor='black')
    ax.hist(df_without, bins=bins, alpha=0.5, color='#A23B72', label='Without Deadband', edgecolor='black')
    ax.hist(df_without_droop, bins=bins, alpha=0.5, color='#F18F01', label='Without Droop', edgecolor='black')
    ax.axvline(x=-deadband_limit, color='red', linestyle='--', linewidth=2.5)
    ax.axvline(x=deadband_limit, color='red', linestyle='--', linewidth=2.5, label=f'Deadband ±{deadband_limit}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Frequency Deviation df (p.u.)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Scenarios', fontsize=12, fontweight='bold')
    ax.set_title('Distribution Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Side-by-side comparison of statistics
    ax = axes[0, 1]
    categories = ['With\nDeadband', 'Without\nDeadband', 'Without\nDroop']
    means = [df_with.mean(), df_without.mean(), df_without_droop.mean()]
    stds = [df_with.std(), df_without.std(), df_without_droop.std()]
    
    x_pos = np.arange(len(categories))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, 
                  color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=deadband_limit, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Deadband ±{deadband_limit}')
    ax.axhline(y=-deadband_limit, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel('Frequency Deviation df (p.u.)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Frequency Deviation ± Std Dev', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.0001, f'{mean:.6f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Box plot comparison
    ax = axes[1, 0]
    box_data = [df_with, df_without, df_without_droop]
    bp = ax.boxplot(box_data, labels=categories, patch_artist=True,
                    widths=0.6, showfliers=True)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=deadband_limit, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=-deadband_limit, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel('Frequency Deviation df (p.u.)', fontsize=12, fontweight='bold')
    ax.set_title('Box Plot Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. CDF (Cumulative Distribution)
    ax = axes[1, 1]
    sorted_with = np.sort(df_with)
    sorted_without = np.sort(df_without)
    sorted_without_droop = np.sort(df_without_droop)
    cdf_with = np.arange(1, len(sorted_with) + 1) / len(sorted_with)
    cdf_without = np.arange(1, len(sorted_without) + 1) / len(sorted_without)
    cdf_without_droop = np.arange(1, len(sorted_without_droop) + 1) / len(sorted_without_droop)
    
    ax.plot(sorted_with, cdf_with, linewidth=2.5, color='#2E86AB', label='With Deadband')
    ax.plot(sorted_without, cdf_without, linewidth=2.5, color='#A23B72', label='Without Deadband')
    ax.plot(sorted_without_droop, cdf_without_droop, linewidth=2.5, color='#F18F01', label='Without Droop')
    ax.axvline(x=-deadband_limit, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=deadband_limit, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Deadband ±{deadband_limit}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Frequency Deviation df (p.u.)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved frequency comparison plot: {output_dir / 'frequency_comparison.png'}")
    
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
    ax.plot(totals_with['scenario'], totals_with['Pd'], 
            'o-', label='With Deadband', linewidth=1, markersize=3, alpha=0.7, color='#2E86AB')
    ax.plot(totals_without['scenario'], totals_without['Pd'], 
            's-', label='Without Deadband', linewidth=1, markersize=3, alpha=0.7, color='#A23B72')
    ax.plot(totals_without_droop['scenario'], totals_without_droop['Pd'], 
            '^-', label='Without Droop', linewidth=1, markersize=3, alpha=0.7, color='#F18F01')
    ax.set_xlabel('Scenario ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Active Load (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Total System Load', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. Total Generation
    ax = axes[0, 1]
    ax.plot(totals_with['scenario'], totals_with['Pg'], 
            'o-', label='With Deadband', linewidth=1, markersize=3, alpha=0.7, color='#2E86AB')
    ax.plot(totals_without['scenario'], totals_without['Pg'], 
            's-', label='Without Deadband', linewidth=1, markersize=3, alpha=0.7, color='#A23B72')
    ax.plot(totals_without_droop['scenario'], totals_without_droop['Pg'], 
            '^-', label='Without Droop', linewidth=1, markersize=3, alpha=0.7, color='#F18F01')
    ax.set_xlabel('Scenario ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Active Generation (MW)', fontsize=12, fontweight='bold')
    ax.set_title('Total System Generation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. Power Imbalance
    ax = axes[1, 0]
    ax.plot(totals_with['scenario'], totals_with['imbalance'], 
            'o-', label='With Deadband', linewidth=1, markersize=3, alpha=0.7, color='#2E86AB')
    ax.plot(totals_without['scenario'], totals_without['imbalance'], 
            's-', label='Without Deadband', linewidth=1, markersize=3, alpha=0.7, color='#A23B72')
    ax.plot(totals_without_droop['scenario'], totals_without_droop['imbalance'], 
            '^-', label='Without Droop', linewidth=1, markersize=3, alpha=0.7, color='#F18F01')
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
    
    # Create generator labels: "Gen X @ Bus Y"
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

def print_summary_stats(bus_with, bus_without, bus_without_droop, gen_with, gen_without, gen_without_droop):
    """Print summary statistics"""
    print("\n" + "="*110)
    print(" "*40 + "STATISTICAL SUMMARY")
    print("="*110)
    
    print("\n┌─ VOLTAGE STATISTICS " + "─"*87 + "┐")
    print(f"│                           With Deadband    Without Deadband   Without Droop      Diff (No DB)   Diff (No Droop) │")
    print(f"│ Mean Voltage (p.u.):      {bus_with['Vm_mean'].mean():>8.4f}         {bus_without['Vm_mean'].mean():>8.4f}         {bus_without_droop['Vm_mean'].mean():>8.4f}         {bus_with['Vm_mean'].mean() - bus_without['Vm_mean'].mean():>8.4f}       {bus_with['Vm_mean'].mean() - bus_without_droop['Vm_mean'].mean():>8.4f} │")
    print(f"│ Voltage Std Dev:          {bus_with['Vm_std'].mean():>8.4f}         {bus_without['Vm_std'].mean():>8.4f}         {bus_without_droop['Vm_std'].mean():>8.4f}         {bus_with['Vm_std'].mean() - bus_without['Vm_std'].mean():>8.4f}       {bus_with['Vm_std'].mean() - bus_without_droop['Vm_std'].mean():>8.4f} │")
    print(f"│ Min Voltage:              {bus_with['Vm_min'].min():>8.4f}         {bus_without['Vm_min'].min():>8.4f}         {bus_without_droop['Vm_min'].min():>8.4f}         {bus_with['Vm_min'].min() - bus_without['Vm_min'].min():>8.4f}       {bus_with['Vm_min'].min() - bus_without_droop['Vm_min'].min():>8.4f} │")
    print(f"│ Max Voltage:              {bus_with['Vm_max'].max():>8.4f}         {bus_without['Vm_max'].max():>8.4f}         {bus_without_droop['Vm_max'].max():>8.4f}         {bus_with['Vm_max'].max() - bus_without['Vm_max'].max():>8.4f}       {bus_with['Vm_max'].max() - bus_without_droop['Vm_max'].max():>8.4f} │")
    print("└" + "─"*108 + "┘")
    
    print("\n┌─ GENERATION STATISTICS " + "─"*84 + "┐")
    print(f"│                           With Deadband    Without Deadband   Without Droop      Diff (No DB)   Diff (No Droop) │")
    print(f"│ Avg Gen Output (MW):      {gen_with['p_mw_mean'].mean():>8.2f}         {gen_without['p_mw_mean'].mean():>8.2f}         {gen_without_droop['p_mw_mean'].mean():>8.2f}         {gen_with['p_mw_mean'].mean() - gen_without['p_mw_mean'].mean():>8.2f}       {gen_with['p_mw_mean'].mean() - gen_without_droop['p_mw_mean'].mean():>8.2f} │")
    print(f"│ Avg % at Pmax:            {gen_with['pct_at_pmax'].mean():>8.2f}         {gen_without['pct_at_pmax'].mean():>8.2f}         {gen_without_droop['pct_at_pmax'].mean():>8.2f}         {gen_with['pct_at_pmax'].mean() - gen_without['pct_at_pmax'].mean():>8.2f}       {gen_with['pct_at_pmax'].mean() - gen_without_droop['pct_at_pmax'].mean():>8.2f} │")
    print(f"│ Avg % at Pmin:            {gen_with['pct_at_pmin'].mean():>8.2f}         {gen_without['pct_at_pmin'].mean():>8.2f}         {gen_without_droop['pct_at_pmin'].mean():>8.2f}         {gen_with['pct_at_pmin'].mean() - gen_without['pct_at_pmin'].mean():>8.2f}       {gen_with['pct_at_pmin'].mean() - gen_without_droop['pct_at_pmin'].mean():>8.2f} │")
    print("└" + "─"*108 + "┘")
    
    print("\n┌─ LOAD STATISTICS " + "─"*90 + "┐")
    print(f"│                           With Deadband    Without Deadband   Without Droop      Diff (No DB)   Diff (No Droop) │")
    print(f"│ Total Load (MW):          {bus_with['Pd_mean'].sum():>8.2f}         {bus_without['Pd_mean'].sum():>8.2f}         {bus_without_droop['Pd_mean'].sum():>8.2f}         {bus_with['Pd_mean'].sum() - bus_without['Pd_mean'].sum():>8.2f}       {bus_with['Pd_mean'].sum() - bus_without_droop['Pd_mean'].sum():>8.2f} │")
    print(f"│ Total Gen (MW):           {bus_with['Pg_mean'].sum():>8.2f}         {bus_without['Pg_mean'].sum():>8.2f}         {bus_without_droop['Pg_mean'].sum():>8.2f}         {bus_with['Pg_mean'].sum() - bus_without['Pg_mean'].sum():>8.2f}       {bus_with['Pg_mean'].sum() - bus_without_droop['Pg_mean'].sum():>8.2f} │")
    imb_with = bus_with['Pg_mean'].sum() - bus_with['Pd_mean'].sum()
    imb_without = bus_without['Pg_mean'].sum() - bus_without['Pd_mean'].sum()
    imb_without_droop = bus_without_droop['Pg_mean'].sum() - bus_without_droop['Pd_mean'].sum()
    print(f"│ Imbalance (MW):           {imb_with:>8.2f}         {imb_without:>8.2f}         {imb_without_droop:>8.2f}         {imb_with - imb_without:>8.2f}       {imb_with - imb_without_droop:>8.2f} │")
    print("└" + "─"*108 + "┘")
    
    print("\n" + "="*110)

def main():
    print("\n" + "="*110)
    print(" "*35 + "DROOP CONTROL BUS-LEVEL ANALYSIS")
    print(" "*25 + "With Deadband vs Without Deadband vs Without Droop")
    print("="*110)
    
    # Option to run diagnostics first
    RUN_DIAGNOSTICS = True  # Set to True to see detailed runtime data structure
    
    if RUN_DIAGNOSTICS:
        print("\n[DIAGNOSTIC MODE] Examining runtime data structure...")
        diagnose_runtime_data(with_deadband_dir)
        diagnose_runtime_data(without_deadband_dir)
        diagnose_runtime_data(without_droop_dir)
        print("\nDiagnostics complete. Proceeding with analysis...\n")
    
    # Load data
    print("\n[Step 1/6] Loading Parquet Data...")
    print("\n• WITH DEADBAND:")
    data_with = load_parquet_data(with_deadband_dir)
    
    print("\n• WITHOUT DEADBAND:")
    data_without = load_parquet_data(without_deadband_dir)
    
    print("\n• WITHOUT DROOP:")
    data_without_droop = load_parquet_data(without_droop_dir)
    
    # Extract frequency statistics
    print("\n[Step 2/6] Extracting Frequency Deviation...")
    print("\n• Processing WITH DEADBAND:")
    freq_with = extract_frequency_statistics(data_with)
    
    print("\n• Processing WITHOUT DEADBAND:")
    freq_without = extract_frequency_statistics(data_without)
    
    print("\n• Processing WITHOUT DROOP:")
    freq_without_droop = extract_frequency_statistics(data_without_droop)
    
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
    print_summary_stats(bus_with, bus_without, bus_without_droop, gen_with, gen_without, gen_without_droop)
    
    # Print frequency statistics
    if not freq_with.empty and not freq_without.empty and not freq_without_droop.empty:
        print("\n" + "="*110)
        print(" "*40 + "FREQUENCY DEVIATION STATISTICS ⭐ KEY METRIC")
        print("="*110)
        deadband = 0.0006
        
        df_with = freq_with['df'].values
        df_without = freq_without['df'].values
        df_without_droop = freq_without_droop['df'].values
        
        inside_with = np.sum((df_with >= -deadband) & (df_with <= deadband))
        inside_without = np.sum((df_without >= -deadband) & (df_without <= deadband))
        inside_without_droop = np.sum((df_without_droop >= -deadband) & (df_without_droop <= deadband))
        
        print(f"\n{'Metric':<35} {'With Deadband':<20} {'Without Deadband':<20} {'Without Droop':<20}")
        print("-" * 110)
        print(f"{'Number of scenarios':<35} {len(df_with):<20} {len(df_without):<20} {len(df_without_droop):<20}")
        print(f"{'Mean df (p.u.)':<35} {df_with.mean():<20.6f} {df_without.mean():<20.6f} {df_without_droop.mean():<20.6f}")
        print(f"{'Std df (p.u.)':<35} {df_with.std():<20.6f} {df_without.std():<20.6f} {df_without_droop.std():<20.6f}")
        print(f"{'Min df (p.u.)':<35} {df_with.min():<20.6f} {df_without.min():<20.6f} {df_without_droop.min():<20.6f}")
        print(f"{'Max df (p.u.)':<35} {df_with.max():<20.6f} {df_without.max():<20.6f} {df_without_droop.max():<20.6f}")
        print(f"{'Scenarios inside ±{deadband}':<35} {inside_with} ({100*inside_with/len(df_with):.1f}%)"
              f"{'    ':<3} {inside_without} ({100*inside_without/len(df_without):.1f}%)"
              f"{'    ':<3} {inside_without_droop} ({100*inside_without_droop/len(df_without_droop):.1f}%)")
        print("=" * 110)
    
    # Create plots
    print("\n[Step 5/6] Generating Plots...")
    output_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Plot frequency comparison FIRST (most important)
    if not freq_with.empty and not freq_without.empty and not freq_without_droop.empty:
        plot_frequency_comparison(freq_with, freq_without, freq_without_droop, output_dir)
    else:
        print("  ! Skipping frequency plots - no frequency data available")
    
    # Plot system-wide totals (load, generation, imbalance)
    plot_system_totals(data_with, data_without, data_without_droop, output_dir)
    
    # Plot bus-level comparison (voltage and generation only)
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
    
    if not freq_with.empty:
        freq_with.to_csv(output_dir / 'freq_stats_with_deadband.csv', index=False)
    if not freq_without.empty:
        freq_without.to_csv(output_dir / 'freq_stats_without_deadband.csv', index=False)
    if not freq_without_droop.empty:
        freq_without_droop.to_csv(output_dir / 'freq_stats_without_droop.csv', index=False)
    
    print(f"✓ Saved CSV files to: {output_dir}")
    
    print("\n" + "="*110)
    print(" "*45 + "ANALYSIS COMPLETE!")
    print("="*110)
    print(f"\nResults Location: {output_dir}")
    print("Files Created:")
    print("  • frequency_comparison.png      - ⭐ KEY: Frequency deviation analysis (3-way comparison)")
    print("  • system_totals.png             - Total load, generation, imbalance vs scenarios (3-way)")
    print("  • bus_comparison.png            - Bus voltage and generation (6 plots, 3-way)")
    print("  • gen_comparison.png            - Generator utilization and saturation (3-way)")
    print("  • freq_stats_*.csv              - Frequency deviation data (all 3 cases)")
    print("  • bus_stats_*.csv               - Detailed bus statistics (all 3 cases)")
    print("  • gen_stats_*.csv               - Detailed generator statistics (all 3 cases)")
    print("\n" + "="*110 + "\n")
    
    plt.show()

if __name__ == "__main__":
    main()