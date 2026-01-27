import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# ============================================================================
# AUTO-DETECT DATA DIRECTORY
# ============================================================================
def find_data_directory(base_dir_name):
    """Automatically find the correct data directory path"""
    base = Path(base_dir_name)
    
    # Try common patterns
    possible_paths = [
        base / 'raw',  # Direct: data_out_droop_publication/raw
        base / list(base.glob('case*'))[0].name / 'raw' if list(base.glob('case*')) else None,  # With case folder
    ]
    
    # Also check what actually exists
    if base.exists():
        print(f"\nFound base directory: {base}")
        print("Contents:")
        for item in base.iterdir():
            print(f"  - {item.name}")
            if item.is_dir():
                for subitem in item.iterdir():
                    print(f"      - {subitem.name}")
    
    for path in possible_paths:
        if path and path.exists():
            print(f"\n✓ Found data directory: {path}")
            return path
    
    return None

# Try to find the publication data
print("="*80)
print("SEARCHING FOR DATA DIRECTORY")
print("="*80)

data_dir = find_data_directory('data_out_droop_publication')

if data_dir is None:
    print("\n❌ Could not find data directory!")
    print("\nPlease check:")
    print("1. Did you run the data generation with config_droop_publication.yaml?")
    print("2. What is the actual directory name?")
    print("\nTo manually set the path, edit this script and set:")
    print("   data_dir = Path('your/actual/path/here')")
    sys.exit(1)

# Create output directory
output_dir = Path('./droop_plots_publication')
output_dir.mkdir(exist_ok=True)

# Load data
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

try:
    bus_data = pd.read_parquet(data_dir / 'bus_data.parquet')
    gen_data = pd.read_parquet(data_dir / 'gen_data.parquet')
    scenarios = pd.read_parquet(data_dir / 'scenarios_agg_load_profile.parquet')
    runtime_data = pd.read_parquet(data_dir / 'runtime_data.parquet')
    
    print(f"✓ Bus data:     {len(bus_data):>6} records")
    print(f"✓ Gen data:     {len(gen_data):>6} records")
    print(f"✓ Scenarios:    {len(scenarios):>6} records")
    print(f"✓ Runtime data: {len(runtime_data):>6} records")
except FileNotFoundError as e:
    print(f"\n❌ Error loading data: {e}")
    print(f"\nLooked in: {data_dir}")
    print("\nPlease verify the path and try again.")
    sys.exit(1)

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Check if frequency data exists
has_frequency = 'frequency_deviation' in runtime_data.columns or 'df' in runtime_data.columns

print("\n" + "="*80)
print("PREPARING ANALYSIS DATA")
print("="*80)

# Calculate total load per scenario
total_load = scenarios.groupby('load_scenario')['p_mw'].sum().reset_index()
total_load.columns = ['scenario', 'total_load_mw']

# Get total generation per scenario
gen_total = gen_data.groupby('scenario')[['p_mw', 'q_mvar']].sum().reset_index()
gen_total.columns = ['scenario', 'total_pg_mw', 'total_qg_mvar']

# Get voltage statistics per scenario
voltage_stats = bus_data.groupby('scenario').agg({
    'Vm': ['mean', 'std', 'min', 'max']
}).reset_index()
voltage_stats.columns = ['scenario', 'vm_mean', 'vm_std', 'vm_min', 'vm_max']

# Merge all scenario-level data
scenario_data = total_load.merge(gen_total, on='scenario')
scenario_data = scenario_data.merge(voltage_stats, on='scenario')

# Check for frequency data in runtime_data
if has_frequency:
    if 'frequency_deviation' in runtime_data.columns:
        freq_col = 'frequency_deviation'
    else:
        freq_col = 'df'
    
    freq_data = runtime_data.groupby('scenario')[[freq_col]].first().reset_index()
    freq_data.columns = ['scenario', 'freq_deviation']
    freq_data['system_frequency'] = 1.0 + freq_data['freq_deviation']
    scenario_data = scenario_data.merge(freq_data, on='scenario', how='left')

# Calculate losses
scenario_data['losses_mw'] = scenario_data['total_pg_mw'] - scenario_data['total_load_mw']
scenario_data['loss_percentage'] = (scenario_data['losses_mw'] / scenario_data['total_pg_mw']) * 100

# Per-bus voltage statistics
bus_voltage_stats = bus_data.groupby('bus').agg({
    'Vm': ['mean', 'std', 'min', 'max', 'count']
}).reset_index()
bus_voltage_stats.columns = ['bus', 'vm_mean', 'vm_std', 'vm_min', 'vm_max', 'count']
bus_voltage_stats = bus_voltage_stats.sort_values('bus')

print(f"✓ Created scenario analysis dataset: {len(scenario_data)} scenarios")
print(f"✓ Per-bus voltage statistics: {len(bus_voltage_stats)} buses")

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\n{'LOAD STATISTICS':-^80}")
print(f"  Mean load:        {scenario_data['total_load_mw'].mean():>10.2f} MW")
print(f"  Std deviation:    {scenario_data['total_load_mw'].std():>10.2f} MW")
print(f"  Min load:         {scenario_data['total_load_mw'].min():>10.2f} MW")
print(f"  Max load:         {scenario_data['total_load_mw'].max():>10.2f} MW")
print(f"  Range:            {scenario_data['total_load_mw'].max() - scenario_data['total_load_mw'].min():>10.2f} MW")

print(f"\n{'GENERATION STATISTICS':-^80}")
print(f"  Mean active power:     {scenario_data['total_pg_mw'].mean():>10.2f} MW")
print(f"  Std deviation:         {scenario_data['total_pg_mw'].std():>10.2f} MW")
print(f"  Min generation:        {scenario_data['total_pg_mw'].min():>10.2f} MW")
print(f"  Max generation:        {scenario_data['total_pg_mw'].max():>10.2f} MW")
print(f"  Mean reactive power:   {scenario_data['total_qg_mvar'].mean():>10.2f} MVAr")

print(f"\n{'LOSS STATISTICS':-^80}")
print(f"  Mean losses:           {scenario_data['losses_mw'].mean():>10.2f} MW")
print(f"  Mean loss percentage:  {scenario_data['loss_percentage'].mean():>10.3f} %")

print(f"\n{'VOLTAGE STATISTICS':-^80}")
print(f"  Mean voltage:          {bus_data['Vm'].mean():>10.4f} pu")
print(f"  Std deviation:         {bus_data['Vm'].std():>10.4f} pu")
print(f"  Min voltage:           {bus_data['Vm'].min():>10.4f} pu")
print(f"  Max voltage:           {bus_data['Vm'].max():>10.4f} pu")
print(f"  Voltage range:         {bus_data['Vm'].max() - bus_data['Vm'].min():>10.4f} pu")

if has_frequency:
    print(f"\n{'FREQUENCY STATISTICS':-^80}")
    print(f"  Mean frequency:        {scenario_data['system_frequency'].mean():>10.6f} pu")
    print(f"  Mean deviation:        {scenario_data['freq_deviation'].mean():>10.2e} pu")
    print(f"  Std deviation:         {scenario_data['freq_deviation'].std():>10.2e} pu")
    
    if len(scenario_data) > 1:
        slope_f = np.polyfit(scenario_data['total_load_mw'], scenario_data['system_frequency'], 1)[0]
        print(f"  Observed f-P droop:    {slope_f:>10.2e} pu/MW")

print(f"\n{'CONVERGENCE':-^80}")
if 'termination_status' in runtime_data.columns:
    status_counts = runtime_data['termination_status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status:<25}: {count:>6} ({count/len(runtime_data)*100:>5.1f}%)")

print("\n" + "="*80)
print("✓ Data loaded and analyzed successfully!")
print(f"✓ Output will be saved to: {output_dir.absolute()}")
print("="*80)