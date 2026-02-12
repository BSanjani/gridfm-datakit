import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# ============================================================================
# CONFIGURATION - Update this path to your data directory
# ============================================================================
# Based on auto-detection, the path should be:
data_dir = Path('./data_out_droop_publication/case24_ieee_rts/raw')

# Alternative paths if you have different data:
# data_dir = Path('./data_out_droop_final/case24_ieee_rts/raw')
# data_dir = Path('./data_out_droop_comprehensive/case24_ieee_rts_droop_comprehensive/raw')

# Create output directory
output_dir = Path('./droop_plots_publication')
output_dir.mkdir(exist_ok=True)

# Load data
print("="*80)
print("LOADING DATA")
print("="*80)
print(f"Data directory: {data_dir}")

if not data_dir.exists():
    print(f"\n❌ ERROR: Data directory not found: {data_dir}")
    print("\nPlease check the path and update line 17 in this script.")
    import sys
    sys.exit(1)

bus_data = pd.read_parquet(data_dir / 'bus_data.parquet')
gen_data = pd.read_parquet(data_dir / 'gen_data.parquet')
scenarios = pd.read_parquet(data_dir / 'scenarios_agg_load_profile.parquet')
runtime_data = pd.read_parquet(data_dir / 'runtime_data.parquet')

print(f"✓ Bus data:     {len(bus_data):>6} records")
print(f"✓ Gen data:     {len(gen_data):>6} records")
print(f"✓ Scenarios:    {len(scenarios):>6} records")
print(f"✓ Runtime data: {len(runtime_data):>6} records")

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
print(f"  Std deviation:         {scenario_data['losses_mw'].std():>10.2f} MW")
print(f"  Min losses:            {scenario_data['losses_mw'].min():>10.2f} MW")
print(f"  Max losses:            {scenario_data['losses_mw'].max():>10.2f} MW")
print(f"  Mean loss percentage:  {scenario_data['loss_percentage'].mean():>10.3f} %")

print(f"\n{'VOLTAGE STATISTICS (ALL BUSES, ALL SCENARIOS)':-^80}")
print(f"  Mean voltage:          {bus_data['Vm'].mean():>10.4f} pu")
print(f"  Std deviation:         {bus_data['Vm'].std():>10.4f} pu")
print(f"  Min voltage:           {bus_data['Vm'].min():>10.4f} pu")
print(f"  Max voltage:           {bus_data['Vm'].max():>10.4f} pu")
print(f"  Voltage range:         {bus_data['Vm'].max() - bus_data['Vm'].min():>10.4f} pu")

print(f"\n{'VOLTAGE STATISTICS (SCENARIO AVERAGES)':-^80}")
print(f"  Mean avg voltage:      {scenario_data['vm_mean'].mean():>10.4f} pu")
print(f"  Min avg voltage:       {scenario_data['vm_min'].min():>10.4f} pu")
print(f"  Max avg voltage:       {scenario_data['vm_max'].max():>10.4f} pu")

if has_frequency:
    print(f"\n{'FREQUENCY STATISTICS':-^80}")
    print(f"  Mean frequency:        {scenario_data['system_frequency'].mean():>10.6f} pu")
    print(f"  Mean deviation:        {scenario_data['freq_deviation'].mean():>10.2e} pu")
    print(f"  Std deviation:         {scenario_data['freq_deviation'].std():>10.2e} pu")
    print(f"  Min deviation:         {scenario_data['freq_deviation'].min():>10.2e} pu")
    print(f"  Max deviation:         {scenario_data['freq_deviation'].max():>10.2e} pu")
    
    # Calculate droop coefficient from data
    if len(scenario_data) > 1:
        slope_f = np.polyfit(scenario_data['total_load_mw'], scenario_data['system_frequency'], 1)[0]
        print(f"  Observed f-P droop:    {slope_f:>10.2e} pu/MW")
        print(f"  (Configured mp: 0.2)")

print(f"\n{'GENERATOR STATISTICS':-^80}")
print(f"  Number of generators:  {gen_data['idx'].nunique():>10}")
print(f"  Mean P per generator:  {gen_data['p_mw'].mean():>10.2f} MW")
print(f"  Mean Q per generator:  {gen_data['q_mvar'].mean():>10.2f} MVAr")
print(f"  Max P observed:        {gen_data['p_mw'].max():>10.2f} MW")
print(f"  Max Q observed:        {gen_data['q_mvar'].max():>10.2f} MVAr")

print(f"\n{'CONVERGENCE STATISTICS':-^80}")
if 'termination_status' in runtime_data.columns:
    status_counts = runtime_data['termination_status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status:<25}: {count:>6} ({count/len(runtime_data)*100:>5.1f}%)")

if 'solve_time' in runtime_data.columns:
    print(f"\n{'SOLVER PERFORMANCE':-^80}")
    print(f"  Mean solve time:       {runtime_data['solve_time'].mean():>10.4f} s")
    print(f"  Total time:            {runtime_data['solve_time'].sum():>10.2f} s")
    print(f"  Min solve time:        {runtime_data['solve_time'].min():>10.4f} s")
    print(f"  Max solve time:        {runtime_data['solve_time'].max():>10.4f} s")

# ============================================================================
# PLOT 1: Load Distribution
# ============================================================================
print("\n" + "="*80)
print("CREATING PLOTS")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(scenario_data['total_load_mw'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(scenario_data['total_load_mw'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scenario_data["total_load_mw"].mean():.1f} MW')
ax.set_xlabel('Total Load (MW)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Load Distribution Across Scenarios', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '01_load_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_load_distribution.png")

# ============================================================================
# PLOT 2: Generation vs Load
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(scenario_data['total_load_mw'], scenario_data['total_pg_mw'], 
           s=30, alpha=0.6, color='steelblue', label='Data points')
ax.plot([scenario_data['total_load_mw'].min(), scenario_data['total_load_mw'].max()],
        [scenario_data['total_load_mw'].min(), scenario_data['total_load_mw'].max()],
        'r--', linewidth=2, label='Perfect match (no losses)')

# Add regression line
z = np.polyfit(scenario_data['total_load_mw'], scenario_data['total_pg_mw'], 1)
p = np.poly1d(z)
x_line = np.linspace(scenario_data['total_load_mw'].min(), scenario_data['total_load_mw'].max(), 100)
ax.plot(x_line, p(x_line), 'g-', linewidth=2, alpha=0.7, label=f'Fit: slope={z[0]:.4f}')

ax.set_xlabel('Total Load (MW)', fontsize=12)
ax.set_ylabel('Total Generation (MW)', fontsize=12)
ax.set_title('Generation vs Load', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '02_generation_vs_load.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_generation_vs_load.png")

# ============================================================================
# PLOT 3: Voltage Distribution (All Buses)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(bus_data['Vm'], bins=50, color='purple', edgecolor='black', alpha=0.7)
ax.axvline(bus_data['Vm'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {bus_data["Vm"].mean():.4f} pu')
ax.axvline(1.0, color='green', linestyle=':', linewidth=2, label='Nominal: 1.0 pu')
ax.set_xlabel('Voltage Magnitude (pu)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Voltage Distribution (All Buses, All Scenarios)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '03_voltage_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_voltage_distribution.png")

# ============================================================================
# PLOT 4: Per-Bus Voltage Box Plot
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

# Prepare data for box plot
bus_voltage_data = [bus_data[bus_data['bus'] == bus]['Vm'].values for bus in sorted(bus_data['bus'].unique())]
bus_labels = sorted(bus_data['bus'].unique())

bp = ax.boxplot(bus_voltage_data, labels=bus_labels, patch_artist=True, showmeans=True)

# Color the boxes
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

# Add horizontal lines for limits
ax.axhline(1.0, color='green', linestyle=':', linewidth=1.5, label='Nominal (1.0 pu)', alpha=0.7)
ax.axhline(0.95, color='orange', linestyle='--', linewidth=1.5, label='Typical lower limit (0.95 pu)', alpha=0.7)
ax.axhline(1.05, color='orange', linestyle='--', linewidth=1.5, label='Typical upper limit (1.05 pu)', alpha=0.7)

ax.set_xlabel('Bus Number', fontsize=12)
ax.set_ylabel('Voltage Magnitude (pu)', fontsize=12)
ax.set_title('Voltage Range by Bus (Across All Scenarios)', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / '04_voltage_by_bus.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_voltage_by_bus.png")

# ============================================================================
# PLOT 5: Per-Bus Voltage Statistics Table (as plot)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = []
table_data.append(['Bus', 'Mean (pu)', 'Std (pu)', 'Min (pu)', 'Max (pu)', 'Range (pu)'])
for _, row in bus_voltage_stats.iterrows():
    table_data.append([
        f"{int(row['bus'])}",
        f"{row['vm_mean']:.4f}",
        f"{row['vm_std']:.4f}",
        f"{row['vm_min']:.4f}",
        f"{row['vm_max']:.4f}",
        f"{row['vm_max'] - row['vm_min']:.4f}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.10, 0.15, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(6):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

plt.title('Per-Bus Voltage Statistics', fontsize=14, fontweight='bold', pad=20)
plt.savefig(output_dir / '05_voltage_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_voltage_table.png")

# ============================================================================
# PLOT 6: Generator Active Power
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(gen_data['p_mw'], bins=40, color='darkgreen', edgecolor='black', alpha=0.7)
ax.axvline(gen_data['p_mw'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {gen_data["p_mw"].mean():.1f} MW')
ax.set_xlabel('Active Power (MW)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Generator Active Power Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '06_generator_active_power.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_generator_active_power.png")

# ============================================================================
# PLOT 7: Generator Reactive Power
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(gen_data['q_mvar'], bins=40, color='darkorange', edgecolor='black', alpha=0.7)
ax.axvline(gen_data['q_mvar'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {gen_data["q_mvar"].mean():.1f} MVAr')
ax.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Reactive Power (MVAr)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Generator Reactive Power Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '07_generator_reactive_power.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_generator_reactive_power.png")

# ============================================================================
# PLOT 8: Power Losses vs Load
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(scenario_data['total_load_mw'], scenario_data['losses_mw'], 
                     c=scenario_data['loss_percentage'], cmap='YlOrRd', s=40, alpha=0.7)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Loss Percentage (%)', fontsize=11)

# Add trend line
z = np.polyfit(scenario_data['total_load_mw'], scenario_data['losses_mw'], 2)
p = np.poly1d(z)
x_line = np.linspace(scenario_data['total_load_mw'].min(), scenario_data['total_load_mw'].max(), 100)
ax.plot(x_line, p(x_line), 'b--', linewidth=2, label='Quadratic fit', alpha=0.7)

ax.set_xlabel('Total Load (MW)', fontsize=12)
ax.set_ylabel('Losses (MW)', fontsize=12)
ax.set_title('Power Losses vs Load', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '08_losses_vs_load.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 08_losses_vs_load.png")

# ============================================================================
# PLOT 9: DROOP EFFECT - Frequency vs Load (if available)
# ============================================================================
if has_frequency:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(scenario_data['total_load_mw'], scenario_data['system_frequency']*60, 
               s=40, alpha=0.6, color='steelblue', label='Data points')
    
    # Add regression line
    z = np.polyfit(scenario_data['total_load_mw'], scenario_data['system_frequency']*60, 1)
    p = np.poly1d(z)
    x_line = np.linspace(scenario_data['total_load_mw'].min(), scenario_data['total_load_mw'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
            label=f'Droop: {z[0]:.2e} Hz/MW\n(mp = 0.2 configured)')
    
    # Add nominal frequency line
    ax.axhline(60.0, color='green', linestyle=':', linewidth=2, label='Nominal (60 Hz)', alpha=0.7)
    
    ax.set_xlabel('Total Load (MW)', fontsize=12)
    ax.set_ylabel('System Frequency (Hz)', fontsize=12)
    ax.set_title('Droop Effect: Frequency-Load Characteristic', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '09_droop_frequency_load.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 09_droop_frequency_load.png")

# ============================================================================
# PLOT 10: DROOP EFFECT - Voltage vs Reactive Power
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate per-scenario average voltage and total Q
v_q_data = scenario_data[['scenario', 'vm_mean', 'total_qg_mvar']].copy()

ax.scatter(v_q_data['total_qg_mvar'], v_q_data['vm_mean'], 
           s=40, alpha=0.6, color='purple', label='Data points')

# Add regression line
z = np.polyfit(v_q_data['total_qg_mvar'], v_q_data['vm_mean'], 1)
p = np.poly1d(z)
x_line = np.linspace(v_q_data['total_qg_mvar'].min(), v_q_data['total_qg_mvar'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
        label=f'Droop: {z[0]:.2e} pu/MVAr\n(mq = 0.2 configured)')

# Add nominal voltage line
ax.axhline(1.0, color='green', linestyle=':', linewidth=2, label='Nominal (1.0 pu)', alpha=0.7)

ax.set_xlabel('Total Reactive Power (MVAr)', fontsize=12)
ax.set_ylabel('Average System Voltage (pu)', fontsize=12)
ax.set_title('Droop Effect: Voltage-Reactive Power Characteristic', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '10_droop_voltage_reactive.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 10_droop_voltage_reactive.png")

# ============================================================================
# PLOT 11: Generator Load Sharing
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Get power output for a sample of scenarios
sample_scenarios = sorted(scenario_data['scenario'].unique())[::50]  # Every 50th scenario
gen_power_samples = gen_data[gen_data['scenario'].isin(sample_scenarios)]

# Pivot to get generators vs scenarios
gen_pivot = gen_power_samples.pivot_table(index='scenario', columns='idx', values='p_mw', aggfunc='first')

# Plot only first 10 generators for clarity
sample_gens = gen_pivot.columns[:10]
for gen_id in sample_gens:
    ax.plot(gen_pivot.index, gen_pivot[gen_id], marker='o', markersize=4, 
            linewidth=1.5, alpha=0.7, label=f'Gen {gen_id}')

ax.set_xlabel('Scenario Number', fontsize=12)
ax.set_ylabel('Active Power Output (MW)', fontsize=12)
ax.set_title('Generator Load Sharing Across Scenarios (Sample)', fontsize=14, fontweight='bold')
ax.legend(loc='best', ncol=2, fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '11_generator_load_sharing.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 11_generator_load_sharing.png")

# ============================================================================
# SAVE SUMMARY REPORT AS TEXT FILE
# ============================================================================
with open(output_dir / 'summary_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("DROOP CONTROL ANALYSIS - SUMMARY REPORT\n")
    f.write("IEEE 24-Bus RTS System\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"{'LOAD STATISTICS':-^80}\n")
    f.write(f"  Mean load:        {scenario_data['total_load_mw'].mean():>10.2f} MW\n")
    f.write(f"  Std deviation:    {scenario_data['total_load_mw'].std():>10.2f} MW\n")
    f.write(f"  Min load:         {scenario_data['total_load_mw'].min():>10.2f} MW\n")
    f.write(f"  Max load:         {scenario_data['total_load_mw'].max():>10.2f} MW\n")
    f.write(f"  Range:            {scenario_data['total_load_mw'].max() - scenario_data['total_load_mw'].min():>10.2f} MW\n\n")
    
    f.write(f"{'GENERATION STATISTICS':-^80}\n")
    f.write(f"  Mean active power:     {scenario_data['total_pg_mw'].mean():>10.2f} MW\n")
    f.write(f"  Mean reactive power:   {scenario_data['total_qg_mvar'].mean():>10.2f} MVAr\n\n")
    
    f.write(f"{'LOSS STATISTICS':-^80}\n")
    f.write(f"  Mean losses:           {scenario_data['losses_mw'].mean():>10.2f} MW\n")
    f.write(f"  Mean loss percentage:  {scenario_data['loss_percentage'].mean():>10.3f} %\n\n")
    
    f.write(f"{'VOLTAGE STATISTICS':-^80}\n")
    f.write(f"  Mean voltage:          {bus_data['Vm'].mean():>10.4f} pu\n")
    f.write(f"  Min voltage:           {bus_data['Vm'].min():>10.4f} pu\n")
    f.write(f"  Max voltage:           {bus_data['Vm'].max():>10.4f} pu\n\n")
    
    if has_frequency:
        f.write(f"{'FREQUENCY STATISTICS':-^80}\n")
        f.write(f"  Mean frequency:        {scenario_data['system_frequency'].mean():>10.6f} pu\n")
        f.write(f"  Mean deviation:        {scenario_data['freq_deviation'].mean():>10.2e} pu\n\n")
        
        slope_f = np.polyfit(scenario_data['total_load_mw'], scenario_data['system_frequency'], 1)[0]
        f.write(f"  Observed f-P droop:    {slope_f:>10.2e} pu/MW\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("PER-BUS VOLTAGE STATISTICS\n")
    f.write("="*80 + "\n\n")
    f.write(f"{'Bus':<6} {'Mean (pu)':<12} {'Std (pu)':<12} {'Min (pu)':<12} {'Max (pu)':<12} {'Range (pu)':<12}\n")
    f.write("-"*80 + "\n")
    for _, row in bus_voltage_stats.iterrows():
        f.write(f"{int(row['bus']):<6} {row['vm_mean']:<12.4f} {row['vm_std']:<12.4f} "
                f"{row['vm_min']:<12.4f} {row['vm_max']:<12.4f} {row['vm_max'] - row['vm_min']:<12.4f}\n")

print("✓ Saved: summary_report.txt")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll plots and summary saved to: {output_dir.absolute()}")
print("\nGenerated files:")
for file in sorted(output_dir.glob('*')):
    print(f"  - {file.name}")