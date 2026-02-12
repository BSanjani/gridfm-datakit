import pandas as pd
import numpy as np
from pathlib import Path

# Paths to all three datasets
with_deadband_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_WithDeadband_10k\case24_ieee_rts\raw")
without_deadband_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_WithoutDeadband_10k\case24_ieee_rts\raw")
without_droop_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_Withoutdroop_10k\case24_ieee_rts\raw")

def identify_failed_scenarios(bus_file, voltage_threshold=0.8):
    """Identify scenarios where voltage collapse occurred"""
    bus_df = pd.read_parquet(bus_file)
    
    # Find scenarios with any bus below threshold
    failed_buses = bus_df[bus_df['Vm'] < voltage_threshold]
    failed_scenarios = failed_buses['scenario'].unique()
    
    return failed_scenarios, bus_df

def filter_datasets_by_scenarios(valid_scenarios):
    """Filter all parquet files to only include valid scenarios"""
    
    datasets = {
        'With Deadband': with_deadband_dir,
        'Without Deadband': without_deadband_dir,
        'Without Droop': without_droop_dir
    }
    
    results = {}
    
    for name, directory in datasets.items():
        print(f"\nFiltering: {name}")
        print(f"  Valid scenarios to keep: {len(valid_scenarios)}")
        
        filtered_data = {}
        
        # Filter each data file
        for filename in ['bus_data.parquet', 'gen_data.parquet', 'branch_data.parquet', 'runtime_data.parquet']:
            filepath = directory / filename
            if filepath.exists():
                df = pd.read_parquet(filepath)
                original_size = len(df)
                
                # Filter to valid scenarios
                if 'scenario' in df.columns:
                    df_filtered = df[df['scenario'].isin(valid_scenarios)]
                    filtered_size = len(df_filtered)
                    
                    print(f"    {filename:25s}: {original_size:8d} → {filtered_size:8d} rows ({100*filtered_size/original_size:.1f}%)")
                    
                    key = filename.replace('_data.parquet', '')
                    filtered_data[key] = df_filtered
                else:
                    print(f"    {filename:25s}: No 'scenario' column, keeping as-is")
                    key = filename.replace('_data.parquet', '')
                    filtered_data[key] = df
            else:
                print(f"    {filename:25s}: Not found")
        
        results[name] = filtered_data
    
    return results

def compare_statistics(results):
    """Compare statistics before and after filtering"""
    
    print("\n" + "="*100)
    print(" "*30 + "COMPARISON: CONVERGED SCENARIOS ONLY")
    print("="*100)
    
    for name, data in results.items():
        print(f"\n{name}:")
        bus_df = data.get('bus')
        if bus_df is not None:
            print(f"  Scenarios: {bus_df['scenario'].nunique()}")
            print(f"  Voltage stats:")
            print(f"    Mean:  {bus_df['Vm'].mean():.6f} p.u.")
            print(f"    Std:   {bus_df['Vm'].std():.6f} p.u.")
            print(f"    Min:   {bus_df['Vm'].min():.6f} p.u.")
            print(f"    Max:   {bus_df['Vm'].max():.6f} p.u.")
            
            # Bus 5 specific stats
            bus5 = bus_df[bus_df['bus'] == 5]
            print(f"  Bus 5 voltage:")
            print(f"    Mean:  {bus5['Vm'].mean():.6f} p.u.")
            print(f"    Std:   {bus5['Vm'].std():.6f} p.u.")
            print(f"    Min:   {bus5['Vm'].min():.6f} p.u.")
            print(f"    Max:   {bus5['Vm'].max():.6f} p.u.")

print("="*100)
print(" "*20 + "POWER FLOW CONVERGENCE ANALYSIS & DATASET FILTERING")
print("="*100)

# Step 1: Identify failed scenarios in each dataset
print("\n[STEP 1] Identifying Failed Scenarios (Vm < 0.8 p.u.)")

voltage_threshold = 0.8

failed_with, bus_with = identify_failed_scenarios(with_deadband_dir / 'bus_data.parquet', voltage_threshold)
failed_without, bus_without = identify_failed_scenarios(without_deadband_dir / 'bus_data.parquet', voltage_threshold)
failed_without_droop, bus_without_droop = identify_failed_scenarios(without_droop_dir / 'bus_data.parquet', voltage_threshold)

print(f"\nFailed scenarios (Vm < {voltage_threshold} p.u.):")
print(f"  With Deadband:     {len(failed_with):6d} / {bus_with['scenario'].nunique():6d} ({100*len(failed_with)/bus_with['scenario'].nunique():.2f}%)")
print(f"  Without Deadband:  {len(failed_without):6d} / {bus_without['scenario'].nunique():6d} ({100*len(failed_without)/bus_without['scenario'].nunique():.2f}%)")
print(f"  Without Droop:     {len(failed_without_droop):6d} / {bus_without_droop['scenario'].nunique():6d} ({100*len(failed_without_droop)/bus_without_droop['scenario'].nunique():.2f}%)")

# Step 2: Find common valid scenarios across all three datasets
print("\n[STEP 2] Finding Common Valid Scenarios Across All Three Datasets")

all_scenarios_with = set(bus_with['scenario'].unique())
all_scenarios_without = set(bus_without['scenario'].unique())
all_scenarios_without_droop = set(bus_without_droop['scenario'].unique())

valid_with = all_scenarios_with - set(failed_with)
valid_without = all_scenarios_without - set(failed_without)
valid_without_droop = all_scenarios_without_droop - set(failed_without_droop)

print(f"\nValid scenarios in each dataset:")
print(f"  With Deadband:     {len(valid_with):6d}")
print(f"  Without Deadband:  {len(valid_without):6d}")
print(f"  Without Droop:     {len(valid_without_droop):6d}")

# Find common valid scenarios
common_valid = valid_with & valid_without & valid_without_droop

print(f"\nCommon valid scenarios across all three: {len(common_valid)}")

# Step 3: Filter datasets
print("\n[STEP 3] Filtering All Datasets to Common Valid Scenarios")

filtered_results = filter_datasets_by_scenarios(common_valid)

# Step 4: Compare statistics
compare_statistics(filtered_results)

# Step 5: Save filtered data (optional)
print("\n" + "="*100)
print("SAVE FILTERED DATA? (Uncomment the code below if you want to save)")
print("="*100)

save_filtered = False  # Set to True to save filtered datasets

if save_filtered:
    output_base = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit")
    
    for name, data in filtered_results.items():
        # Create output directory
        dir_name = name.replace(' ', '_').lower()
        output_dir = output_base / f"filtered_{dir_name}" / "case24_ieee_rts" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving: {name} → {output_dir}")
        
        for key, df in data.items():
            filename = f"{key}_data.parquet"
            output_file = output_dir / filename
            df.to_parquet(output_file)
            print(f"  ✓ {filename}")

print("\n" + "="*100)
print("SUMMARY & RECOMMENDATIONS")
print("="*100)

print(f"""
The analysis shows:

1. FAILURE RATES:
   - With Deadband:    {100*len(failed_with)/bus_with['scenario'].nunique():.2f}% scenarios failed
   - Without Deadband: {100*len(failed_without)/bus_without['scenario'].nunique():.2f}% scenarios failed  
   - Without Droop:    {100*len(failed_without_droop)/bus_without_droop['scenario'].nunique():.2f}% scenarios failed
   
2. CONVERGED SCENARIOS:
   - Common valid scenarios: {len(common_valid)} ({100*len(common_valid)/max(len(all_scenarios_with), len(all_scenarios_without), len(all_scenarios_without_droop)):.1f}% of total)
   
3. RECOMMENDATION:
   You have TWO options for analysis:
   
   OPTION A - Include All Scenarios (Current Approach):
   ✓ Shows the full picture including failure modes
   ✓ Demonstrates that droop control prevents voltage collapse
   ✗ Statistics are skewed by failed scenarios (especially Bus 5)
   
   OPTION B - Compare Only Converged Scenarios:
   ✓ Fair comparison of system performance
   ✓ Clean statistics without voltage collapse outliers
   ✗ Removes {100*len(failed_without_droop)/bus_without_droop['scenario'].nunique():.2f}% of "Without Droop" data
   
   For a fair comparison of control strategies, use OPTION B.
   To show that droop prevents collapse, use OPTION A with proper context.

4. TO USE FILTERED DATA:
   - Set save_filtered = True above to save cleaned datasets
   - Update your plotting script paths to use filtered_* directories
   - Re-run your analysis with clean data
""")

print("="*100)