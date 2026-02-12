import pandas as pd
import numpy as np
from pathlib import Path

# Paths to all three datasets
with_deadband_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_WithDeadband_10k\case24_ieee_rts\raw")
without_deadband_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_WithoutDeadband_10k\case24_ieee_rts\raw")
without_droop_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_Withoutdroop_10k\case24_ieee_rts\raw")

def deep_inspect_runtime(directory, label):
    """Deeply inspect runtime data to find frequency information"""
    print("\n" + "="*100)
    print(f"{label:^100}")
    print("="*100)
    
    runtime_file = directory / 'runtime_data.parquet'
    if not runtime_file.exists():
        print(f"✗ File not found: {runtime_file}")
        return None
    
    print(f"\n✓ Loading: {runtime_file}")
    df = pd.read_parquet(runtime_file)
    
    print(f"\n[1] BASIC STRUCTURE")
    print(f"  Shape: {df.shape} (rows × columns)")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n[2] ALL COLUMNS WITH DETAILS")
    print(f"  {'Column Name':<30} {'Type':<15} {'Non-Null':<12} {'Unique':<10} {'Sample Values'}")
    print(f"  {'-'*30} {'-'*15} {'-'*12} {'-'*10} {'-'*50}")
    
    for col in df.columns:
        non_null = df[col].notna().sum()
        unique = df[col].nunique()
        
        # Get sample values
        if df[col].dtype in ['float64', 'float32']:
            samples = df[col].head(5).tolist()
            sample_str = f"[{', '.join([f'{v:.6f}' for v in samples])}]"
        elif df[col].dtype in ['int64', 'int32']:
            samples = df[col].head(5).tolist()
            sample_str = f"{samples}"
        else:
            samples = df[col].head(3).tolist()
            sample_str = f"{samples}"
        
        print(f"  {col:<30} {str(df[col].dtype):<15} {non_null:<12} {unique:<10} {sample_str[:50]}")
    
    print(f"\n[3] NUMERIC COLUMNS - DETAILED STATISTICS")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        print(f"\n  Column: '{col}'")
        print(f"    Type: {df[col].dtype}")
        print(f"    Count: {df[col].count()} / {len(df)} ({100*df[col].count()/len(df):.1f}%)")
        print(f"    Mean:  {df[col].mean():.10f}")
        print(f"    Std:   {df[col].std():.10f}")
        print(f"    Min:   {df[col].min():.10f}")
        print(f"    25%:   {df[col].quantile(0.25):.10f}")
        print(f"    50%:   {df[col].quantile(0.50):.10f}")
        print(f"    75%:   {df[col].quantile(0.75):.10f}")
        print(f"    Max:   {df[col].max():.10f}")
        
        # Check for zeros
        zero_count = (df[col] == 0).sum()
        non_zero_count = (df[col] != 0).sum()
        print(f"    Zeros: {zero_count} ({100*zero_count/len(df):.1f}%)")
        print(f"    Non-zeros: {non_zero_count} ({100*non_zero_count/len(df):.1f}%)")
        
        # Show distribution of non-zero values
        if non_zero_count > 0:
            non_zero_vals = df[df[col] != 0][col]
            print(f"    Non-zero range: [{non_zero_vals.min():.10f}, {non_zero_vals.max():.10f}]")
            print(f"    Non-zero mean: {non_zero_vals.mean():.10f}")
        
        # Show first 20 actual values
        print(f"    First 20 values: {df[col].head(20).tolist()}")
    
    print(f"\n[4] SEARCHING FOR FREQUENCY-RELATED DATA")
    
    # Search all columns for frequency keywords
    freq_keywords = ['freq', 'df', 'f', 'delta', 'deviation', 'hz', 'omega', 'angular']
    found_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in freq_keywords):
            found_cols.append(col)
    
    if found_cols:
        print(f"  ✓ Found {len(found_cols)} frequency-related columns:")
        for col in found_cols:
            print(f"\n    '{col}':")
            print(f"      Values: {df[col].describe().to_dict()}")
            print(f"      Sample (first 30): {df[col].head(30).tolist()}")
    else:
        print(f"  ✗ No obvious frequency-related columns found")
    
    print(f"\n[5] CHECKING FOR HIDDEN/DERIVED FREQUENCY")
    
    # Maybe frequency is constant (60 Hz nominal) and deviation is in another form?
    # Check if there's a column that could be frequency in Hz
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            # Check if values are around 60 Hz or 50 Hz (power system frequencies)
            if 59.5 <= mean_val <= 60.5 or 49.5 <= mean_val <= 50.5:
                print(f"\n  Possible frequency column: '{col}'")
                print(f"    Mean: {mean_val:.6f} Hz")
                print(f"    Range: [{values.min():.6f}, {values.max():.6f}]")
                print(f"    Sample: {values.head(20).tolist()}")
    
    print(f"\n[6] FULL DATA PREVIEW (First 10 rows)")
    print(df.head(10).to_string())
    
    print(f"\n[7] FULL DATA PREVIEW (Last 10 rows)")
    print(df.tail(10).to_string())
    
    return df

print("="*100)
print(" "*25 + "COMPREHENSIVE RUNTIME DATA FREQUENCY DIAGNOSIS")
print("="*100)

# Inspect all three datasets
runtime_with = deep_inspect_runtime(with_deadband_dir, "WITH DEADBAND")
runtime_without = deep_inspect_runtime(without_deadband_dir, "WITHOUT DEADBAND")
runtime_without_droop = deep_inspect_runtime(without_droop_dir, "WITHOUT DROOP")

# Compare the three
print("\n" + "="*100)
print(" "*35 + "CROSS-DATASET COMPARISON")
print("="*100)

if all(df is not None for df in [runtime_with, runtime_without, runtime_without_droop]):
    print(f"\n[COLUMN COMPARISON]")
    cols_with = set(runtime_with.columns)
    cols_without = set(runtime_without.columns)
    cols_without_droop = set(runtime_without_droop.columns)
    
    all_cols = cols_with | cols_without | cols_without_droop
    
    print(f"\n  {'Column':<30} {'With DB':<10} {'Without DB':<12} {'Without Droop':<15}")
    print(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*15}")
    for col in sorted(all_cols):
        in_with = '✓' if col in cols_with else '✗'
        in_without = '✓' if col in cols_without else '✗'
        in_without_droop = '✓' if col in cols_without_droop else '✗'
        print(f"  {col:<30} {in_with:<10} {in_without:<12} {in_without_droop:<15}")
    
    print(f"\n[NUMERIC COMPARISON]")
    # Compare numeric columns that exist in all three
    common_numeric = (cols_with & cols_without & cols_without_droop)
    common_numeric = [col for col in common_numeric 
                     if runtime_with[col].dtype in [np.number] 
                     or pd.api.types.is_numeric_dtype(runtime_with[col])]
    
    if common_numeric:
        print(f"\n  Common numeric columns: {len(common_numeric)}")
        for col in sorted(common_numeric):
            print(f"\n  '{col}':")
            print(f"    With Deadband:     mean={runtime_with[col].mean():.6f}, std={runtime_with[col].std():.6f}, range=[{runtime_with[col].min():.6f}, {runtime_with[col].max():.6f}]")
            print(f"    Without Deadband:  mean={runtime_without[col].mean():.6f}, std={runtime_without[col].std():.6f}, range=[{runtime_without[col].min():.6f}, {runtime_without[col].max():.6f}]")
            print(f"    Without Droop:     mean={runtime_without_droop[col].mean():.6f}, std={runtime_without_droop[col].std():.6f}, range=[{runtime_without_droop[col].min():.6f}, {runtime_without_droop[col].max():.6f}]")

print("\n" + "="*100)
print(" "*30 + "DIAGNOSIS COMPLETE")
print("="*100)

print("""
NEXT STEPS:
1. Review the output above to identify the frequency deviation column
2. Check if frequency is stored in a different format (Hz instead of p.u.)
3. Verify if the column name matches what your plotting script expects
4. If no frequency column exists, check the simulation setup - frequency might not be recorded
""")

print("="*100)