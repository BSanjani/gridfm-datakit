import pandas as pd
import numpy as np
from pathlib import Path

# Paths to all three datasets
with_deadband_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_WithDeadband_10k\case24_ieee_rts\raw")
without_deadband_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_WithoutDeadband_10k\case24_ieee_rts\raw")
without_droop_dir = Path(r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_Withoutdroop_10k\case24_ieee_rts\raw")

def diagnose_runtime_data(directory, label):
    """Examine runtime data for convergence information"""
    print("\n" + "="*100)
    print(f" "*30 + f"RUNTIME DATA: {label}")
    print("="*100)
    
    runtime_file = directory / 'runtime_data.parquet'
    if not runtime_file.exists():
        print(f"\n✗ File not found: {runtime_file}")
        return None
    
    print(f"\n✓ Loading: {runtime_file}")
    runtime_df = pd.read_parquet(runtime_file)
    
    print(f"\n[1] BASIC INFO")
    print(f"  Shape: {runtime_df.shape}")
    print(f"  Columns: {list(runtime_df.columns)}")
    
    print(f"\n[2] DATA TYPES")
    print(runtime_df.dtypes)
    
    print(f"\n[3] FIRST 10 ROWS")
    print(runtime_df.head(10))
    
    print(f"\n[4] SUMMARY STATISTICS")
    print(runtime_df.describe())
    
    print(f"\n[5] CHECKING FOR CONVERGENCE FLAGS")
    # Look for common convergence-related column names
    convergence_keywords = ['converge', 'success', 'status', 'solved', 'error', 'flag']
    convergence_cols = [col for col in runtime_df.columns 
                       if any(keyword in col.lower() for keyword in convergence_keywords)]
    
    if convergence_cols:
        print(f"  Found convergence-related columns: {convergence_cols}")
        for col in convergence_cols:
            print(f"\n  Column '{col}':")
            print(f"    Unique values: {runtime_df[col].unique()}")
            print(f"    Value counts:")
            print(runtime_df[col].value_counts())
    else:
        print(f"  ✗ No obvious convergence flag columns found")
    
    print(f"\n[6] CHECKING FOR FREQUENCY DEVIATION")
    freq_keywords = ['freq', 'df', 'f', 'delta', 'deviation', 'hz']
    freq_cols = [col for col in runtime_df.columns 
                if any(keyword in col.lower() for keyword in freq_keywords)]
    
    if freq_cols:
        print(f"  Found frequency-related columns: {freq_cols}")
        for col in freq_cols:
            print(f"\n  Column '{col}':")
            print(f"    Type: {runtime_df[col].dtype}")
            print(f"    Non-null: {runtime_df[col].notna().sum()}/{len(runtime_df)}")
            print(f"    Non-zero: {(runtime_df[col] != 0).sum()}/{len(runtime_df)}")
            print(f"    Unique values: {runtime_df[col].nunique()}")
            print(f"    Range: [{runtime_df[col].min()}, {runtime_df[col].max()}]")
            print(f"    Mean: {runtime_df[col].mean():.8f}")
            print(f"    Std: {runtime_df[col].std():.8f}")
            print(f"    Sample values (first 20):")
            print(f"      {runtime_df[col].head(20).tolist()}")
    else:
        print(f"  ✗ No frequency-related columns found")
    
    print(f"\n[7] ALL COLUMNS - DETAILED LOOK")
    for col in runtime_df.columns:
        print(f"\n  Column '{col}':")
        print(f"    Type: {runtime_df[col].dtype}")
        print(f"    Non-null: {runtime_df[col].notna().sum()}/{len(runtime_df)}")
        print(f"    Unique: {runtime_df[col].nunique()}")
        if runtime_df[col].dtype in ['float64', 'int64']:
            print(f"    Range: [{runtime_df[col].min()}, {runtime_df[col].max()}]")
            print(f"    Mean: {runtime_df[col].mean()}")
            print(f"    Non-zero: {(runtime_df[col] != 0).sum()}")
        print(f"    Sample: {runtime_df[col].head(5).tolist()}")
    
    return runtime_df

# Main execution
print("\n" + "="*100)
print(" "*25 + "COMPREHENSIVE RUNTIME DATA DIAGNOSIS")
print("="*100)

runtime_with = diagnose_runtime_data(with_deadband_dir, "WITH DEADBAND")
runtime_without = diagnose_runtime_data(without_deadband_dir, "WITHOUT DEADBAND")
runtime_without_droop = diagnose_runtime_data(without_droop_dir, "WITHOUT DROOP")

print("\n" + "="*100)
print(" "*35 + "COMPARISON SUMMARY")
print("="*100)

if runtime_with is not None and runtime_without is not None and runtime_without_droop is not None:
    print(f"\nNumber of scenarios:")
    print(f"  With Deadband:     {len(runtime_with)}")
    print(f"  Without Deadband:  {len(runtime_without)}")
    print(f"  Without Droop:     {len(runtime_without_droop)}")
    
    print(f"\nColumn comparison:")
    cols_with = set(runtime_with.columns)
    cols_without = set(runtime_without.columns)
    cols_without_droop = set(runtime_without_droop.columns)
    
    common_cols = cols_with & cols_without & cols_without_droop
    print(f"  Common columns: {sorted(common_cols)}")
    
    unique_with = cols_with - cols_without - cols_without_droop
    unique_without = cols_without - cols_with - cols_without_droop
    unique_without_droop = cols_without_droop - cols_with - cols_without
    
    if unique_with:
        print(f"  Unique to With Deadband: {sorted(unique_with)}")
    if unique_without:
        print(f"  Unique to Without Deadband: {sorted(unique_without)}")
    if unique_without_droop:
        print(f"  Unique to Without Droop: {sorted(unique_without_droop)}")

print("\n" + "="*100)