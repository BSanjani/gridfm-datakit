import pandas as pd
import os


import pandas as pd

# Path to droop control results
results_dir = r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee14_droop_withdeadband\case14_ieee\raw"


# Path to results
# results_dir = r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee14_base_case\case14_ieee\raw"

print("=" * 70)
print("GRIDFM IEEE 14-BUS RESULTS - BUS VOLTAGE DATA")
print("=" * 70)

# Read bus voltage data from parquet
bus_data_path = os.path.join(results_dir, "bus_data.parquet")
gen_data_path = os.path.join(results_dir, "gen_data.parquet")
branch_data_path = os.path.join(results_dir, "branch_data.parquet")

if os.path.exists(branch_data_path):
    print(f"\n✓ Reading branch data from: {branch_data_path}")
    branch_df = pd.read_parquet(branch_data_path)
    print(f"  Shape: {branch_df.shape[0]} rows × {branch_df.shape[1]} columns")
    print(f"  Columns: {list(branch_df.columns)}")
    
    # Show branch data
    print("\n" + "-" * 70)
    print("First 10 rows of branch data:")
    print("-" * 70)
    print(branch_df.head(10))

if os.path.exists(gen_data_path):
    print(f"\n✓ Reading generator data from: {gen_data_path}")
    gen_df = pd.read_parquet(gen_data_path)
    print(f"  Shape: {gen_df.shape[0]} rows × {gen_df.shape[1]} columns")
    print(f"  Columns: {list(gen_df.columns)}")
    
    # Show generator data    print("\n" + "-" * 70)
    print("First 10 rows of generator data:")
    print("-" * 70)
    print(gen_df.head(10))




if os.path.exists(bus_data_path):
    print(f"\n✓ Reading: {bus_data_path}")
    
    # Read the parquet file
    df = pd.read_parquet(bus_data_path)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n  Columns: {list(df.columns)}")
    
    # Show first few rows
    print("\n" + "-" * 70)
    print("First 10 rows of data:")
    print("-" * 70)
    print(df.head(15))
    
    # Extract Bus 5 data
    print("\n" + "=" * 70)
    print("BUS 5 VOLTAGE DATA")
    print("=" * 70)
    
    # Check if there's a bus column
    if 'bus' in df.columns or 'bus_i' in df.columns or 'bus_id' in df.columns:
        # Find the bus column name
        bus_col = None
        for col in ['bus', 'bus_i', 'bus_id', 'bus_num']:
            if col in df.columns:
                bus_col = col
                break
        
        if bus_col:
            # Filter for bus 5
            bus5_data = df[df[bus_col] == 5]
            
            if len(bus5_data) > 0:
                print(f"\n✓ Found Bus 5 data (GRIDFM 0-indexed = PSCAD Bus 6):")
                print("-" * 70)
                print(bus5_data)
                
                # Extract voltage magnitude
                for vm_col in ['vm', 'Vm', 'v_mag', 'voltage_magnitude', 'V']:
                    if vm_col in df.columns:
                        vm = bus5_data[vm_col].values[0]
                        print(f"\n✓ Voltage Magnitude: {vm:.6f} pu")
                        break
                
                # Extract voltage angle
                for va_col in ['va', 'Va', 'v_ang', 'voltage_angle', 'theta']:
                    if va_col in df.columns:
                        va = bus5_data[va_col].values[0]
                        print(f"✓ Voltage Angle: {va:.6f} degrees")
                        break
            else:
                print(f"\n✗ No data found for bus = 5")
                print(f"Available buses: {sorted(df[bus_col].unique())}")
    
    # If no bus column, assume row index corresponds to bus
    else:
        print("\n⚠ No bus identifier column found.")
        print("Assuming row index corresponds to bus number (0-indexed):")
        
        if len(df) > 5:
            print(f"\nBus 5 data (row 5):")
            print("-" * 70)
            print(df.iloc[5])
            
            # Try to find voltage columns
            for col in df.columns:
                if 'vm' in col.lower() or 'v_mag' in col.lower():
                    print(f"\n✓ Voltage Magnitude ({col}): {df.iloc[5][col]:.6f} pu")
                if 'va' in col.lower() or 'v_ang' in col.lower() or 'theta' in col.lower():
                    print(f"✓ Voltage Angle ({col}): {df.iloc[5][col]:.6f} degrees")
        else:
            print(f"\n✗ DataFrame has only {len(df)} rows, cannot access row 5")
    
    # Summary of all bus voltages
    print("\n" + "=" * 70)
    print("ALL BUS VOLTAGES SUMMARY")
    print("=" * 70)
    
    # Find voltage magnitude column
    vm_col = None
    for col in ['vm', 'Vm', 'v_mag', 'voltage_magnitude', 'V']:
        if col in df.columns:
            vm_col = col
            break
    
    if vm_col:
        print(f"\nVoltage Magnitudes (pu):")
        print("-" * 70)
        if 'bus' in df.columns or 'bus_i' in df.columns:
            bus_col = 'bus' if 'bus' in df.columns else 'bus_i'
            voltage_summary = df[[bus_col, vm_col]].sort_values(bus_col)
            print(voltage_summary.to_string(index=False))
        else:
            print(df[vm_col])
    
else:
    print(f"\n✗ File not found: {bus_data_path}")
    print("Please check the path.")

print("\n" + "=" * 70)
print("📝 REMINDER: Bus Numbering")
print("=" * 70)
print("GRIDFM uses 0-indexed buses:")
print("  GRIDFM Bus 0 = PSCAD Bus 1 (Slack)")
print("  GRIDFM Bus 5 = PSCAD Bus 6")
print("  To compare with PSCAD: PSCAD_bus = GRIDFM_bus + 1")
print("=" * 70)