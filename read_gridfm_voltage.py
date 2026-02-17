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

# Read bus data
df_bus = pd.read_parquet(bus_data_path)

print("=" * 80)
print("GENERATOR REACTIVE POWER CHECK")
print("=" * 80)

# Generator buses in GRIDFM (0-indexed)
gen_buses = [0, 1, 2, 5, 7]
pscad_buses = [1, 2, 3, 6, 8]

# IEEE 14-bus Q limits from standard data
q_limits = {
    1: {'Qmin': -999, 'Qmax': 999, 'Type': 'Slack'},
    2: {'Qmin': -30.0, 'Qmax': 30.0, 'Type': 'PV'},
    3: {'Qmin': 0.0, 'Qmax': 40.0, 'Type': 'SYNC'},
    6: {'Qmin': -6.0, 'Qmax': 24.0, 'Type': 'SYNC'},
    8: {'Qmin': -6.0, 'Qmax': 24.0, 'Type': 'SYNC'}
}

print("\nGenerator Reactive Power Output:")
print("-" * 80)
print(f"{'PSCAD Bus':<12} {'GRIDFM Bus':<12} {'Type':<8} {'Qg (MVAr)':<12} {'Qmin':<10} {'Qmax':<10} {'Status'}")
print("-" * 80)

for gridfm_bus, pscad_bus in zip(gen_buses, pscad_buses):
    bus_data = df_bus[df_bus['bus'] == gridfm_bus]
    
    if len(bus_data) > 0:
        qg = bus_data['Qg'].values[0]
        vm = bus_data['Vm'].values[0]
        
        limits = q_limits[pscad_bus]
        qmin = limits['Qmin']
        qmax = limits['Qmax']
        gen_type = limits['Type']
        
        # Check if hitting limits
        status = "✓ OK"
        if qg >= qmax - 0.5:
            status = "⚠️ AT Qmax!"
        elif qg <= qmin + 0.5:
            status = "⚠️ AT Qmin!"
        
        print(f"{pscad_bus:<12} {gridfm_bus:<12} {gen_type:<8} {qg:<12.2f} {qmin:<10.1f} {qmax:<10.1f} {status}")
        
        # Extra detail for Bus 8
        if pscad_bus == 8:
            print(f"   → Bus 8 Voltage: {vm:.6f} pu (Setpoint should be 1.090 pu)")
            if abs(vm - 1.090) > 0.01:
                print(f"   → Voltage deviation: {vm - 1.090:+.4f} pu from setpoint")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)

# Check Bus 8 specifically
bus8 = df_bus[df_bus['bus'] == 7]
if len(bus8) > 0:
    qg8 = bus8['Qg'].values[0]
    vm8 = bus8['Vm'].values[0]
    
    print(f"\n🔍 Bus 8 (GRIDFM Bus 7) Detailed Analysis:")
    print(f"  Voltage:        {vm8:.6f} pu")
    print(f"  Target:         1.090 pu")
    print(f"  Difference:     {vm8 - 1.090:+.6f} pu ({(vm8 - 1.090)/1.090 * 100:+.2f}%)")
    print(f"  Qg:             {qg8:.2f} MVAr")
    print(f"  Q Limits:       {q_limits[8]['Qmin']:.1f} to {q_limits[8]['Qmax']:.1f} MVAr")
    
    if qg8 >= 23.5:
        print(f"\n⚠️ ISSUE: Generator is hitting Qmax limit!")
        print(f"  → Bus becomes PQ type (voltage not controlled)")
        print(f"  → Voltage drops below setpoint")
        print(f"\nPossible causes:")
        print(f"  1. System needs more reactive power than generator can provide")
        print(f"  2. Check if PSCAD has same Q limits (Qmax=24 MVAr)")
        print(f"  3. Check if PSCAD generator is also hitting limits")
    elif qg8 <= -5.5:
        print(f"\n⚠️ ISSUE: Generator is hitting Qmin limit!")
        print(f"  → Bus becomes PQ type (voltage not controlled)")
    else:
        print(f"\n✓ Generator is within Q limits")
        print(f"  → But voltage is still {abs(vm8 - 1.090):.4f} pu away from setpoint")
        print(f"\nPossible causes:")
        print(f"  1. Check PSCAD voltage setpoint (should be 1.090 pu)")
        print(f"  2. Different generator/exciter models in PSCAD")
        print(f"  3. Check if PSCAD reached steady-state")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("Compare with PSCAD:")
print("  1. Check Bus 8 generator Qg in PSCAD")
print("  2. Verify Q limits: Qmin=-6, Qmax=24 MVAr")
print("  3. Verify voltage setpoint = 1.090 pu")
print("  4. Check if generator is hitting Q limits")
print("=" * 80)

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