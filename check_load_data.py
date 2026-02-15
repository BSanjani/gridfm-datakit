import pandas as pd
import os

# Path to results
results_dir = r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee14_base_case\case14_ieee\raw"

print("=" * 80)
print("CHECKING ACTUAL LOADS FROM GRIDFM RESULTS")
print("=" * 80)

# Read bus data from results (this shows what was actually simulated)
bus_data_path = os.path.join(results_dir, "bus_data.parquet")

if os.path.exists(bus_data_path):
    df_bus = pd.read_parquet(bus_data_path)
    
    print("\nColumns in bus results:")
    print(df_bus.columns.tolist())
    
    # Look for load columns (pd, qd)
    load_cols = [col for col in df_bus.columns if 'pd' in col.lower() or 'qd' in col.lower() or 'load' in col.lower()]
    
    if load_cols:
        print(f"\nLoad-related columns found: {load_cols}")
        
        # Display bus data with loads
        print("\n" + "=" * 80)
        print("ACTUAL BUS LOADS FROM SIMULATION RESULTS")
        print("=" * 80)
        
        # Try to find pd and qd columns
        pd_col = None
        qd_col = None
        bus_col = None
        
        for col in df_bus.columns:
            if col.lower() == 'pd' or col.lower() == 'p_load':
                pd_col = col
            if col.lower() == 'qd' or col.lower() == 'q_load':
                qd_col = col
            if col.lower() in ['bus', 'bus_i', 'bus_id']:
                bus_col = col
        
        if pd_col and qd_col:
            # Show the loads
            if bus_col:
                df_bus['PSCAD_Bus'] = df_bus[bus_col] + 1
                display_df = df_bus[[bus_col, 'PSCAD_Bus', pd_col, qd_col]].copy()
                display_df.columns = ['GRIDFM Bus', 'PSCAD Bus', 'P Load (MW)', 'Q Load (MVAr)']
            else:
                display_df = df_bus[[pd_col, qd_col]].copy()
                display_df.columns = ['P Load (MW)', 'Q Load (MVAr)']
                display_df['GRIDFM Bus'] = range(len(display_df))
                display_df['PSCAD Bus'] = display_df['GRIDFM Bus'] + 1
            
            print(display_df.to_string(index=False))
            
            # Calculate totals
            total_p = display_df['P Load (MW)'].sum()
            total_q = display_df['Q Load (MVAr)'].sum()
            
            print("\n" + "=" * 80)
            print("TOTAL LOADS FROM RESULTS:")
            print("=" * 80)
            print(f"Total P Load: {total_p:.2f} MW")
            print(f"Total Q Load: {total_q:.2f} MW")
            
            # Compare with base case
            print("\n" + "=" * 80)
            print("COMPARISON WITH BASE CASE:")
            print("=" * 80)
            base_total_p = 259.0
            base_total_q = 73.5
            
            print(f"Base case Total P: {base_total_p:.2f} MW")
            print(f"Results Total P:   {total_p:.2f} MW")
            print(f"Difference:        {total_p - base_total_p:.2f} MW ({(total_p - base_total_p)/base_total_p * 100:.2f}%)")
            
            print(f"\nBase case Total Q: {base_total_q:.2f} MVAr")
            print(f"Results Total Q:   {total_q:.2f} MVAr")
            print(f"Difference:        {total_q - base_total_q:.2f} MVAr ({(total_q - base_total_q)/base_total_q * 100:.2f}%)")
            
            if abs(total_p - base_total_p) < 0.1 and abs(total_q - base_total_q) < 0.1:
                print("\n✅ LOADS MATCH BASE CASE (Scenario 0)")
                print("   You can use standard IEEE 14-bus loads for PSCAD comparison")
            else:
                print("\n⚠️ LOADS DIFFER FROM BASE CASE")
                print("   Your results used a MODIFIED scenario with perturbed loads!")
                print("   Use the actual load values above for PSCAD comparison")
        else:
            print(f"\n❌ Could not find pd/qd columns in results")
            print("Available columns:", df_bus.columns.tolist())
    else:
        print("\n⚠️ No load columns found in bus results")
        print("Available columns:", df_bus.columns.tolist())
        
    # Check if there's scenario information
    if 'scenario' in df_bus.columns:
        scenarios = df_bus['scenario'].unique()
        print("\n" + "=" * 80)
        print("SCENARIO INFORMATION:")
        print("=" * 80)
        print(f"Scenarios in results: {scenarios}")
        if len(scenarios) == 1 and scenarios[0] == 0:
            print("✅ Only Scenario 0 (base case) found")
        else:
            print(f"⚠️ Multiple scenarios found: {len(scenarios)}")
            
else:
    print(f"\n❌ Results file not found: {bus_data_path}")

# Also check the scenario load profile file
print("\n" + "=" * 80)
print("CHECKING LOAD SCENARIO FILE:")
print("=" * 80)

scenario_file = os.path.join(results_dir, "scenarios_agg_load_profile.parquet")
if os.path.exists(scenario_file):
    df_scenario = pd.read_parquet(scenario_file)
    print(f"\nScenario file found!")
    print(f"Shape: {df_scenario.shape}")
    print(f"Columns: {df_scenario.columns.tolist()}")
    
    if 'scenario' in df_scenario.columns:
        print(f"\nNumber of scenarios: {len(df_scenario['scenario'].unique())}")
        print(f"Scenario indices: {sorted(df_scenario['scenario'].unique())}")
else:
    print("Scenario file not found")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print("1. If loads match base case (259 MW) → Use standard IEEE loads in PSCAD")
print("2. If loads differ → You ran a perturbed scenario, use actual values above")
print("3. For proper comparison, re-run GRIDFM with corrected YAML (no perturbations)")
print("=" * 80)