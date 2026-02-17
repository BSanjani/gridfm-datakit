import pandas as pd
import os

# Path to results
results_dir = r"C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee14_base_case\case14_ieee\raw"

bus_data_path = os.path.join(results_dir, "bus_data.parquet")
gen_data_path = os.path.join(results_dir, "gen_data.parquet")

print("=" * 80)
print("ACTUAL GENERATOR DATA FROM GRIDFM RESULTS")
print("=" * 80)

# Read generator data
df_gen = pd.read_parquet(gen_data_path)

print("\nGenerator columns available:")
print(df_gen.columns.tolist())

print("\n" + "=" * 80)
print("GENERATOR Q LIMITS AND OUTPUTS (FROM ACTUAL RESULTS)")
print("=" * 80)

# Check what columns exist for Q limits
if 'min_q_mvar' in df_gen.columns and 'max_q_mvar' in df_gen.columns:
    print(f"\n{'Gen#':<6} {'Bus':<6} {'Qg(MVAr)':<12} {'Qmin':<10} {'Qmax':<10} {'Status'}")
    print("-" * 80)
    
    for idx, gen in df_gen.iterrows():
        gen_num = int(gen['idx']) + 1
        bus = int(gen['bus'])
        qg = gen['q_mvar']
        qmin = gen['min_q_mvar']
        qmax = gen['max_q_mvar']
        
        # Check if hitting limits
        status = "✓ OK"
        if qg >= qmax - 0.5:
            status = "⚠️ AT Qmax!"
        elif qg <= qmin + 0.5:
            status = "⚠️ AT Qmin!"
        
        # Convert to PSCAD bus numbering
        pscad_bus = bus + 1
        
        print(f"{gen_num:<6} {pscad_bus:<6} {qg:<12.2f} {qmin:<10.1f} {qmax:<10.1f} {status}")
        
        # Highlight Bus 2 (PSCAD) = Bus 1 (GRIDFM)
        if pscad_bus == 2:
            print(f"   ↑ Bus 2: Check if Qmax = 50 (new) or 30 (old)")
            if abs(qmax - 50.0) < 0.1:
                print(f"   ✓ CORRECT: Using NEW Q limit (Qmax=50)")
            elif abs(qmax - 30.0) < 0.1:
                print(f"   ✗ WRONG: Still using OLD Q limit (Qmax=30)")
                print(f"   → Did you re-run GRIDFM after editing the .m file?")

else:
    print("\n⚠️ Q limit columns not found. Available columns:")
    print(df_gen.columns.tolist())
    print("\nShowing all generator data:")
    print(df_gen)

# Also check bus data for Q output
print("\n" + "=" * 80)
print("CROSS-CHECK: Q OUTPUT FROM BUS DATA")
print("=" * 80)

df_bus = pd.read_parquet(bus_data_path)

gen_buses = [0, 1, 2, 5, 7]
pscad_buses = [1, 2, 3, 6, 8]

print(f"\n{'PSCAD Bus':<12} {'GRIDFM Bus':<12} {'Qg (MVAr)':<12} {'Vm (pu)'}")
print("-" * 80)

for gridfm_bus, pscad_bus in zip(gen_buses, pscad_buses):
    bus_data = df_bus[df_bus['bus'] == gridfm_bus]
    if len(bus_data) > 0:
        qg = bus_data['Qg'].values[0]
        vm = bus_data['Vm'].values[0]
        print(f"{pscad_bus:<12} {gridfm_bus:<12} {qg:<12.2f} {vm:.6f}")
        
        if pscad_bus == 2:
            if abs(qg - 30.0) < 0.5:
                print(f"   → Bus 2 Q still at 30 MVAr (hitting old limit)")
            elif qg > 35.0:
                print(f"   → Bus 2 Q > 30 MVAr (using new Qmax=50)")

print("\n" + "=" * 80)
print("INSTRUCTIONS:")
print("=" * 80)
print("If Bus 2 Qmax still shows 30 (not 50):")
print("  1. Verify you edited the correct .m file")
print("  2. Re-run: python -m gridfm_datakit --config your_config.yaml")
print("  3. Make sure results go to the same output directory")
print("=" * 80)