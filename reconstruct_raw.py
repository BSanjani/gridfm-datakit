import pandas as pd
import os

# --- PATHS ---
# Adjust this path if necessary to point to where your "branch_data.parquet" lives
DATA_DIR = r"data_out/case24_ieee_rts/raw"
OUTPUT_RAW = os.path.abspath("reconstructed_case24.raw")

def reconstruct():
    print(f"Reading grid topology from: {DATA_DIR}")
    
    # 1. Load Parquet Data
    try:
        bus_df = pd.read_parquet(os.path.join(DATA_DIR, "bus_data.parquet"))
        branch_df = pd.read_parquet(os.path.join(DATA_DIR, "branch_data.parquet"))
        gen_df = pd.read_parquet(os.path.join(DATA_DIR, "gen_data.parquet"))
    except Exception as e:
        print(f"Error reading parquet files: {e}")
        return

    print(f"   Found {len(bus_df)} buses, {len(branch_df)} branches, {len(gen_df)} generators.")

    with open(OUTPUT_RAW, 'w') as f:
        # HEADER
        f.write("0, 100.0, 33, 1, 1, 60.0  / PSS/E-33 Format Reconstructed\n")
        f.write("RECONSTRUCTED CASE 24\n")
        f.write("   0.0000,   0.0000\n") # Case ID lines

        # --- BUS DATA ---
        # Format: I, 'NAME', BASEKV, IDE, AREA, ZONE, OWNER, VM, VA, NVHI, NVLO, EVHI, EVLO
        # Note: We must adjust indices +1 if they are 0-based
        for idx, row in bus_df.iterrows():
            bus_id = int(row.name) + 1 if row.name == 0 or bus_df.index.min() == 0 else int(row.name)
            # Default values if columns missing
            base_kv = row.get('base_kv', 138.0)
            vm = row.get('vm', 1.0)
            va = row.get('va', 0.0)
            f.write(f"{bus_id:>4}, 'BUS-{bus_id:<4}', {base_kv:6.1f}, 1, 1, 1, 1, {vm:8.4f}, {va:8.4f}, 1.1, 0.9, 1.1, 0.9\n")
        f.write("0 / End of Bus Data\n")

        # --- LOAD DATA ---
        # We will skip static loads for now or set them to 0 if not in parquet
        # Often loads are on the buses, but let's see if simulation converges with just Gens first.
        f.write("0 / End of Load Data\n")

        # --- FIXED SHUNTS ---
        f.write("0 / End of Fixed Shunt Data\n")

        # --- GENERATOR DATA ---
        # Format: I, ID, PG, QG, QT, QB, VS, IREG, MBASE, ZR, ZX, RT, XT, GTAP, STAT, RMPCT, PT, PB, O1, F1
        # We need to match the gen_df to buses.
        # Assuming gen_df has a 'bus' column.
        gen_df['bus_id'] = gen_df['bus'].astype(int) + 1 # Fix 0-index
        
        # Create unique IDs for gens on same bus
        gen_df['temp_id'] = gen_df.groupby('bus_id').cumcount() + 1
        gen_df['temp_id'] = gen_df['temp_id'].astype(str)

        for _, row in gen_df.iterrows():
            bus_id = row['bus_id']
            gid = row['temp_id']
            pg = row.get('pg', 0.0) * 100 # Often in p.u., convert to MW if needed? 
            # WAIT: Check if Parquet is PU or MW. Usually PU in these datasets.
            # If values are like 0.16, it is PU. If 16.0, it is MW.
            # Let's assume MW for output or check magnitude.
            # For reconstruction, we can just put placeholders and let the CSV update update it later.
            
            f.write(f"{bus_id:>4}, '{gid:>1}', 0.0, 0.0, 999.0, -999.0, 1.0, 0, 100.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 100.0, 999.0, 0.0, 0, 1.0\n")
            
        f.write("0 / End of Generator Data\n")

        # --- BRANCH DATA ---
        # Format: I, J, CKT, R, X, B, RATEA, RATEB, RATEC, GI, BI, GJ, BJ, ST, MET, LEN, O1, F1
        for _, row in branch_df.iterrows():
            f_bus = int(row['f_bus']) + 1
            t_bus = int(row['t_bus']) + 1
            r = row.get('br_r', 0.001)
            x = row.get('br_x', 0.01)
            b = row.get('br_b', 0.0)
            rate = row.get('rate_a', 0.0)
            if rate == 0: rate = 999.0 # No limit
            
            f.write(f"{f_bus:>4}, {t_bus:>4}, '1', {r:10.5f}, {x:10.5f}, {b:10.5f}, {rate:6.1f}, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 0.0, 1, 1.0\n")
            
        f.write("0 / End of Branch Data\n")
        
        # FILLER FOR REMAINING SECTIONS
        f.write("0 / End of Transformer Data\n")
        f.write("0 / End of Area Data\n")
        f.write("0 / End of Two-Terminal DC Data\n")
        f.write("0 / End of VSC DC Line Data\n")
        f.write("0 / End of Switched Shunt Data\n")
        f.write("0 / End of Impedance Correction Data\n")
        f.write("0 / End of Multi-Terminal DC Data\n")
        f.write("0 / End of Multi-Section Line Data\n")
        f.write("0 / End of Zone Data\n")
        f.write("0 / End of Inter-Area Transfer Data\n")
        f.write("0 / End of Owner Data\n")
        f.write("0 / End of FACTS Device Data\n")
        f.write("Q\n")

    print(f"✅ Reconstructed RAW file saved to: {OUTPUT_RAW}")

if __name__ == "__main__":
    reconstruct()