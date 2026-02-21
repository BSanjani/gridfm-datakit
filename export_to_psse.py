import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
# Update this path to your specific data folder
JULIA_DATA_PATH = r'C:\Users\Bestu\Documents\GitHub\gridfm-datakit\data_out_ieee24_droop_droop_10k\case24_ieee_rts\raw'
OUTPUT_FILE = 'ieee24_scenario0.raw'
SCENARIO_ID = 0  # We want the base scenario

def write_psse_raw():
    print(f"Reading data from: {JULIA_DATA_PATH}")
    
    # 1. Load Parquet Files
    try:
        bus_df = pd.read_parquet(os.path.join(JULIA_DATA_PATH, 'bus_data.parquet'))
        gen_df = pd.read_parquet(os.path.join(JULIA_DATA_PATH, 'gen_data.parquet'))
        branch_df = pd.read_parquet(os.path.join(JULIA_DATA_PATH, 'branch_data.parquet'))
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Filter for Scenario 0
    s0_bus = bus_df[bus_df['scenario'] == SCENARIO_ID].copy()
    s0_gen = gen_df[gen_df['scenario'] == SCENARIO_ID].copy()
    s0_branch = branch_df[branch_df['scenario'] == SCENARIO_ID].copy()

    # Sort to ensure order
    s0_bus.sort_values('bus', inplace=True)
    s0_gen.sort_values('bus', inplace=True)

    # 3. Open File for Writing (PSS/E v33 Format)
    with open(OUTPUT_FILE, 'w') as f:
        # Header: 0, 100.00 (MVA Base), 33 (Version), ...
        f.write("0, 100.00, 33, 0, 1, 60.00\n")
        f.write("Two-line description provided by Python Script\n")
        f.write("IEEE 24-Bus System - Scenario 0 Export\n")
        
        # --- BUS DATA ---
        # Format: I, 'NAME', BASKV, IDE, AREA, ZONE, OWNER, VM, VA, NVHI, NVLO, EVHI, EVLO
        print(f"Writing {len(s0_bus)} Buses...")
        for _, row in s0_bus.iterrows():
            bus_id = int(row['bus'])
            base_kv = row.get('vn_kv', 138.0) # Default to 138 if missing
            
            # Determine Bus Type (IDE):
            # 1 = Load (PQ), 2 = Gen (PV), 3 = Swing, 4 = Isolated
            # We default to 1, then upgrade to 2 if gen exists, then 3 if slack
            bus_type = 1 
            
            # Check if this bus has a generator
            gens_at_bus = s0_gen[s0_gen['bus'] == bus_id]
            if not gens_at_bus.empty:
                bus_type = 2
                # Check for Slack
                if gens_at_bus['is_slack_gen'].any():
                    bus_type = 3
            
            # Voltage and Angle (from solution or flat start)
            vm = row.get('Vm', 1.0)
            va = row.get('Va', 0.0) # Degrees usually required
            
            f.write(f"{bus_id:>6}, 'BUS{bus_id:<4}', {base_kv:8.2f}, {bus_type}, 1, 1, 1, {vm:8.5f}, {va:8.5f}\n")
            
        f.write("0 / End of Bus Data\n")

        # --- LOAD DATA ---
        # Format: I, ID, STATUS, AREA, ZONE, PL, QL, IP, IQ, YP, YQ, OWNER
        print("Writing Loads...")
        for _, row in s0_bus.iterrows():
            pd_mw = row.get('Pd', 0.0)
            qd_mvar = row.get('Qd', 0.0)
            
            # Only write load line if Pd or Qd is non-zero
            if abs(pd_mw) > 1e-6 or abs(qd_mvar) > 1e-6:
                bus_id = int(row['bus'])
                f.write(f"{bus_id:>6}, '1 ', 1, 1, 1, {pd_mw:10.5f}, {qd_mvar:10.5f}, 0.0, 0.0, 0.0, 0.0, 1\n")
                
        f.write("0 / End of Load Data\n")

        # --- FIXED SHUNT DATA ---
        # Format: I, ID, STATUS, GL, BL
        print("Writing Shunts...")
        for _, row in s0_bus.iterrows():
            gs = row.get('GS', 0.0)
            bs = row.get('BS', 0.0)
            if abs(gs) > 1e-6 or abs(bs) > 1e-6:
                bus_id = int(row['bus'])
                f.write(f"{bus_id:>6}, '1 ', 1, {gs:10.5f}, {bs:10.5f}\n")
        f.write("0 / End of Fixed Shunt Data\n")

        # --- GENERATOR DATA ---
        # Format: I, ID, PG, QG, QT, QB, VS, IREG, MBASE, ZR, ZX, RT, XT, GTAP, STAT, RMPCT, PT, PB, O1, F1
        print(f"Writing {len(s0_gen)} Generators...")
        for _, row in s0_gen.iterrows():
            bus_id = int(row['bus'])
            pg = row.get('p_mw', 0.0)
            qg = row.get('q_mvar', 0.0)
            qmax = row.get('max_q_mvar', 999.0)
            qmin = row.get('min_q_mvar', -999.0)
            pmax = row.get('max_p_mw', 999.0)
            pmin = row.get('min_p_mw', 0.0)
            status = int(row.get('in_service', 1))
            
            # PSS/E needs Voltage Setpoint (VS). Usually same as Bus Vm.
            # We can lookup the bus voltage, or just default to 1.0 or the generator setpoint
            vs = 1.0 # Or lookup s0_bus[s0_bus['bus']==bus_id]['Vm']
            
            f.write(f"{bus_id:>6}, '1 ', {pg:10.3f}, {qg:10.3f}, {qmax:10.3f}, {qmin:10.3f}, {vs:6.4f}, 0, 100.0, 0.0, 1.0, 0.0, 0.0, 1.0, {status}, 100.0, {pmax:10.3f}, {pmin:10.3f}\n")
            
        f.write("0 / End of Generator Data\n")

        # --- BRANCH DATA ---
        # Format: I, J, CKT, R, X, B, RATEA, RATEB, RATEC, RATIO, ANGLE, STATUS, ...
        print(f"Writing {len(s0_branch)} Branches...")
        for _, row in s0_branch.iterrows():
            f_bus = int(row.get('f_bus', row.get('source_bus', 0))) # Handle likely column names
            t_bus = int(row.get('t_bus', row.get('target_bus', 0)))
            
            r = row.get('r', 0.001)
            x = row.get('x', 0.01)
            b_chg = row.get('b', 0.0)
            rate_a = row.get('rate_a', 0.0)
            status = int(row.get('status', 1)) # Default to active
            
            # Transformer handling (Tap/Shift)
            tap = row.get('tap', 0.0)
            shift = row.get('shift', 0.0)
            if tap == 0.0: tap = 1.0 # PSS/E defaults tap to 1.0 for lines
            
            f.write(f"{f_bus:>6}, {t_bus:>6}, '1 ', {r:10.6f}, {x:10.6f}, {b_chg:10.6f}, {rate_a:8.2f}, 0.0, 0.0, {tap:8.5f}, {shift:8.5f}, {status}, 0.0, 0.0\n")
            
        f.write("0 / End of Branch Data\n")
        
        # End of File Marker
        f.write("Q")

    print(f"✅ Successfully wrote {OUTPUT_FILE}")

if __name__ == "__main__":
    write_psse_raw()