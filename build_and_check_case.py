import psspy
import json
import os

# 1. SETUP
psspy.psseinit(50000)
psspy.newcase_2([0,0], 100.0, 33, "") # 100 MVA Base
work_dir = r'C:\Users\Bestu\Documents\GitHub\gridfm-datakit'
os.chdir(work_dir)

# 2. LOAD DATA
with open('ieee24_data.json', 'r') as f:
    data = json.load(f)
bus_kv_map = {int(b['bus']) + 1: float(b['vn_kv']) for b in data['buses']}

# 3. BUILD BUSES
print(f"Processing {len(data['buses'])} buses...")
for bus in data['buses']:
    bus_num = int(bus['bus']) + 1
    # Fix: Insert 0.0 for angle so limits align
    psspy.bus_data_3(bus_num, [1, 1, 1, 1], 
        [float(bus['vn_kv']), 1.0, 0.0, 1.1, 0.9, 1.1, 0.9], f'BUS{bus_num}')

# 4. BUILD LOADS
print(f"Processing loads...")
for bus in data['buses']:
    pd, qd = float(bus['Pd']), float(bus['Qd'])
    if abs(pd) > 0.001 or abs(qd) > 0.001:
        psspy.load_data_4(int(bus['bus'])+1, '1', [1,1,1,1], [pd, qd, 0, 0, 0, 0])

# 5. BUILD GENERATORS (The Fix)
print(f"Processing {len(data['generators'])} generators...")
for gen in data['generators']:
    bus_num = int(gen['bus']) + 1
    # FIX: Truncate ID to 2 chars max
    gen_id = str(int(gen['idx']) % 100 + 1)[:2]
    
    # FIX: Standard 17-value array
    realar = [
        float(gen['p_mw']), float(gen['q_mvar']), 
        float(gen['max_q_mvar']), float(gen['min_q_mvar']), 
        float(bus_kv_map[bus_num]), 0.0, 100.0, 
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 
        float(gen['max_p_mw']), float(gen['min_p_mw'])
    ]
    
    psspy.plant_data(bus_num, 0, [float(bus_kv_map[bus_num]), 1.0]) # Set Bus Type 2
    ierr = psspy.machine_data_2(bus_num, gen_id, [int(gen['in_service']), 0], realar)
    if ierr > 0: print(f"  Error adding Gen at {bus_num}: {ierr}")

# 6. BUILD BRANCHES (With Transformer Check)
print(f"Processing {len(data['branches'])} branches...")
for branch in data['branches']:
    f_bus, t_bus = int(branch['from_bus'])+1, int(branch['to_bus'])+1
    ckt = str(int(branch['idx']) % 9 + 1)
    vals = [float(branch['r']), float(branch['x']), float(branch['b']), float(branch['rate_a'])]
    
    # Check for Transformer (Voltage mismatch)
    if abs(bus_kv_map[f_bus] - bus_kv_map[t_bus]) > 0.1:
        # Add Transformer
        psspy.two_winding_data_3(f_bus, t_bus, ckt, [1,1,1,0,0,0,1,0,1,0,0,0,1,1,1],
            [vals[0], vals[1], 100.0, 1.0, 0.0, 0.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, vals[3], vals[3], vals[3], 0,0,0,0,0])
    else:
        # Add Line
        psspy.branch_data(f_bus, t_bus, ckt, [1,1,0,0], 
            [vals[0], vals[1], vals[2], vals[3], vals[3], vals[3], 0,0,0,0,0,0])

# 7. SOLVE & SAVE
psspy.fnsl([0,0,0,1,1,0,99,0])
psspy.save('ieee24_fixed.sav')

# 8. PRINT FINAL COUNTS (Verification)
print("\n" + "="*30)
print("   PSS/E CASE REPORT")
print("="*30)
_, n_bus = psspy.abuscount(-1, 1) # 1=In-service
_, n_gen = psspy.amachcount(-1, 1)
_, n_load = psspy.aloadcount(-1, 1)
_, n_branch = psspy.abrncount(-1, 1) # Non-transformer
_, n_xfmr = psspy.atrncount(-1, 1)   # Transformers

print(f"  Buses (Active):        {n_bus}")
print(f"  Generators (Active):   {n_gen}")
print(f"  Loads (Active):        {n_load}")
print(f"  Lines (Active):        {n_branch}")
print(f"  Transformers (Active): {n_xfmr}")
print("="*30)