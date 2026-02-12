import psspy
import json
import os

print("="*40)
print("   BUILDING CASE: GENERATOR FOCUS")
print("="*40)

# --- 1. SETUP & RESIZING (CRITICAL FIX) ---
# We initialize with 50,000 buses. This forces PSS/E to create 
# large tables for machines, solving the "Table Full" error at #21.
psspy.psseinit(200000) 

# 🔥 THIS IS THE MISSING PIECE
psspy.set_maximums(
    ngen=1000,   # generators
    nld=1000,
    nfxshunt=1000,
    nbrn=5000,
    ntr=2000,
)

# Standard new case. 
# [0,1] = [reset, output_options], 100.0 = SBASE
psspy.newcase_2([0,1], 100.0)

work_dir = r'C:\Users\Bestu\Documents\GitHub\gridfm-datakit'
os.chdir(work_dir)

# --- 2. LOAD DATA ---
with open('ieee24_data.json', 'r') as f:
    data = json.load(f)

# --- 3. SILENT BUS & LOAD ADDITION ---
# (We add these silently so we can focus on the Generator output)
print(f"Silently adding {len(data['buses'])} buses and loads...")
for bus in data['buses']:
    bus_num = int(bus['bus']) + 1
    psspy.bus_data_3(bus_num, [1, 1, 1, 1], [float(bus['vn_kv']), 1.0, 0.0, 1.1, 0.9, 1.1, 0.9], f'BUS{bus_num}')
    
    pd, qd = float(bus['Pd']), float(bus['Qd'])
    if abs(pd) > 0.001 or abs(qd) > 0.001:
        psspy.load_data_4(bus_num, '1', [1,1,1,1], [pd, qd, 0, 0, 0, 0])

# --- 4. BUILD GENERATORS (DETAILED ANALYSIS) ---
print("\n" + "="*40)
print("   STARTING GENERATOR ADDITION")
print("="*40)

bus_gen_count = {} 
success_count = 0

for i, gen in enumerate(data['generators']):
    bus_num = int(gen['bus']) + 1
    
    # ID Logic: 1, 2, 3... based on how many gens are on this bus
    count = bus_gen_count.get(bus_num, 0) + 1
    bus_gen_count[bus_num] = count
    gen_id = str(count)

    # 1. Ensure Bus Type is Gen (Code 2)
    # psspy.bus_data_3(bus_num, [2, 1, 1, 1])
    # psspy.bus_chng_3(bus_num, [2, -1, -1, -1], [], "")
    SWING_BUS = 21

    psspy.bus_chng_3(SWING_BUS, [3, -1, -1, -1], [], "")


    
    # 2. Add Plant Data (Required before Machine)
    # We suppress the output warning code from this specific call
    # psspy.plant_data(bus_num, 0, [1.0, 1.0])
    if count == 1:
        psspy.plant_data(bus_num, 0, [1.0, 1.0])


    # 3. Prepare Data
    # P_Gen, Q_Gen, Max_P, Min_P, P_Source, Min_Reactive, MBase
    realar = [
        float(gen['p_mw']),      # PGEN
        float(gen['q_mvar']),    # QGEN
        9999.0,                  # PT
        -9999.0,                 # PB
        9999.0,                  # QT
        -9999.0,                 # QB
        1.0,                     # VS
        0.0,                     # IREG
        100.0,                   # MBASE
        0.0, 0.0,                # ZR, ZX
        1.0,                     # RT
        0.0, 0.0,                # XT, GTAP
        1.0                      # STAT
    ]


    print(f"[Gen #{i+1}] Attempting: Bus {bus_num} (ID '{gen_id}')... ", end='')

    # 4. Add Machine
    # ierr = psspy.machine_data_2(bus_num, gen_id, [int(gen['in_service']), 0], realar_short)
    ierr = psspy.machine_data_2(
        bus_num,
        gen_id,
        [int(gen['in_service']), 0],
        realar
    )

    
    # --- ERROR ANALYSIS ---
    if ierr == 0:
        print("SUCCESS.")
        success_count += 1
    elif ierr == -1:
        # Code -1 is NOT a failure. It means "Success, but default ownership assigned".
        print("SUCCESS (with Ownership Warning).")
        success_count += 1
    else:
        # Real Failure (e.g., Code 5 = Table Full)
        print(f"FAIL! Error Code: {ierr}")
        # Stop immediate output to let user see the error
        if ierr == 5: 
            print("      >>> FATAL ERROR: Machine Limit Reached.")

# --- 5. BRANCHES (Silent) ---


print("\n" + "="*40)
print("   GENERATORS CURRENTLY IN PSS/E")
print("="*40)

ierr, buses = psspy.amachint(-1, 4, 'NUMBER')
ierr, ids   = psspy.amachchar(-1, 4, 'ID')
ierr, pgen  = psspy.amachreal(-1, 4, 'PGEN')
ierr, qgen  = psspy.amachreal(-1, 4, 'QGEN')

psse_gens = set()

for b, i, p, q in zip(buses, ids, pgen, qgen):
    bus = b[0]
    # gid = i[0].strip()
    gid = i[0].strip().zfill(2)
    pg  = p[0]
    qg  = q[0]

    print(f" Bus {bus:3d} | ID '{gid}' | P={pg:8.2f} MW | Q={qg:8.2f} Mvar")
    psse_gens.add((bus, gid))

# Adding branches to complete case topology
print("\nSilently adding branches...")
bus_kv_map = {int(b['bus']) + 1: float(b['vn_kv']) for b in data['buses']}
for branch in data['branches']:
    f, t = int(branch['from_bus'])+1, int(branch['to_bus'])+1
    c = str(int(branch['idx']) % 9 + 1)
    vals = [float(branch['r']), float(branch['x']), float(branch['b']), float(branch['rate_a'])]
    if abs(bus_kv_map[f] - bus_kv_map[t]) > 1.0:
        psspy.two_winding_data_3(f, t, c, [1,1,1,0,0,0,1,0,1,0,0,0,1,1,1], [vals[0], vals[1], 100.0, 1.0, 0.0, 0.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, vals[3], vals[3], vals[3], 0,0,0,0,0])
    else:
        psspy.branch_data(f, t, c, [1,1,0,0], [vals[0], vals[1], vals[2], vals[3], vals[3], vals[3], 0,0,0,0,0,0])




print("\n" + "="*40)
print("   GENERATORS EXPECTED FROM JSON")
print("="*40)

json_gens = set()

bus_gen_count = {}
for gen in data['generators']:
    bus = int(gen['bus']) + 1
    count = bus_gen_count.get(bus, 0) + 1
    bus_gen_count[bus] = count
    gid = f"{count:02d}"

    print(f" Bus {bus:3d} | ID '{gid}' | P={gen['p_mw']:8.2f} MW | Q={gen['q_mvar']:8.2f} Mvar")
    json_gens.add((bus, gid))

print("\n" + "="*40)
print("   MISSING GENERATORS (JSON - PSS/E)")
print("="*40)

missing = json_gens - psse_gens

if not missing:
    print(" None 🎉")
else:
    for b, i in sorted(missing):
        print(f" Missing: Bus {b}, ID '{i}'")




# --- 6. SUMMARY ---
psspy.fnsl() # Solve to prove stability
psspy.save('ieee24_debugged.sav')

print("\n" + "="*40)
print("   ANALYSIS COMPLETE")
print("="*40)
print(f"  Target Generators: {len(data['generators'])}")
print(f"  Added Generators:  {success_count}")
print("="*40)