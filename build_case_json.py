"""
Build IEEE 24-Bus case in PSS/E from JSON data
Pure Python - no pandas required
"""
import psspy
import json
import os

print("="*60)
print("Building IEEE 24-Bus Case in PSS/E")
print("="*60)

# Initialize PSS/E
psspy.psseinit(50000)


# Create new empty case
print("\nCreating new case...")
psspy.newcase_2([0,0],100.0,33,"")

# Load JSON data
os.chdir(r'C:\Users\Bestu\Documents\GitHub\gridfm-datakit')

with open('ieee24_data.json', 'r') as f:
    data = json.load(f)

bus_data = data['buses']
gen_data = data['generators']
branch_data = data['branches']

print(f"\nLoaded data:")
print(f"  Buses: {len(bus_data)}")
print(f"  Generators: {len(gen_data)}")
print(f"  Branches: {len(branch_data)}")

# Add buses
print("\nAdding buses...")
for bus in bus_data:
    bus_num = int(bus['bus']) + 1
    bus_name = f'BUS{bus_num}'
    base_kv = float(bus['vn_kv'])
    
    if bus['REF'] == 1:
        ide = 3
    elif bus['PV'] == 1:
        ide = 2
    else:
        ide = 1
    
    
    # Inserted 0.0 for Voltage Angle to align limits
    ierr = psspy.bus_data_3(bus_num, [ide, 1, 1, 1], [base_kv, 1.0, 0.0, 1.1, 0.9, 1.1, 0.9], bus_name)

ierr, nbuses = psspy.abuscount()
print(f"  Added {nbuses} buses")

# Add branches
print("\nAdding branches...")
for branch in branch_data:
    from_bus = int(branch['from_bus']) + 1
    to_bus = int(branch['to_bus']) + 1
    ckt = str(int(branch['idx']) % 10 + 1)
    r = float(branch['r'])
    x = float(branch['x'])
    b = float(branch['b'])
    rate = float(branch['rate_a'])
    
    
    ierr = psspy.branch_data(from_bus, to_bus, ckt, [1, 1, 0, 0], 
                         [r, x, b, rate, rate, rate, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ierr, nbranch = psspy.abrncount()
print(f"  Added {nbranch} branches")

# Add loads
print("\nAdding loads...")
load_count = 0
for bus in bus_data:
    bus_num = int(bus['bus']) + 1
    pd = float(bus['Pd'])
    qd = float(bus['Qd'])
    
    if abs(pd) > 0.001 or abs(qd) > 0.001:
        # ierr = psspy.load_data_4(i=bus_num, id='1', intgar1=1, realar=[pd, qd])
        ierr = psspy.load_data_4(bus_num, '1', [1, 1, 1, 1], [pd, qd, 0.0, 0.0, 0.0, 0.0])
        load_count += 1

print(f"  Added {load_count} loads")

# Add generators
print("\nAdding generators...")
gen_count = 0
for gen in gen_data:
    bus_num = int(gen['bus']) + 1
    gen_id = str(int(gen['idx']) % 100 + 1)
    
    pg = float(gen['p_mw'])
    qg = float(gen['q_mvar'])
    qmax = float(gen['max_q_mvar'])
    qmin = float(gen['min_q_mvar'])
    
    # Get voltage from bus data
    bus_idx = int(gen['bus'])
    vs = float(bus_data[bus_idx]['Vm'])
    
    pmax = float(gen['max_p_mw'])
    pmin = float(gen['min_p_mw'])
    status = int(gen['in_service'])
    mbase = 100.0  # Must be defined!
    
    # CORRECT PSS/E MAPPING (Must be 17 elements long)
    # Indices: 1:PG, 2:QG, 3:QMAX, 4:QMIN, 5:VS, 6:RMPCT, 7:MBASE, 
    #          8:R, 9:X, 10:RT, 11:XT, 12:GTAP, 13:STAT, 14:XN, 15:RN, 16:PMAX, 17:PMIN
    realar = [
        pg,    # 1
        qg,    # 2
        qmax,  # 3
        qmin,  # 4
        vs,    # 5
        0.0,   # 6 (RMPCT)
        mbase, # 7 (MBASE) - Critical!
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, # 8-15 (Impedances/Taps)
        pmax,  # 16
        pmin   # 17
    ]

    # Ensure bus is treated as a plant first (Type 2) to avoid "no plant data" error
    # This sets the regulated voltage setpoint (VS) explicitly
    psspy.plant_data(bus_num, 0, [vs, 1.0])

    ierr = psspy.machine_data_2(bus_num, gen_id, [status, 0], realar)
    
    if ierr > 0:
        print(f"  ERROR adding Gen at Bus {bus_num}: Code {ierr}")
    else:
        gen_count += 1

print(f"  Generators added: {gen_count}")




ierr, ngen = psspy.amachcount()
print(f"  Added {ngen} generators")

# Solve power flow
print("\nSolving power flow...")
ierr = psspy.fnsl([0,0,0,1,1,0,99,0])
if ierr == 0:
    print("  Power flow CONVERGED")
else:
    print(f"  Power flow returned code: {ierr}")

# Save
save_file = r'C:\Users\Bestu\Documents\GitHub\gridfm-datakit\ieee24_gridfm.sav'
ierr = psspy.save(save_file)
if ierr == 0:
    print(f"\nCase saved to: {save_file}")
else:
    print(f"\nSave returned code: {ierr}")

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)