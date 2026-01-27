import sys
import os
import csv

# --- CONFIGURATION ---
# Path to PSS/E 36.4 (Python 3.13 version)
PSSE_LOCATION = r"C:\Program Files\PTI\PSSE36\36.4\PSSPY313"
PSSBIN_LOCATION = r"C:\Program Files\PTI\PSSE36\36.4\PSSBIN"

# --- 1. INITIALIZE PSS/E ---
# Fix DLL logic for Python 3.8+
try:
    os.add_dll_directory(PSSBIN_LOCATION)
except AttributeError:
    pass

if PSSE_LOCATION not in sys.path: sys.path.append(PSSE_LOCATION)
if PSSBIN_LOCATION not in sys.path: sys.path.append(PSSBIN_LOCATION)
os.environ['PATH'] = PSSBIN_LOCATION + ';' + os.environ['PATH']

import psspy
import redirect

# --- 2. LOAD ARGUMENTS ---
if len(sys.argv) < 3:
    print("Error: Usage is -> python run_psse_logic.py <RAW_FILE> <CSV_FILE>")
    sys.exit(1)

# === THE FIX STARTS HERE ===
# Convert relative paths to Absolute Windows Paths (C:\...)
raw_file = os.path.abspath(sys.argv[1])
csv_file = os.path.abspath(sys.argv[2])

# Verify they exist before asking PSS/E to load them
if not os.path.exists(raw_file):
    print(f"ERROR: RAW file not found at: {raw_file}")
    sys.exit(1)
if not os.path.exists(csv_file):
    print(f"ERROR: CSV file not found at: {csv_file}")
    sys.exit(1)
# === THE FIX ENDS HERE ===

# Redirect PSS/E output to console so we can see it
redirect.psse2py()
psspy.psseinit(150000)

print(f"\n[PSS/E] Loading Base Case: {raw_file}")
ierr = psspy.read(0, raw_file)
if ierr != 0:
    print(f"Error loading RAW file (Code {ierr}). Check path!")
    sys.exit(1)

# --- 3. APPLY CHANGES ---
print(f"[PSS/E] Applying updates from: {csv_file}")
count = 0

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            # Parse row data
            bus = int(row['bus'])
            gen_id = str(row['id']) # ID must be a string
            p_new = float(row['p_mw'])
            q_new = float(row['q_mvar'])
            
            # UPDATE MACHINE DATA
            # machine_data_4(bus, id, intgar, realar)
            # We assume the generator exists. 
            # realar index 0 = P, index 1 = Q
            ierr = psspy.machine_data_4(bus, gen_id, [], [p_new, q_new])
            
            if ierr > 0:
                print(f"   Warning: Generator not found? Bus {bus} ID {gen_id} (Err {ierr})")
            else:
                count += 1
        except ValueError as e:
            print(f"   Skipping bad row: {row} ({e})")

print(f"[PSS/E] Successfully updated {count} generators.")

# --- 4. SOLVE ---
print("[PSS/E] Running Power Flow (FNSL)...")
# Lock options: Flat start=0, Tap adjustment=1, Area interchange=0
psspy.fnsl([0,0,0,1,1,0,99,0])

if psspy.solved() == 0:
    print("\n✅ CASE CONVERGED SUCCESSFULLY!")
else:
    print("\n❌ CASE DIVERGED (Solution failed).")