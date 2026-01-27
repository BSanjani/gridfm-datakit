"""
PSS/E Debug Test - Step by step testing
"""
import psspy
import os

print("="*60)
print("PSS/E Debug Test")
print("="*60)

# Step 1: Initialize
print("\nStep 1: Initializing PSS/E...")
try:
    psspy.psseinit(50000)
    print("   SUCCESS: PSS/E initialized")
except Exception as e:
    print(f"   ERROR: {e}")
    import sys
    sys.exit(1)

# Step 2: Change directory
print("\nStep 2: Changing to data directory...")
try:
    os.chdir(r'C:\Users\Bestu\Documents\GitHub\gridfm-datakit')
    print(f"   Current directory: {os.getcwd()}")
    print("   Files in directory:")
    for f in os.listdir('.'):
        if f.endswith('.raw') or f.endswith('.dyr'):
            print(f"      {f}")
except Exception as e:
    print(f"   ERROR: {e}")

# Step 3: Try to load case
print("\nStep 3: Attempting to load RAW file...")
try:
    ierr = psspy.case('ieee24_psse.raw')
    print(f"   Return code: {ierr}")
    if ierr == 0:
        print("   SUCCESS: Case loaded")
    else:
        print(f"   WARNING: Non-zero return code {ierr}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Debug test complete")
print("="*60)