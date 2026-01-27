import sys
import os

# --- 1. DEFINE YOUR PSS/E PATHS ---
# Based on your provided path: C:\Program Files\PTI\PSSE36\36.4\PSSPY314
# PLEASE VERIFY: Is 'PSSPY314' actually 'PSSPY310' or 'PSSPY311'? 
# The Python version running this script MUST match that number.

PSSE_LOCATION = r"C:\Program Files\PTI\PSSE36\36.4\PSSPY314"  
PSSBIN_LOCATION = r"C:\Program Files\PTI\PSSE36\36.4\PSSBIN" 

# --- 2. VALIDATE PATHS ---
if not os.path.exists(PSSE_LOCATION):
    raise FileNotFoundError(f"Could not find PSS/E Python folder at: {PSSE_LOCATION}")
if not os.path.exists(PSSBIN_LOCATION):
    raise FileNotFoundError(f"Could not find PSS/E Binary folder at: {PSSBIN_LOCATION}")

# --- 3. CONFIGURE ENVIRONMENT ---
# Add PSS/E to Python Path so 'import psspy' works
if PSSE_LOCATION not in sys.path:
    sys.path.append(PSSE_LOCATION)
if PSSBIN_LOCATION not in sys.path:
    sys.path.append(PSSBIN_LOCATION)

# Add PSSBIN to Windows 'PATH' so it can find the .dll files (Crucial!)
os.environ['PATH'] = PSSBIN_LOCATION + ';' + os.environ['PATH']

# --- 4. IMPORT PSS/E ---
try:
    import psspy
    import redirect
    # Optional: Redirect PSS/E output to this Python console
    redirect.psse2py()
    
    # Initialize PSS/E (150,000 bus limit is standard for v36)
    psspy.psseinit(150000)
    print(f"Success! PSS/E 36 linked to Python {sys.version.split()[0]}")
    
except ImportError as e:
    print("Error importing psspy. Check that your Python version matches the PSSPY folder version.")
    print(f"Current Python: {sys.version}")
    raise e

# --- 5. YOUR GRIDFM CODE STARTS HERE ---
import gridfm_datakit as gfm

# You can now use PSS/E commands and GridFM commands in the same script
# Example: gfm.generate() -> psspy.read()