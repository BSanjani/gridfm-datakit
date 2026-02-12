from pypower.api import case9, runpf, ppoption

print("Testing basic PyPower...")
ppc = case9()
print(f"Loaded {ppc['bus'].shape[0]} buses")

ppopt = ppoption(VERBOSE=1, OUT_ALL=0)
result, success = runpf(ppc, ppopt)

if success:
    print("✓ PyPower works!")
else:
    print("✗ PyPower failed")