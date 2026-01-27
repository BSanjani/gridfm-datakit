# debug_test.py

print("1. Testing imports...")
try:
    from pfdelta.solver import solve_power_flow
    print("   ✓ solver imported")
except Exception as e:
    print(f"   ✗ solver import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    from pfdelta.validation import validate_droop_config
    print("   ✓ validation imported")
except Exception as e:
    print(f"   ✗ validation import failed: {e}")
    exit(1)

try:
    from pfdelta.utils import process_droop_results
    print("   ✓ utils imported")
except Exception as e:
    print(f"   ✗ utils import failed: {e}")
    exit(1)

print("\n2. Testing Julia...")
try:
    from juliacall import Main as jl
    jl.seval("println(\"Julia connected!\")")
    print("   ✓ Julia imported")
except Exception as e:
    print(f"   ✗ Julia import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n3. Testing config loading...")
try:
    import yaml
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"   ✓ Config loaded")
    print(f"   Droop enabled: {config['droop_control']['enabled']}")
except Exception as e:
    print(f"   ✗ Config loading failed: {e}")
    exit(1)

print("\n✓ All basic checks passed! Ready to run test_droop.py")