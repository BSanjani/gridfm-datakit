"""
Step 1: Simple PyPower Test
Run this first to make sure PyPower droop solver works
"""

import numpy as np
from pypower.api import case9, case14, case24_ieee_rts
from pypower_droop_solver import PyPowerDroopSolver


def test_pypower_droop():
    """Test PyPower droop solver with a simple case"""
    
    print("\n" + "="*70)
    print(" "*20 + "PYPOWER DROOP SOLVER TEST")
    print("="*70)
    
    # Start with a small case for testing
    print("\n[Step 1/3] Loading test case...")
    ppc = case9()  # Start small!
    print(f"  ✓ Loaded IEEE 9-bus system")
    print(f"    {ppc['bus'].shape[0]} buses")
    print(f"    {ppc['gen'].shape[0]} generators")
    print(f"    {ppc['branch'].shape[0]} branches")
    
    # Configure droop control
    print("\n[Step 2/3] Configuring droop control...")
    droop_config = {
        'enabled': True,
        'mp': 0.05,
        'mq': 0.05,
        'V_0': 1.0,
        'frequency_deadband': 0.0006,  # Try 0.0 for no deadband first
        'voltage_deadband': 0.0,
        'droop_buses': [1, 2, 3]  # IEEE 9-bus: assume gens at buses 1, 2, 3
    }
    
    print(f"  Droop configuration:")
    print(f"    mp = {droop_config['mp']}")
    print(f"    mq = {droop_config['mq']}")
    print(f"    Frequency deadband = {droop_config['frequency_deadband']} p.u.")
    print(f"    Droop buses = {droop_config['droop_buses']}")
    
    # Solve
    print("\n[Step 3/3] Solving with droop control...")
    solver = PyPowerDroopSolver(ppc, droop_config)
    results = solver.solve(verbose=True)
    
    # Summary
    print("\n" + "="*70)
    if results['converged']:
        print("✅ SUCCESS: PyPower droop solver converged")
        print(f"   Frequency deviation: {results['df']:.6f} p.u.")
        print(f"   Iterations: {results['iterations']}")
        print("\nNext step: Run comparison with Julia solver")
        print("  → Use: python compare_pypower_julia.py")
    else:
        print("❌ FAILED: PyPower droop solver did not converge")
        print("   Check generator limits and network parameters")
    print("="*70 + "\n")
    
    return results


def test_with_and_without_deadband():
    """Compare results with and without deadband"""
    
    print("\n" + "="*70)
    print(" "*15 + "DEADBAND EFFECT TEST")
    print("="*70)
    
    ppc = case9()
    
    # Test 1: Without deadband
    print("\n[Test 1/2] Running WITHOUT deadband...")
    config_no_db = {
        'enabled': True,
        'mp': 0.05,
        'mq': 0.05,
        'V_0': 1.0,
        'frequency_deadband': 0.0,  # NO DEADBAND
        'voltage_deadband': 0.0,
        'droop_buses': [1, 2, 3]
    }
    
    solver1 = PyPowerDroopSolver(ppc, config_no_db)
    results1 = solver1.solve(verbose=False)
    
    # Test 2: With deadband
    print("\n[Test 2/2] Running WITH deadband...")
    config_with_db = {
        'enabled': True,
        'mp': 0.05,
        'mq': 0.05,
        'V_0': 1.0,
        'frequency_deadband': 0.0006,  # WITH DEADBAND
        'voltage_deadband': 0.0,
        'droop_buses': [1, 2, 3]
    }
    
    solver2 = PyPowerDroopSolver(ppc, config_with_db)
    results2 = solver2.solve(verbose=False)
    
    # Compare
    print("\n" + "="*70)
    print(" "*20 + "COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Metric':<30} {'Without DB':<15} {'With DB':<15} {'Difference'}")
    print("-"*70)
    print(f"{'Converged':<30} {str(results1['converged']):<15} {str(results2['converged']):<15} -")
    print(f"{'Frequency deviation (df)':<30} {results1['df']:<15.6f} {results2['df']:<15.6f} {results2['df']-results1['df']:>10.6f}")
    print(f"{'Iterations':<30} {results1['iterations']:<15} {results2['iterations']:<15} {results2['iterations']-results1['iterations']:>10}")
    
    # Voltage comparison
    v1_mean = np.mean(results1['ppc']['bus'][:, 7])  # VM column
    v2_mean = np.mean(results2['ppc']['bus'][:, 7])
    print(f"{'Mean voltage (p.u.)':<30} {v1_mean:<15.4f} {v2_mean:<15.4f} {v2_mean-v1_mean:>10.6f}")
    
    # Generation comparison
    p1_total = np.sum(results1['ppc']['gen'][:, 1])  # PG column
    p2_total = np.sum(results2['ppc']['gen'][:, 1])
    print(f"{'Total generation (MW)':<30} {p1_total:<15.2f} {p2_total:<15.2f} {p2_total-p1_total:>10.2f}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test 1: Basic functionality
    print("\n" + "🔧 "*20)
    print("STEP 1: Testing PyPower droop solver")
    print("🔧 "*20)
    results = test_pypower_droop()
    
    # Test 2: Deadband effect
    if results['converged']:
        print("\n\n" + "📊 "*20)
        print("STEP 2: Testing deadband effect")
        print("📊 "*20)
        test_with_and_without_deadband()