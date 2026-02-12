"""PyPower Droop Control Solver - Fixed to match Julia"""
import numpy as np
from pypower.api import ppoption, runpf
from pypower.idx_bus import PD, QD, VM, VA, BUS_I
from pypower.idx_gen import GEN_BUS, PG, QG, QMAX, QMIN, PMAX, PMIN
import copy


class PyPowerDroopSolver:
    """Droop control solver for PyPower"""
    
    def __init__(self, ppc, droop_config):
        self.ppc = copy.deepcopy(ppc)
        self.config = droop_config
        self.baseMVA = ppc['baseMVA']
        
        # Find droop generators
        self.droop_gen_idx = []
        for bus_num in droop_config['droop_buses']:
            gen_idx = np.where(self.ppc['gen'][:, GEN_BUS] == bus_num)[0]
            self.droop_gen_idx.extend(gen_idx.tolist())
        
        self.droop_gen_idx = np.array(self.droop_gen_idx)
        
        # Store original setpoints
        self.Pg_setpoint = self.ppc['gen'][:, PG].copy() / self.baseMVA
        self.Qg_setpoint = self.ppc['gen'][:, QG].copy() / self.baseMVA
        
        # Get per-generator droop coefficients
        self.gen_droop_map = droop_config.get('gen_droop_map', {})
        
        # Filter generators with headroom
        self.active_droop_gen_idx = []
        for idx in self.droop_gen_idx:
            pmax = self.ppc['gen'][idx, PMAX] / self.baseMVA
            pg_set = self.Pg_setpoint[idx]
            headroom = pmax - pg_set
            if headroom < 0.001:
                print(f"    Gen {idx}: pg={pg_set:.3f}, pmax={pmax:.3f}, headroom={headroom:.4f}")
            if headroom > 0.001:  # Only if >1% headroom
                self.active_droop_gen_idx.append(idx)
        
        self.active_droop_gen_idx = np.array(self.active_droop_gen_idx)
        
        print(f"Initialized PyPower Droop Solver:")
        print(f"  Droop buses: {droop_config['droop_buses']}")
        print(f"  Total droop generators: {len(self.droop_gen_idx)}")
        print(f"  Active droop generators (with headroom): {len(self.active_droop_gen_idx)}")
        print(f"  Default mp = {droop_config['mp']}, mq = {droop_config['mq']}")
    
    def solve(self, verbose=True):
        if not self.config['enabled']:
            ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
            results_ppc = runpf(self.ppc, ppopt)[0]
            return {'ppc': results_ppc, 'converged': results_ppc['success'], 'df': 0.0, 'iterations': 0}
        
        if verbose:
            print("\n" + "="*60)
            print("PYPOWER DROOP CONTROL SOLVER")
            print("="*60)
            
            # ADD THIS DEBUG BLOCK
            print("\nDEBUG - Initial State:")
            print(f"  Pg_setpoint (first 5): {self.Pg_setpoint[:5]}")
            print(f"  Current Pg (first 5):  {self.ppc['gen'][:5, PG] / self.baseMVA}")
            print(f"  Total Pg_setpoint: {np.sum(self.Pg_setpoint):.4f} p.u.")
            print(f"  Total current Pg:  {np.sum(self.ppc['gen'][:, PG]) / self.baseMVA:.4f} p.u.")
            print(f"  Total load:        {np.sum(self.ppc['bus'][:, PD]) / self.baseMVA:.4f} p.u.")
            print(f"  Are setpoints same as current? {np.allclose(self.Pg_setpoint, self.ppc['gen'][:, PG] / self.baseMVA)}")
            print()
        
        df = 0.0
        max_iter = 50
        tol = 1e-6
        
        for iteration in range(max_iter):
            # Apply P-f droop only to active generators
            df_eff = self._apply_deadband(df)
            
            for idx in self.active_droop_gen_idx:
                # Get per-generator mp
                mp_i = self.gen_droop_map.get(idx, {}).get('mp', self.config['mp'])
                
                # Apply droop: pg = pg_set - (1/mp) * df_eff
                new_pg = self.Pg_setpoint[idx] - (1 / mp_i) * df_eff
                new_pg = np.clip(new_pg, 
                                self.ppc['gen'][idx, PMIN] / self.baseMVA,
                                self.ppc['gen'][idx, PMAX] / self.baseMVA)
                self.ppc['gen'][idx, PG] = new_pg * self.baseMVA
            
            # Run power flow
            ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
            results_ppc, success = runpf(self.ppc, ppopt)
            
            if not success:
                print(f"Warning: Power flow did not converge at iteration {iteration+1}")
                break
            
            # Apply Q-V droop only to active generators
            for idx in self.active_droop_gen_idx:
                # Get per-generator mq
                mq_i = self.gen_droop_map.get(idx, {}).get('mq', self.config['mq'])
                
                bus_num = int(results_ppc['gen'][idx, GEN_BUS])
                bus_idx = np.where(results_ppc['bus'][:, BUS_I] == bus_num)[0][0]
                V_bus = results_ppc['bus'][bus_idx, VM]
                dV = V_bus - self.config['V_0']
                
                new_qg = self.Qg_setpoint[idx] - (1 / mq_i) * dV
                new_qg = np.clip(new_qg,
                                results_ppc['gen'][idx, QMIN] / self.baseMVA,
                                results_ppc['gen'][idx, QMAX] / self.baseMVA)
                self.ppc['gen'][idx, QG] = new_qg * self.baseMVA
            
            # Calculate new df from active droop generators
            total_Pg_droop = np.sum(results_ppc['gen'][self.active_droop_gen_idx, PG]) / self.baseMVA
            total_Pg_set = np.sum(self.Pg_setpoint[self.active_droop_gen_idx])
            
            # Average mp for active generators
            mp_avg = np.mean([self.gen_droop_map.get(idx, {}).get('mp', self.config['mp']) 
                             for idx in self.active_droop_gen_idx])
            n_active = len(self.active_droop_gen_idx)
            
            df_new = -(mp_avg / n_active) * (total_Pg_droop - total_Pg_set)
            
            # Check convergence
            df_error = abs(df_new - df)
            if df_error < tol:
                if verbose:
                    print(f"Converged at iteration {iteration+1}")
                    print(f"  df = {df_new:.6f} p.u.")
                break
            
            # Update with relaxation
            df = 0.3 * df_new + 0.7 * df
        
        ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        final_ppc, success = runpf(self.ppc, ppopt)
        
        if verbose:
            self._print_results(final_ppc, df)
        
        return {'ppc': final_ppc, 'converged': success, 'df': df, 'iterations': iteration + 1}
    
    def _apply_deadband(self, df):
        deadband = self.config['frequency_deadband']
        if deadband < 1e-9:
            return df
        epsilon = 0.05
        sharpness = 10.0
        delta = 1e-6
        df_abs = np.sqrt(df**2 + delta)
        activation = 0.5 * (1 + np.tanh(sharpness * (df_abs - deadband) / (deadband + delta)))
        return epsilon * df + (1 - epsilon) * df * activation
    
    def _print_results(self, ppc, df):
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"\nFrequency deviation: {df:.6f} p.u.")
        if abs(df) <= self.config['frequency_deadband']:
            print("Status: INSIDE DEADBAND")
        else:
            print("Status: OUTSIDE DEADBAND")
        
        total_gen = np.sum(ppc['gen'][:, PG])
        total_load = np.sum(ppc['bus'][:, PD])
        print(f"\nTotal Generation: {total_gen:.2f} MW")
        print(f"Total Load: {total_load:.2f} MW")
        print(f"Losses: {total_gen - total_load:.2f} MW")
        print("="*60 + "\n")