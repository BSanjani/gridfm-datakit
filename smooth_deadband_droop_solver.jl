function build_droop_pf(pm::AbstractPowerModel)
    # =========================================================================
    # DROOP-CONTROLLED POWER FLOW WITH SMOOTH DEADBAND (NO BINARY VARIABLES)
    # Uses smooth approximation compatible with NLP solvers like Ipopt
    # =========================================================================
    
    # 1. Standard PowerModels Variables
    variable_bus_voltage(pm)
    variable_gen_power(pm)
    variable_branch_power(pm)  # CRITICAL: Must have branch variables

    # 2. Get Configuration
    droop_config = pm.data["droop_config"]
    secondary_enabled = haskey(droop_config, "secondary_control") && droop_config["secondary_control"]["enabled"]
    
    # Get deadband values
    freq_deadband = get(droop_config, "frequency_deadband", 0.0)
    volt_deadband = get(droop_config, "voltage_deadband", 0.0)
    
    # Droop parameters
    droop_buses = Set(droop_config["droop_buses"])
    mp_global = droop_config["mp"]
    mq_global = droop_config["mq"]
    v_ref_global = droop_config["V_0"]

    # 3. Define Global Frequency Variable
    var(pm)[:df] = @variable(pm.model, base_name="df", start=0.0, lower_bound=-0.1, upper_bound=0.1)

    # 4. Define Secondary Control Variables
    if secondary_enabled
        var(pm)[:integral_error] = @variable(pm.model, base_name="integral_error", start=0.0, lower_bound=-1.0, upper_bound=1.0)
        var(pm)[:P_secondary] = Dict()
        for (i, gen) in ref(pm, :gen)
            var(pm)[:P_secondary][i] = @variable(pm.model, base_name="P_secondary_$(i)", start=0.0)
        end
    end

    # 5. Objective Function
    if secondary_enabled
        w_p = get(droop_config["secondary_control"], "w_primary", 1.0)
        w_s = get(droop_config["secondary_control"], "w_secondary", 0.1)
        @objective(pm.model, Min, w_p * var(pm, :df)^2 + w_s * var(pm, :integral_error)^2)
    else
        @objective(pm.model, Min, var(pm, :df)^2)
    end

    # 6. Constraints
    
    # A. Bus Power Balance
    for (i, bus) in ref(pm, :bus)
        constraint_power_balance(pm, i)
    end

    # B. Frequency-Integral Relationship
    if secondary_enabled
        T_s = droop_config["secondary_control"]["time_constant"]
        @constraint(pm.model, var(pm, :integral_error) == var(pm, :df) * T_s)
    end

    # C. Generator Droop Control with SMOOTH Deadband
    df_var = var(pm, :df)
    
    # Smoothing parameter (smaller = sharper transition, larger = smoother)
    epsilon = 1e-6
    
    for (i, gen) in ref(pm, :gen)
        bus_id = gen["gen_bus"]
        pg_var = var(pm, :pg, i)
        qg_var = var(pm, :qg, i)
        vm_var = var(pm, :vm, bus_id)

        if bus_id in droop_buses
            p_set = gen["pg"]
            q_set = gen["qg"]
            
            # =================================================================
            # ACTIVE POWER (P-f) DROOP WITH SMOOTH DEADBAND
            # =================================================================
            if freq_deadband > 0.0
                # Smooth deadband function: df_eff = sign(df) * max(0, |df| - deadband)
                # Using smooth approximation:
                # df_eff = df * max(0, (|df| - deadband)) / (|df| + epsilon)
                
                # Create nonlinear constraint for smooth deadband
                if secondary_enabled
                    p_sec_var = var(pm, :P_secondary)[i]
                    @NLconstraint(pm.model, 
                        pg_var == p_set - (1/mp_global) * (
                            df_var * max(0, (sqrt(df_var^2 + epsilon) - freq_deadband)) / 
                            (sqrt(df_var^2 + epsilon) + epsilon)
                        ) + p_sec_var
                    )
                else
                    @NLconstraint(pm.model, 
                        pg_var == p_set - (1/mp_global) * (
                            df_var * max(0, (sqrt(df_var^2 + epsilon) - freq_deadband)) / 
                            (sqrt(df_var^2 + epsilon) + epsilon)
                        )
                    )
                end
                
            else
                # No frequency deadband - standard P-f droop
                if secondary_enabled
                    p_sec_var = var(pm, :P_secondary)[i]
                    @constraint(pm.model, pg_var == p_set - (1/mp_global) * df_var + p_sec_var)
                else
                    @constraint(pm.model, pg_var == p_set - (1/mp_global) * df_var)
                end
            end

            # =================================================================
            # REACTIVE POWER (Q-V) DROOP WITH SMOOTH DEADBAND
            # =================================================================
            if volt_deadband > 0.0
                # Voltage deviation
                dv = vm_var - v_ref_global
                
                # Smooth deadband for voltage
                @NLconstraint(pm.model, 
                    qg_var == q_set - (1/mq_global) * (
                        dv * max(0, (sqrt(dv^2 + epsilon) - volt_deadband)) / 
                        (sqrt(dv^2 + epsilon) + epsilon)
                    )
                )
                
            else
                # No voltage deadband - standard Q-V droop
                @constraint(pm.model, qg_var == q_set - (1/mq_global) * (vm_var - v_ref_global))
            end
            
        else
            # Non-droop generators: fixed setpoints
            constraint_gen_setpoint_active(pm, i)
            constraint_gen_setpoint_reactive(pm, i)
        end
    end
    
    # D. Secondary Control Constraint
    if secondary_enabled
        K_I = droop_config["secondary_control"]["K_I"]
        int_err_var = var(pm, :integral_error)
        
        total_secondary = @expression(pm.model, sum(var(pm, :P_secondary)[i] for (i, gen) in ref(pm, :gen) if gen["gen_bus"] in droop_buses))
        @constraint(pm.model, total_secondary == K_I * int_err_var)
    end
    
    # E. Branch Flow Constraints
    for (i, branch) in ref(pm, :branch)
        constraint_ohms_yt_from(pm, i)
        constraint_ohms_yt_to(pm, i)
        constraint_voltage_angle_difference(pm, i)
        constraint_thermal_limit_from(pm, i)
        constraint_thermal_limit_to(pm, i)
    end
end

# =============================================================================
# SMOOTH DEADBAND FUNCTION EXPLAINED:
# =============================================================================
# 
# Mathematical form:
#   df_eff = df * max(0, |df| - deadband) / |df|
# 
# Smooth approximation:
#   |df| ≈ sqrt(df² + ε)  where ε = 1e-6
#   df_eff = df * max(0, sqrt(df² + ε) - deadband) / (sqrt(df² + ε) + ε)
# 
# Behavior:
#   - When |df| < deadband: df_eff ≈ 0 (inside deadband)
#   - When |df| > deadband: df_eff ≈ df - sign(df)*deadband (outside deadband)
#   - Smooth and differentiable everywhere (Ipopt can handle it!)
# 
# Example with freq_deadband = 0.001:
#   df = 0.0005 → df_eff ≈ 0 (inside deadband)
#   df = 0.002  → df_eff ≈ 0.001 (outside deadband, response starts)
#   df = -0.003 → df_eff ≈ -0.002
# 
# =============================================================================
