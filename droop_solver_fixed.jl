using PowerModels
using JuMP
using Ipopt

function solve_ac_pf_unified(network_data, droop_params)
    droop_enabled = get(droop_params, "enabled", false)
    
    if droop_enabled
        println("="^60)
        println("MODE: DROOP CONTROL ENABLED")
        println("="^60)
        network_data["droop_config"] = droop_params
        pm = instantiate_model(network_data, ACPPowerModel, build_droop_pf_fixed)
    else
        println("="^60)
        println("MODE: TRADITIONAL SLACK BUS")
        println("="^60)
        pm = instantiate_model(network_data, ACPPowerModel, build_traditional_pf)
    end
    
    solver = optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 5,
        "sb" => "yes",
        "max_iter" => 3000,
        "tol" => 1e-6
    )
    
    result = optimize_model!(pm, optimizer=solver)
    
    # Extract df and save it
    if droop_enabled && haskey(var(pm), :df)
        result["df"] = JuMP.value(var(pm, :df))
        println("\n*** EXTRACTED df = $(result["df"]) ***")
    end
    
    return result
end

function build_droop_pf_fixed(pm::AbstractPowerModel)
    variable_bus_voltage(pm)
    variable_gen_power(pm)
    variable_branch_power(pm)
    
    droop_config = pm.data["droop_config"]
    freq_deadband = get(droop_config, "frequency_deadband", 0.0)
    
    droop_buses = Set(parse(Int, bus_id) for bus_id in droop_config["droop_buses"])
    mp_global = droop_config["mp"]
    mq_global = droop_config["mq"]
    v_ref_global = droop_config["V_0"]
    
    # Frequency deviation variable
    df_var = @variable(pm.model, base_name="df", start=0.0, lower_bound=-0.1, upper_bound=0.1)
    var(pm)[:df] = df_var
    
    # *** FIX: DO NOT MINIMIZE df! ***
    # The droop constraint determines df, not the objective
    # Use feasibility objective or minimize generation cost
    @objective(pm.model, Min, sum(var(pm, :pg, i)^2 for (i,gen) in ref(pm, :gen)) * 0.0001)
    
    # Power balance
    for (i, bus) in ref(pm, :bus)
        constraint_power_balance(pm, i)
    end
    
    # Generator constraints
    for (i, gen) in ref(pm, :gen)
        bus_id = gen["gen_bus"]
        
        if bus_id in droop_buses
            # Q-V droop only (P is handled globally)
            qg_var = var(pm, :qg, i)
            vm_var = var(pm, :vm, bus_id)
            q_set = gen["qg"]
            dv = vm_var - v_ref_global
            @constraint(pm.model, qg_var == q_set - (1/mq_global) * dv)
        else
            # Fixed setpoints
            constraint_gen_setpoint_active(pm, i)
            constraint_gen_setpoint_reactive(pm, i)
        end
    end
    
    # *** KEY: P-f droop constraint (this determines df!) ***
    total_droop_pg = sum(var(pm, :pg, i) for (i, gen) in ref(pm, :gen) if gen["gen_bus"] in droop_buses)
    total_setpoint_pg = sum(gen["pg"] for (i, gen) in ref(pm, :gen) if gen["gen_bus"] in droop_buses)
    n_droop_gens = length(droop_buses)
    
    if freq_deadband > 1e-9
        ε = 0.05
        sharpness = 10.0
        δ = 1e-6
        df_abs = sqrt(df_var^2 + δ)
        activation = 0.5 * (1 + tanh(sharpness * (df_abs - freq_deadband)))
        df_eff = ε * df_var + (1 - ε) * df_var * activation
        
        @constraint(pm.model, total_droop_pg == total_setpoint_pg - (n_droop_gens / mp_global) * df_eff)
    else
        @constraint(pm.model, total_droop_pg == total_setpoint_pg - (n_droop_gens / mp_global) * df_var)
    end
    
    # Branch constraints
    for (i, branch) in ref(pm, :branch)
        constraint_ohms_yt_from(pm, i)
        constraint_ohms_yt_to(pm, i)
        constraint_voltage_angle_difference(pm, i)
        constraint_thermal_limit_from(pm, i)
        constraint_thermal_limit_to(pm, i)
    end
end

function build_traditional_pf(pm::AbstractPowerModel)
    variable_bus_voltage(pm)
    variable_gen_power(pm)
    variable_branch_power(pm)
    @objective(pm.model, Min, 0)
    
    for (i, bus) in ref(pm, :bus)
        constraint_power_balance(pm, i)
    end
    
    for (i, gen) in ref(pm, :gen)
        if !gen["is_slack_gen"]
            constraint_gen_setpoint_active(pm, i)
            constraint_gen_setpoint_reactive(pm, i)
        end
    end
    
    for (i, branch) in ref(pm, :branch)
        constraint_ohms_yt_from(pm, i)
        constraint_ohms_yt_to(pm, i)
        constraint_voltage_angle_difference(pm, i)
        constraint_thermal_limit_from(pm, i)
        constraint_thermal_limit_to(pm, i)
    end
end

export solve_ac_pf_unified