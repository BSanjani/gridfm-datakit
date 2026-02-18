# using PowerModels
# using JuMP
# using Ipopt
#using Statistics
# using PowerModels: var, ref

# function solve_ac_pf_unified(network_data, droop_params)
#     droop_enabled = get(droop_params, "enabled", false)
    
#     if droop_enabled
#         println("="^60)
#         println("MODE: DROOP CONTROL ENABLED")
#         println("="^60)
#         network_data["droop_config"] = droop_params
#         pm = instantiate_model(network_data, ACPPowerModel, build_droop_pf_fixed)
#     else
#         println("="^60)
#         println("MODE: TRADITIONAL SLACK BUS")
#         println("="^60)
#         pm = instantiate_model(network_data, ACPPowerModel, build_traditional_pf)
#     end
    
#     solver = optimizer_with_attributes(Ipopt.Optimizer, 
#         "print_level" => 5,
#         "sb" => "yes",
#         "max_iter" => 3000,
#         "tol" => 1e-6
#     )
    
#     result = optimize_model!(pm, optimizer=solver)
    
#     # Extract df and save it
#     if droop_enabled && haskey(var(pm), :df)
#         result["df"] = JuMP.value(var(pm, :df))
#         println("\n*** EXTRACTED df = $(result["df"]) ***")
#     end
    
#     return result
# end

# function build_droop_pf_fixed(pm::AbstractPowerModel)
#     variable_bus_voltage(pm)
#     variable_gen_power(pm)
#     variable_branch_power(pm)
    
#     droop_config = pm.data["droop_config"]
#     freq_deadband = get(droop_config, "frequency_deadband", 0.0)
#     volt_deadband = get(droop_config, "voltage_deadband", 0.0)
    
#     droop_buses = Set(parse(Int, bus_id) for bus_id in droop_config["droop_buses"])
#     mp_global = droop_config["mp"]
#     mq_global = droop_config["mq"]
#     v_ref_global = droop_config["V_0"]
    
#     # Reference angle constraint
#     if !isempty(droop_buses)
#         ref_bus = first(droop_buses)  # Or any bus with a droop gen
#         @constraint(pm.model, var(pm, :va, ref_bus) == 0)
#     else
#         # Fallback if no droop buses
#         ref_bus = first(keys(ref(pm, :bus)))
#         @constraint(pm.model, var(pm, :va, ref_bus) == 0)
#     end
    
#     # Frequency deviation variable
#     df_var = @variable(pm.model, base_name="df", start=0.0, lower_bound=-0.1, upper_bound=0.1)
#     var(pm)[:df] = df_var
    
#     # Compute df_eff globally (shared across all droop gens)
#     if freq_deadband > 1e-9
#         ε = 0.05
#         sharpness = 10.0
#         δ = 1e-6
#         df_abs = sqrt(df_var^2 + δ)
#         activation = 0.5 * (1 + tanh(sharpness * (df_abs - freq_deadband)))
#         df_eff = ε * df_var + (1 - ε) * df_var * activation
#     else
#         df_eff = df_var
#     end
    
#     # Objective: Minimize df^2 for small deviations (feasibility problem)
#     @objective(pm.model, Min, df_var^2)
    
#     # Power balance constraints
#     for (i, bus) in ref(pm, :bus)
#         constraint_power_balance(pm, i)
#     end
    
#     # Generator constraints
#     for (i, gen) in ref(pm, :gen)
#         bus_id = gen["gen_bus"]
#         pg_var = var(pm, :pg, i)
#         qg_var = var(pm, :qg, i)
#         vm_var = var(pm, :vm, bus_id)
#         p_set = gen["pg"]
#         q_set = gen["qg"]
        
#         if bus_id in droop_buses
#             # P-f droop (per generator)
#             @constraint(pm.model, pg_var == p_set - (1/mp_global) * df_eff)
            
#             # Q-V droop (with optional deadband)
#             dv = vm_var - v_ref_global
#             if volt_deadband > 1e-9
#                 ε = 0.05  # Same as freq
#                 sharpness = 10.0
#                 δ = 1e-6
#                 dv_abs = sqrt(dv^2 + δ)
#                 activation = 0.5 * (1 + tanh(sharpness * (dv_abs - volt_deadband)))
#                 dv_eff = ε * dv + (1 - ε) * dv * activation
#                 @constraint(pm.model, qg_var == q_set - (1/mq_global) * dv_eff)
#             else
#                 @constraint(pm.model, qg_var == q_set - (1/mq_global) * dv)
#             end
#         else
#             # Fixed setpoints for non-droop gens
#             constraint_gen_setpoint_active(pm, i)
#             constraint_gen_setpoint_reactive(pm, i)
#         end
#     end
    
#     # *** DIAGNOSTIC OUTPUT ***
#     println("\n" * "="^60)
#     println("DROOP FORMULATION DIAGNOSTICS")
#     println("="^60)
    
#     # Print load information
#     total_load_p = sum(load["pd"] for (i, load) in ref(pm, :load))
#     total_load_q = sum(load["qd"] for (i, load) in ref(pm, :load))
#     println("\nTotal System Load:")
#     println("  P_load = $(round(total_load_p, digits=4)) p.u.")
#     println("  Q_load = $(round(total_load_q, digits=4)) p.u.")
    
#     # Print generator setpoints
#     println("\nGenerator Setpoints (before droop):")
#     println(rpad("Gen", 6), "|", rpad("Bus", 6), "|", rpad("P_set", 10), "|", rpad("Q_set", 10), "|", rpad("Droop?", 8))
#     println("-"^50)
    
#     total_pset = 0.0
#     total_pset_droop = 0.0
    
#     for (i, gen) in ref(pm, :gen)
#         is_droop = gen["gen_bus"] in droop_buses ? "YES" : "NO"
#         println(rpad(string(i), 6), "|", rpad(string(gen["gen_bus"]), 6), "|", 
#                 rpad(string(round(gen["pg"], digits=4)), 10), "|", 
#                 rpad(string(round(gen["qg"], digits=4)), 10), "|", 
#                 rpad(is_droop, 8))
        
#         total_pset += gen["pg"]
#         if gen["gen_bus"] in droop_buses
#             total_pset_droop += gen["pg"]
#         end
#     end
    
#     println("-"^50)
#     println("Total P setpoint (all gens):   $(round(total_pset, digits=4)) p.u.")
#     println("Total P setpoint (droop gens): $(round(total_pset_droop, digits=4)) p.u.")
#     println("Power balance check: P_set - P_load = $(round(total_pset - total_load_p, digits=4)) p.u.")
    
#     imbalance_pct = 100.0 * (total_pset - total_load_p) / total_load_p
#     println("Imbalance: $(round(imbalance_pct, digits=2))%")
    
#     if abs(imbalance_pct) > 5.0
#         println("\n⚠️  WARNING: Large power imbalance (>5%)!")
#         println("   This will cause infeasibility in droop formulation.")
#         println("   The OPF setpoints don't match the actual load.")
#     end
    
#     println("\nDroop Constraint Info:")
#     println("  Number of droop gens: $(n_droop_gens)")
#     println("  mp (droop gain): $(mp_global)")
#     println("  Frequency deadband: $(freq_deadband) p.u.")
#     println("  Expected df range: ±$(round(mp_global * abs(total_pset - total_load_p) / n_droop_gens, digits=6)) p.u.")
#     println("="^60 * "\n")
    
#     # Branch constraints
#     for (i, branch) in ref(pm, :branch)
#         constraint_ohms_yt_from(pm, i)
#         constraint_ohms_yt_to(pm, i)
#         constraint_voltage_angle_difference(pm, i)
#         constraint_thermal_limit_from(pm, i)
#         constraint_thermal_limit_to(pm, i)
#     end
# end

# function build_traditional_pf(pm::AbstractPowerModel)
#     variable_bus_voltage(pm)
#     variable_gen_power(pm)
#     variable_branch_power(pm)
#     @objective(pm.model, Min, 0)
    
#     for (i, bus) in ref(pm, :bus)
#         constraint_power_balance(pm, i)
#     end
    
#     for (i, gen) in ref(pm, :gen)
#         if !gen["is_slack_gen"]
#             constraint_gen_setpoint_active(pm, i)
#             constraint_gen_setpoint_reactive(pm, i)
#         end
#     end
    
#     for (i, branch) in ref(pm, :branch)
#         constraint_ohms_yt_from(pm, i)
#         constraint_ohms_yt_to(pm, i)
#         constraint_voltage_angle_difference(pm, i)
#         constraint_thermal_limit_from(pm, i)
#         constraint_thermal_limit_to(pm, i)
#     end
# end

# export solve_ac_pf_unified



using PowerModels
using JuMP
using Ipopt
using Statistics
using PowerModels: var, ref
function solve_ac_pf_unified(network_data, droop_params)
    open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
        write(io, "\n=== Starting solve_ac_pf_unified ===\n")
    end
    
    try
        droop_enabled = get(droop_params, "enabled", false)
        
        if droop_enabled
            open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
                write(io, "="^60 * "\n")
                write(io, "MODE: DROOP CONTROL ENABLED\n")
                write(io, "="^60 * "\n")
            end
            network_data["droop_config"] = droop_params
            pm = instantiate_model(network_data, ACPPowerModel, build_droop_pf_fixed)
        else
            open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
                write(io, "="^60 * "\n")
                write(io, "MODE: TRADITIONAL SLACK BUS\n")
                write(io, "="^60 * "\n")
            end
            pm = instantiate_model(network_data, ACPPowerModel, build_traditional_pf)
        end
        
        solver = optimizer_with_attributes(Ipopt.Optimizer, 
            "print_level" => 10,
            "file_print_level" => 5,
            "output_file" => "ipopt_droop.log",
            "print_frequency_iter" => 1,
            "sb" => "yes",
            "max_iter" => 3000,
            "tol" => 1e-6
        )
        
        result = optimize_model!(pm, optimizer=solver)
        
        status = result["termination_status"]
        open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
            write(io, "\n*** OPTIMIZATION STATUS: $status ***\n")
            write(io, "Full result: $(result)\n")
        end
        
        if !(status in [LOCALLY_SOLVED, OPTIMAL])
            open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
                write(io, "\n*** INFEASIBILITY DETECTED - EXTRA DEBUG ***\n")
            end
            if droop_enabled && haskey(var(pm), :df)
                df_var = var(pm, :df)
                open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
                    write(io, "df start: $(start_value(df_var)), bounds: [$(lower_bound(df_var)), $(upper_bound(df_var))]\n")
                end
            end
            # Sample gen starts
            open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
                write(io, "Sample variable starts (first 5 gens pg):\n")
            end
            gens = ref(pm, :gen)
            for (idx, (i, gen)) in enumerate(gens)
                if idx > 5 break end
                pg_var = var(pm, :pg, i)
                open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
                    write(io, "Gen $i pg start: $(start_value(pg_var))\n")
                end
            end
        else
            if droop_enabled && haskey(var(pm), :df)
                df_val = JuMP.value(var(pm, :df))
                result["df"] = df_val
                open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
                    write(io, "\n*** EXTRACTED df = $df_val ***\n")
                end
            end
        end
        # Add 'pf' key for Python compatibility
        # Ensure solution dict exists and is mutable
        if !haskey(result, "solution")
            result["solution"] = Dict{String,Any}()
        end

        # Force pf into solution dict as a proper Julia Bool (not wrapped)
        pf_status = (result["termination_status"] in [LOCALLY_SOLVED, OPTIMAL])
        result["solution"]["pf"] = pf_status

        open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
            write(io, "\n*** SET result['solution']['pf'] = $pf_status ***\n")
            write(io, "solution keys: $(keys(result["solution"]))\n")
        end

        return result
                
    catch err
        open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
            write(io, "\n*** ERROR IN SOLVE: $(sprint(showerror, err)) ***\n")
        end
        rethrow(err)  # Or return a dummy result if needed
    end
end
function build_droop_pf_fixed(pm::AbstractPowerModel)

    open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
        write(io, "\n>>> ENTERED build_droop_pf_fixed <<<\n")
    end
    open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
        write(io, "Keys in pm.data: $(collect(keys(pm.data)))\n")
        write(io, "Has droop_config? $(haskey(pm.data, "droop_config"))\n")
    end

    # -----------------------------
    # 1. Standard PowerModels vars
    # -----------------------------
    variable_bus_voltage(pm)
    variable_gen_power(pm)
    variable_branch_power(pm)

    # -----------------------------
    # 2. Read droop configuration
    # -----------------------------
    if !haskey(pm.data, "droop_config")
        # Build standard PF if no droop config
        variable_bus_voltage(pm)
        variable_gen_power(pm)
        variable_branch_power(pm)
        
        for (i, bus) in ref(pm, :ref_buses)
            @constraint(pm.model, var(pm, :va, i) == 0)
        end
        
        constraint_model_voltage(pm)
        
        for (i, bus) in ref(pm, :bus)
            constraint_power_balance(pm, i)
        end
        
        for (i, gen) in ref(pm, :gen)
            constraint_gen_setpoint_active(pm, i)
            constraint_gen_setpoint_reactive(pm, i)
        end
        
        for (i, branch) in ref(pm, :branch)
            constraint_ohms_yt_from(pm, i)
            constraint_ohms_yt_to(pm, i)
            constraint_voltage_angle_difference(pm, i)
            constraint_thermal_limit_from(pm, i)
            constraint_thermal_limit_to(pm, i)
        end
        
        return
    end

    droop_config = pm.data["droop_config"]

    freq_deadband  = get(droop_config, "frequency_deadband", 0.0)
    volt_deadband  = get(droop_config, "voltage_deadband", 0.0)

    droop_buses = Set(parse(Int, b) for b in droop_config["droop_buses"])
    mp_fixed = get(droop_config, "mp", nothing)
    mq_fixed = get(droop_config, "mq", nothing)
    mp_min, mp_max = get(droop_config, "mp_range", [0.03, 0.05])
    mq_min, mq_max = get(droop_config, "mq_range", [0.02, 0.04])

    v_ref_global = droop_config["V_0"]

    # -----------------------------
    # 3. Scenario-based droop gains
    # -----------------------------
    mp_map = Dict{Int,Float64}()
    mq_map = Dict{Int,Float64}()

    for bus in droop_buses
        if mp_fixed !== nothing && mq_fixed !== nothing
            mp_map[bus] = Float64(mp_fixed)
            mq_map[bus] = Float64(mq_fixed)
        elseif droop_config["randomize_droop"]
            mp_map[bus] = rand() * (mp_max - mp_min) + mp_min
            mq_map[bus] = rand() * (mq_max - mq_min) + mq_min
        else
            mp_map[bus] = 0.5 * (mp_min + mp_max)
            mq_map[bus] = 0.5 * (mq_min + mq_max)
        end
    end

    pm.data["droop_mp"] = mp_map
    pm.data["droop_mq"] = mq_map

    # -----------------------------
    # 4. Reference angle
    # -----------------------------
    if haskey(droop_config, "ref_bus")
        ref_bus = parse(Int, droop_config["ref_bus"])
        @assert haskey(ref(pm, :bus), ref_bus)
    elseif !isempty(droop_buses)
        ref_bus = first(droop_buses)
    else
        # ref_bus = first(keys(ref(pm, :bus)))
        ref_bus = parse(Int, string(first(keys(ref(pm, :bus)))))
    end

    @constraint(pm.model, var(pm, :va, ref_bus) == 0)

    # -----------------------------
    # 5. Frequency deviation variable
    # -----------------------------
    df = @variable(
        pm.model,
        base_name = "df",
        start = 0.0,
        lower_bound = -0.1,
        upper_bound = 0.1
    )
    var(pm)[:df] = df

    # -----------------------------
    # 6. Effective frequency (deadband)
    # -----------------------------
    if freq_deadband > 1e-9
        ε = 1e-3
        sharpness = 10.0
        δ = 1e-6
        df_abs = sqrt(df^2 + δ)
        activation = 0.5 * (1 + tanh(sharpness * (df_abs - freq_deadband)))
        df_eff = ε * df + (1 - ε) * df * activation
    else
        df_eff = df
    end

    # -----------------------------
    # 7. No objective (pure PF)
    # -----------------------------
    @objective(pm.model, Min, 1000 * df^2)

    # -----------------------------
    # 8. Power balance constraints
    # -----------------------------
    for (i, _) in ref(pm, :bus)
        constraint_power_balance(pm, i)
    end

    # -----------------------------
    # 9. Generator constraints
    # -----------------------------
    for (i, gen) in ref(pm, :gen)

        bus_id = gen["gen_bus"]
        pg = var(pm, :pg, i)
        qg = var(pm, :qg, i)
        vm = var(pm, :vm, bus_id)

        p_set = gen["pg"]
        q_set = gen["qg"]

        if bus_id in droop_buses
            mp_i = pm.data["droop_mp"][bus_id]
            mq_i = pm.data["droop_mq"][bus_id]
            
            # Check if generator has headroom for droop
            pmin = gen["pmin"]
            pmax = gen["pmax"]
            headroom = pmax - p_set
            
            if headroom > 0.01  # Only droop if >1% headroom
                # Active power droop
                @constraint(pm.model,
                    pg == p_set - (1 / mp_i) * df_eff
                )

                # Voltage deviation
                dv = vm - v_ref_global

                if volt_deadband > 1e-9
                    ε = 1e-3
                    sharpness = 10.0
                    δ = 1e-6
                    dv_abs = sqrt(dv^2 + δ)
                    activation = 0.5 * (1 + tanh(sharpness * (dv_abs - volt_deadband)))
                    dv_eff = ε * dv + (1 - ε) * dv * activation
                else
                    dv_eff = dv
                end

                # Reactive droop
                @constraint(pm.model,
                    qg == q_set - (1 / mq_i) * dv_eff
                )
            else
                # At limit, fix setpoint
                constraint_gen_setpoint_active(pm, i)
            end
        end

    end

    # -----------------------------
    # 10. DEBUG LOGGING (build-time)
    # -----------------------------
    open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
        write(io, "\n" * "="^80 * "\n")
        write(io, "DROOP PF BUILD-TIME DEBUG TRACE\n")
        write(io, "="^80 * "\n")

        write(io, "\nDroop gains:\n")
        for b in sort(collect(droop_buses))
            write(io,
                "  Bus $b | mp=$(round(mp_map[b], digits=4)) | mq=$(round(mq_map[b], digits=4))\n"
            )
        end

        total_load_p = sum(load["pd"] for (_, load) in ref(pm, :load))
        total_pset   = sum(gen["pg"] for (_, gen) in ref(pm, :gen))

        write(io, "\nPower summary:\n")
        write(io, "  Total load P        = $(round(total_load_p, digits=4))\n")
        write(io, "  Total gen P set     = $(round(total_pset, digits=4))\n")
        write(io, "  Mismatch (Pset-L)   = $(round(total_pset - total_load_p, digits=4))\n")

        n_droop = length(droop_buses)
        avg_mp  = mean(values(mp_map))

        required_df = avg_mp * (total_load_p - total_pset) / max(n_droop, 1)

        write(io, "\nDroop feasibility check:\n")
        write(io, "  # droop generators  = $n_droop\n")
        write(io, "  average mp          = $(round(avg_mp, digits=4))\n")
        write(io, "  df bounds           = [-0.1, 0.1]\n")
        write(io, "  required df (≈)     = $(round(required_df, digits=6))\n")

        if abs(required_df) > 0.1
            write(io,
                "  ⚠️ REQUIRED df EXCEEDS BOUNDS → INFEASIBLE BY CONSTRUCTION\n"
            )
        end

        write(io, "\nImplied droop response:\n")
        for (_, gen) in ref(pm, :gen)
            b = gen["gen_bus"]
            if b in droop_buses
                mp_i = mp_map[b]
                Δp = -(1 / mp_i) * required_df
                write(io,
                    "  Bus $b | p_set=$(round(gen["pg"], digits=3)) | Δp≈$(round(Δp, digits=3))\n"
                )
            end
        end

        write(io, "="^80 * "\n")
    end

    # -----------------------------
    # 11. Pre-solve diagnostics
    # -----------------------------
    open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
        write(io, "\n" * "="^80 * "\n")
        write(io, "PRE-SOLVE CONSTRAINT CHECK\n")
        write(io, "="^80 * "\n")
        
        # Check if droop constraints can be satisfied
        for (i, gen) in ref(pm, :gen)
            bus_id = gen["gen_bus"]
            if bus_id in droop_buses
                p_set = gen["pg"]
                q_set = gen["qg"]
                pmin = gen["pmin"]
                pmax = gen["pmax"]
                qmin = gen["qmin"]
                qmax = gen["qmax"]
                
                write(io, "\nGen $i (Bus $bus_id):\n")
                write(io, "  P: set=$(round(p_set, digits=3)), limits=[$(round(pmin, digits=3)), $(round(pmax, digits=3))]\n")
                write(io, "  Q: set=$(round(q_set, digits=3)), limits=[$(round(qmin, digits=3)), $(round(qmax, digits=3))]\n")
            end
        end
        write(io, "="^80 * "\n")
    end

    # -----------------------------
    # 11. Branch constraints
    # -----------------------------
    for (i, _) in ref(pm, :branch)
        constraint_ohms_yt_from(pm, i)
        constraint_ohms_yt_to(pm, i)
        constraint_voltage_angle_difference(pm, i)
        constraint_thermal_limit_from(pm, i)
        constraint_thermal_limit_to(pm, i)
    end
end

function solve_droop_pf(network::Dict{String,Any}, optimizer)
    open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
        write(io, "\n=== ENTERED solve_droop_pf ===\n")
        write(io, "Keys in network: $(collect(keys(network)))\n")
        write(io, "Has droop_config? $(haskey(network, "droop_config"))\n")
    end
    if haskey(network, "droop_config")
        pm = instantiate_model(network, ACPPowerModel, build_droop_pf_fixed)
        # Log key droop state at each Ipopt iteration for infeasibility diagnosis.
        try
            base_mva = Float64(get(network, "baseMVA", 100.0))
            droop_cfg = network["droop_config"]
            droop_buses = Set(parse(Int, b) for b in droop_cfg["droop_buses"])
            v0_cfg = get(droop_cfg, "V_0", missing)
            mp_cfg = get(droop_cfg, "mp", missing)
            mq_cfg = get(droop_cfg, "mq", missing)
            iter_log_file = "C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop_iter.log"
            open(iter_log_file, "a") do io
                write(io, "\n=== NEW DROOP SOLVE ===\n")
                write(io, "baseMVA=$base_mva droop_buses=$(collect(droop_buses))\n")
                write(io, "V0=$v0_cfg mp=$mp_cfg mq=$mq_cfg\n")
            end
            JuMP.set_attribute(pm.model, Ipopt.CallbackFunction(), (
                alg_mod,
                iter_count,
                obj_value,
                inf_pr,
                inf_du,
                mu,
                d_norm,
                regularization_size,
                alpha_du,
                alpha_pr,
                ls_trials,
            ) -> begin
                parts = String[]
                for (i, gen) in ref(pm, :gen)
                    bus_id = gen["gen_bus"]
                    if bus_id in droop_buses
                        pg_pu = JuMP.callback_value(pm.model, var(pm, :pg, i))
                        qg_pu = JuMP.callback_value(pm.model, var(pm, :qg, i))
                        vm_pu = JuMP.callback_value(pm.model, var(pm, :vm, bus_id))
                        push!(
                            parts,
                            "gen=$(i) bus=$(bus_id) pgMW=$(round(pg_pu * base_mva, digits=4)) qgMVAr=$(round(qg_pu * base_mva, digits=4)) vm=$(round(vm_pu, digits=6))",
                        )
                    end
                end
                df_val = haskey(var(pm), :df) ? JuMP.callback_value(pm.model, var(pm, :df)) : NaN
                open(iter_log_file, "a") do io
                    write(
                        io,
                        "iter=$(iter_count) obj=$(obj_value) inf_pr=$(inf_pr) inf_du=$(inf_du) mu=$(mu) df=$(df_val) ",
                    )
                    write(io, join(parts, " | ") * "\n")
                end
                return true
            end)
        catch err
            open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
                write(io, "Failed to attach Ipopt iteration callback: $(sprint(showerror, err))\n")
            end
        end
        result = optimize_model!(pm, optimizer=optimizer)
    else
        # No droop config, use standard PF
        result = solve_ac_pf(network, optimizer)
    end
    
    # Add 'pf' key for Python compatibility
    if !haskey(result, "solution")
        result["solution"] = Dict{String,Any}()
    end
    pf_status = (result["termination_status"] in [LOCALLY_SOLVED, OPTIMAL])
    result["solution"]["pf"] = pf_status

    # Extract df if droop is enabled
    if haskey(network, "droop_config") && haskey(var(pm), :df)
        df_value = JuMP.value(var(pm, :df))
        result["solution"]["frequency_deviation"] = df_value
        
        # Calculate actual frequency (assuming base = 60 Hz)
        f_nominal = get(network["droop_config"], "f_nominal", 60.0)
        frequency = f_nominal + df_value * f_nominal
        
        open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
            write(io, "\n*** FREQUENCY RESULTS ***\n")
            write(io, "df (deviation): $df_value p.u.\n")
            write(io, "Frequency: $frequency Hz\n")
            write(io, "*************************\n")
        end
    end
        
    open("C:/Users/Bestu/Documents/GitHub/gridfm-datakit/debug_droop.log", "a") do io
        write(io, "*** Added pf key: $pf_status ***\n")
    end
    
    return result
end


function build_traditional_pf(pm::AbstractPowerModel)
    variable_bus_voltage(pm)
    variable_gen_power(pm)
    variable_branch_power(pm)
    @objective(pm.model, Min, 0)
    
    for (i, bus) in ref(pm, :bus)
        constraint_power_balance(pm, i)
    end
    
    for i in ref(pm, :ref_buses)
        @constraint(pm.model, var(pm, :va, i) == 0)
        @constraint(pm.model, var(pm, :vm, i) == ref(pm, :bus, i)["vm"])
    end
    
    for (i, gen) in ref(pm, :gen)
        if !(gen["gen_bus"] in ref(pm, :ref_buses))
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
