using PowerModels
using JuMP
using Ipopt
using PythonCall

"""
Main entry point called by Python.
Merges droop parameters into the network data and solves.
"""
function solve_ac_pf_droop(network_data, droop_params)
    println("\n=== Starting Droop Power Flow ===")
    
    # network_data is already a Julia dict from PowerModels.parse_file
    network_dict = network_data
    
    # Convert droop_params from Python dict to Julia dict
    droop_dict = pyconvert(Dict, droop_params)
        
    println("Network has:")
    println("  - $(length(network_dict["bus"])) buses")
    println("  - $(length(network_dict["gen"])) generators")
    println("  - $(length(network_dict["branch"])) branches")
    println("  - $(length(network_dict["load"])) loads")
    
    # 1. Inject droop parameters
    network_dict["droop_config"] = droop_dict

    # 2. Instantiate the custom model
    println("\nInstantiating model...")
    pm = instantiate_model(network_dict, ACPPowerModel, build_droop_pf)

    # 3. Solve using Ipopt
    println("\nSolving with Ipopt...")
    result = optimize_model!(pm, optimizer=optimizer_with_attributes(
        Ipopt.Optimizer, 
        "print_level"=>5,
        "max_iter"=>1000,
        "tol"=>1e-6
    ))

    println("\nTermination status: $(result["termination_status"])")
    
    # 4. Extract frequency if successful
    if result["termination_status"] == LOCALLY_SOLVED ||
       result["termination_status"] == ALMOST_LOCALLY_SOLVED
        try
            df_val = value(var(pm, :df))
            result["solution"]["frequency_deviation"] = df_val
            result["solution"]["system_frequency"] = droop_dict["omega_0"] + df_val
            result["solution"]["pf"]=true
            println("Frequency deviation: $df_val pu")
            
            # Calculate branch flows
            update_data!(network_dict, result["solution"])
            flows = calc_branch_flow_ac(network_dict)
            result["solution"]["branch"] = flows["branch"]
            println("Branch flows calculated")
        catch e
            println("Warning: Could not extract frequency deviation or calculate flows: $e")
        end
    else
        println("Solver did not converge successfully")
    end

    return result
end

"""
Deep conversion of Python objects to Julia native types
"""
function deep_convert_dict(obj)
    if pyisinstance(obj, pybuiltins.dict)
        result = Dict{String, Any}()
        for (k, v) in pyconvert(Dict, obj)
            key_str = string(k)
            result[key_str] = deep_convert_dict(v)
        end
        return result
    elseif pyisinstance(obj, pybuiltins.list)
        return [deep_convert_dict(item) for item in pyconvert(Vector, obj)]
    elseif pyisinstance(obj, pybuiltins.int)
        return pyconvert(Int, obj)
    elseif pyisinstance(obj, pybuiltins.float)
        return pyconvert(Float64, obj)
    elseif pyisinstance(obj, pybuiltins.str)
        return pyconvert(String, obj)
    elseif pyisinstance(obj, pybuiltins.bool)
        return pyconvert(Bool, obj)
    elseif pyisnone(obj)
        return nothing
    else
        # Try generic conversion
        try
            return pyconvert(Any, obj)
        catch
            return obj
        end
    end
end

"""
The Mathematical Formulation - Droop Power Flow
This implements the droop control equations from the reference paper.
"""
function build_droop_pf(pm::AbstractPowerModel)
    println("Building droop power flow model...")
    
    # 1. Define Standard Variables
    variable_bus_voltage(pm)
    variable_gen_power(pm)
    variable_branch_power(pm)
    
    println("  Variables defined")

    # 2. Define Global Frequency Deviation Variable
    var(pm)[:df] = @variable(pm.model, base_name="df", start=0.0, 
                             lower_bound=-0.1, upper_bound=0.1)
    
    println("  Frequency deviation variable created")

    # 3. Objective Function
    @objective(pm.model, Min, var(pm)[:df]^2)

    # 4. Get Droop Configuration
    droop_config = pm.data["droop_config"]
    
    droop_buses_raw = droop_config["droop_buses"]
    droop_buses = Set()
    for bus_id in droop_buses_raw
        try
            push!(droop_buses, parse(Int, string(bus_id)))
        catch
            push!(droop_buses, bus_id)
        end
    end
    
    mp_global = droop_config["mp"]
    mq_global = droop_config["mq"]
    omega_0 = droop_config["omega_0"]
    v_0 = droop_config["V_0"]
    
    println("  Droop config: buses=$droop_buses, mp=$mp_global, mq=$mq_global")

    # 5. Reference Bus Setup
    ref_bus_id = get(droop_config, "ref_bus", nothing)
    if !isnothing(ref_bus_id)
        try
            ref_bus_id = parse(Int, string(ref_bus_id))
        catch
        end
    else
        ref_bus_id = first(sort(collect(keys(ref(pm, :bus)))))
    end
    
    println("  Reference bus: $ref_bus_id")
    
    @constraint(pm.model, var(pm, :va, ref_bus_id) == 0)

    # 6. Branch Constraints
    for (i, branch) in ref(pm, :branch)
        constraint_ohms_yt_from(pm, i)
        constraint_ohms_yt_to(pm, i)
    end
    
    println("  Branch constraints added")

    # 7. Power Balance with Droop Control
    droop_gen_count = 0
    
    for (i, bus) in ref(pm, :bus)
        bus_gens = ref(pm, :bus_gens, i)
        bus_loads = ref(pm, :bus_loads, i)
        bus_shunts = ref(pm, :bus_shunts, i)
        
        pg = sum(var(pm, :pg, g) for g in bus_gens; init=0.0)
        qg = sum(var(pm, :qg, g) for g in bus_gens; init=0.0)
        
        pd = sum(ref(pm, :load, l, "pd") for l in bus_loads; init=0.0)
        qd = sum(ref(pm, :load, l, "qd") for l in bus_loads; init=0.0)
        
        gs = sum(ref(pm, :shunt, s, "gs") for s in bus_shunts; init=0.0)
        bs = sum(ref(pm, :shunt, s, "bs") for s in bus_shunts; init=0.0)
        
        vm = var(pm, :vm, i)
        
        p_expr = sum(var(pm, :p, (l,i,j)) for (l,i,j) in ref(pm, :bus_arcs, i))
        q_expr = sum(var(pm, :q, (l,i,j)) for (l,i,j) in ref(pm, :bus_arcs, i))
        
        has_droop = false
        for g in bus_gens
            gen = ref(pm, :gen, g)
            if gen["gen_bus"] in droop_buses
                has_droop = true
                break
            end
        end
        
        if has_droop
            df_var = var(pm)[:df]
            
            @constraint(pm.model, 
                pg - pd - gs * vm^2 == p_expr + (1/mp_global) * df_var
            )
            
            @constraint(pm.model, 
                qg - qd + bs * vm^2 == q_expr + (1/mq_global) * (v_0 - vm)
            )
            
            droop_gen_count += 1
        else
            @constraint(pm.model, 
                pg - pd - gs * vm^2 == p_expr
            )
            @constraint(pm.model, 
                qg - qd + bs * vm^2 == q_expr
            )
        end
    end
    
    println("  Power balance constraints added ($droop_gen_count buses with droop)")

    # 8. Generator Constraints
    for (i, gen) in ref(pm, :gen)
        pg_var = var(pm, :pg, i)
        qg_var = var(pm, :qg, i)
        
        @constraint(pm.model, pg_var >= gen["pmin"])
        @constraint(pm.model, pg_var <= gen["pmax"])
        @constraint(pm.model, qg_var >= gen["qmin"])
        @constraint(pm.model, qg_var <= gen["qmax"])
        
        bus_id = gen["gen_bus"]
        
        if !(bus_id in droop_buses)
            @constraint(pm.model, pg_var == gen["pg"])
            @constraint(pm.model, qg_var == gen["qg"])
        end
    end
    
    println("  Generator constraints added")

    # 9. Voltage Bounds
    for (i, bus) in ref(pm, :bus)
        vm_var = var(pm, :vm, i)
        vmin = get(bus, "vmin", 0.95)
        vmax = get(bus, "vmax", 1.05)
        set_lower_bound(vm_var, vmin)
        set_upper_bound(vm_var, vmax)
    end
    
    println("  Voltage bounds enforced")
    println("Model build complete!")
end