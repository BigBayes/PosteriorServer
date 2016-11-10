using JLD

experiment_names = [ARGS[1]]
# Includes
# NOTE: Notice relative path below, e.g. works when in \results folder
include("../scripts/paths.jl")
push!(LOAD_PATH, "$(source_path)")
include("$(source_path)models.jl")

# Include the experiment variables
# TODO: Loop over elements of experiment_names
include("$(experiments_path)$(experiment_names[1]).jl")

# Most of specs
if dataset == "mnist"
    datasize = 60000
elseif dataset == "cifar_10_gcn_zca"
    datasize = 50000
else
    error("Unknown data set $(dataset)")
end

println("Loading libraries...")
start_time = time()
include("$(source_path)read_easgd_output.jl")
include("$(source_path)read_distbayes_output.jl")
include("$(source_path)read_downpour_output.jl")
using MLUtilities
println("$((time() - start_time)/60) minutes\n")

#=println("Loading Gadfly and Colors")
start_time = time()
using Gadfly
using Colors
println("$((time() - start_time)/60) minutes\n")=#

# Load in data files
unique_varyvalues =  convert(Array{Float64, 1}, unique(varyvalues))
times = Dict{Float64, Array{Any, 1}}()
iters = Dict{Float64, Array{Any, 1}}()
acc = Dict{Float64, Array{Any, 1}}()

for ii = 1:length(unique_varyvalues)
	times[unique_varyvalues[ii]] = cell(0)
	iters[unique_varyvalues[ii]] = cell(0)
	acc[unique_varyvalues[ii]] = cell(0)
end

println("Started reading data...")
start_time = time()

file_suffix = "$(dataset)_$(specs[:model])_$(method)_$(run_suffix)_"
for ii = 1:length(varyvalues)
	vv = varyvalues[ii]
	file_name = "$(file_suffix)$(ii)_$(string(varyparam))=$(vv)"
	specs[varyparam] = vv

    println("$(ii)/$(length(varyvalues))")

    if method == "easgd"
	   dbo = read_elastic_output("$(load_result_path)$(file_name).jld")
	   push!(acc[vv], dbo[:acc])
	   push!(times[vv], dbo[:times])
	   push!(iters[vv], dbo[:worker_iters] * specs[:batchsize] / datasize)
    elseif method == "downpour"
        dbo = read_downpour_output("$(load_result_path)$(file_name).jld")
        push!(acc[vv], dbo[:acc])
        push!(times[vv], dbo[:times])
        push!(iters[vv], dbo[:worker_iters] * specs[:batchsize] / datasize)
    elseif method == "postserver"
        dbo = read_distbayes_output("$(load_result_path)$(file_name).jld")
        push!(acc[vv], dbo[:acc_avgparams])
        push!(times[vv], dbo[:times_dist])
        push!(iters[vv], dbo[:worker_iters_dist] * specs[:batchsize] / datasize)
    else
        error("Unknown method $(method)")
    end
end

println("Finished reading data...")
println("$((time() - start_time)/60) minutes\n")

println("Saving data...")
start_time = time()

jldopen("$(save_result_path)$(file_suffix)all.jld", "w") do file
    write(file, "method", method)
    write(file, "dataset", dataset)
    write(file, "run_suffix", run_suffix)
    write(file, "specs", specs)
    write(file, "varyparam", varyparam)
    write(file, "varyvalues", varyvalues)

    write(file, "acc", acc)
    write(file, "times", times)
    write(file, "iters", iters)
end

println("$((time() - start_time)/60) minutes\n")