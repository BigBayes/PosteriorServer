#println("size of ARGS: $(size(ARGS))")
#println("type of ARGS: $(typeof(ARGS))")

# Check that we have a valid number of arguments
if size(ARGS, 1) != 1
	error("Usage: julia run.jl <experiment filename without extension>")
end
experiment_name = ARGS[1]

# This is the file you need to edit so that the script can access your data etc.
include("paths.jl")
push!(LOAD_PATH, "$(source_path)")

# Add new model factory files here, to be called by the experiment code
if !(contains(experiment_name,"logreg"))
	include("$(source_path)models.jl")
end

# Include the experiment variables
include("$(experiments_path)$(experiment_name).jl")

# Construct file name.
file_name = "$(dataset)_$(specs[:model])_$(method)_$(run_suffix)"


for ii = 1:length(varyvalues)
	vv = varyvalues[ii]
	file_name = "$(dataset)_$(specs[:model])_$(method)_$(run_suffix)_$(ii)_$(string(varyparam))=$(vv)"
	specs[varyparam] = vv
	println("=== Run $(ii): $(varyparam) = $(varyvalues[ii]) ===")
	# Run the experiment
	if method == "postserver"
		include("$(scripts_path)runMochaPosteriorServer.jl")
	elseif method == "logistic_regression"
		include("$(scripts_path)runLogRegPosteriorServer.jl")
	elseif method == "mxpostserver"
		include("$(scripts_path)runMXNetPosteriorServer.jl")
	elseif method == "downpour"
		include("$(scripts_path)runMochaDownpourSGD.jl")
	elseif method == "easgd"
		include("$(scripts_path)runMochaEASGD.jl")
	else
		error("Unknown experiment method in $(experiment_name).jl")
	end
end

# Produce default plots
if haskey(specs, :default_plot) && specs[:default_plot] == true
	println("Producing default figures...")
	if method == "postserver"
		include("$(scripts_path)default_plot_postserver.jl")
	elseif method == "downpour"
		include("$(scripts_path)default_plot_downpour.jl")
	elseif method == "easgd"
		include("$(scripts_path)default_plot_easgd.jl")
	end
end
