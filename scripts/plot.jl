# Generic file to plot extracted experiment data
using JLD

data_name = ARGS[1]

# Includes
# NOTE: Notice relative path below, e.g. works when in \results folder
include("../scripts/paths.jl")
push!(LOAD_PATH, "$(source_path)")
include("$(source_path)models.jl")

# Read the experiment data
file = jldopen("$(save_result_path)$(data_name)_all.jld", "r")
method = read(file, "method")
dataset = read(file, "dataset")
run_suffix = read(file, "run_suffix")
specs = read(file, "specs")
varyparam = read(file, "varyparam")
varyvalues = read(file, "varyvalues")
acc = read(file, "acc")
times = read(file, "times")
iters = read(file, "iters")
close(file);

# Most of specs
if dataset == "mnist"
    datasize = 60000
    ymin = 1.
    ymax = 5.
elseif dataset == "cifar_10_gcn_zca"
    datasize = 50000
    ymin = 20.
    ymax = 40.
else
    error("Unknown data set $(dataset)")
end

println("Loading libraries...")
start_time = time()
using MLUtilities
println("$((time() - start_time)/60) minutes\n")

println("Loading Gadfly and Colors")
start_time = time()
using Gadfly
using Colors
println("$((time() - start_time)/60) minutes\n")

# Load in data files
unique_varyvalues =  convert(Array{Float64, 1}, unique(varyvalues))

println("Started plotting...")
start_time = time()

colors = Colors.linspace(colorant"red", colorant"blue", length(unique_varyvalues))
#colors = Colors.linspace(colorant"black", colorant"black", length(unique_varyvalues))

# Create an array of layers
layers_avgparams = cell(0)

for ii = 1:length(unique_varyvalues)

	vv = unique_varyvalues[ii]

    if vv == 0.0025/6

	this_times = times[vv]
	this_acc = acc[vv]
	this_iters = iters[vv]

	# Plot the interpolated runs
  	(xvals, yvals) = MLUtilities.interpolatedAverage(this_iters, this_acc)
  	push!(layers_avgparams, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colors[ii], line_width=1pt)))

	for kk = 1:length(this_times)
		push!(layers_avgparams, layer(x = this_iters[kk], y = (1.0-this_acc[kk])*100, Geom.line, Theme(default_color = colors[ii], line_width=0.1pt)))
		#push!(layers_avgparams, layer(x = this_iters[kk], y = (1.0-this_acc[kk])*100, Geom.line, Theme(default_color = colorant"grey", line_width=0.1pt)))
	end

    end
end

Gadfly.set_default_plot_size(20cm,15cm)
colorkey = Guide.manual_color_key(string(varyparam), [string(x) for x in unique_varyvalues], colors)

plot_avgparams = plot(
				layers_avgparams...,
				Theme(background_color = colorant"white"),
				Guide.xlabel("epochs per worker"),
				Guide.ylabel("test error in %"),
				Guide.title("$(dataset) $(specs[:model]), $(method)"),
				colorkey,
				Coord.Cartesian(ymin = 0., ymax = 100.))

plot_avgparams_zoomed = plot(
				layers_avgparams...,
				Theme(background_color = colorant"white"),
				Guide.xlabel("epochs per worker"),
				Guide.ylabel("test error in %"),
				Guide.title("$(dataset) $(specs[:model]), $(method)"),
				colorkey,
				Coord.Cartesian(ymin = ymin, ymax = ymax))

draw(PDF(string("$(figures_path)$(method)_$(specs[:model])_$(method)_$(string(varyparam))_$(run_suffix)_avgparams.pdf"), 12.5cm, 10cm), plot_avgparams)
draw(PDF(string("$(figures_path)$(method)_$(specs[:model])_$(method)_$(string(varyparam))_$(run_suffix)_avgparams_zoomed.pdf"), 12.5cm, 10cm), plot_avgparams_zoomed)
println("$((time() - start_time)/60) minutes\n")