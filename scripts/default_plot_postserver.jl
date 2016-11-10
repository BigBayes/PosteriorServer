datasize = size(images, 4)

println("Loading libraries...")
start_time = time()
include("$(source_path)read_distbayes_output.jl")
using MLUtilities
println("$((time() - start_time)/60) minutes\n")

println("Loading Gadfly and Colors")
start_time = time()
using Gadfly
using Colors
println("$((time() - start_time)/60) minutes\n")

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
	dbo = read_distbayes_output("$(load_result_path)$(file_name).jld")

	push!(acc[vv], dbo[:acc_avgparams])
	push!(times[vv], dbo[:times_dist])
	push!(iters[vv], dbo[:worker_iters_dist] * specs[:batchsize] / datasize)
end

println("Finished reading data...")
println("$((time() - start_time)/60) minutes\n")

println("Started plotting...")
start_time = time()

colors = Colors.linspace(colorant"red", colorant"blue", length(unique_varyvalues))

# Create an array of layers
layers_avgparams = cell(0)

for ii = 1:length(unique_varyvalues)
	vv = unique_varyvalues[ii]

	this_times = times[vv]
	this_acc = acc[vv]
	this_iters = iters[vv]

	for kk = 1:length(this_times)
		push!(layers_avgparams, layer(x = this_iters[kk], y = (1.0-this_acc[kk])*100, Geom.line, Theme(default_color = colors[ii], line_width=0.1pt)))
	end

	# Plot the interpolated runs
  	(xvals, yvals) = MLUtilities.interpolatedAverage(this_iters, this_acc)
  	push!(layers_avgparams, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colors[ii], line_width=1pt)))
end

Gadfly.set_default_plot_size(20cm,15cm)
colorkey = Guide.manual_color_key(string(varyparam), [string(x) for x in unique_varyvalues], colors)

if run_suffix != ""
	title_string = string(dataset, " ", specs[:model], ", ", run_suffix)
else
	title_string = string(dataset, " ", specs[:model])
end

plot_avgparams = plot(
				layers_avgparams...,
				Theme(background_color = colorant"white"),
				Guide.xlabel("epochs per worker"),
				Guide.ylabel("test error in %"),
				Guide.title(title_string),
				colorkey,
				Coord.Cartesian(ymin = 0., ymax = 100.))

plot_avgparams_zoomed = plot(
				layers_avgparams...,
				Theme(background_color = colorant"white"),
				Guide.xlabel("epochs per worker"),
				Guide.ylabel("test error in %"),
				Guide.title(title_string),
				colorkey,
				Coord.Cartesian(ymin = 1., ymax = 5.))

draw(PNG(string(figures_path, file_suffix, "_avgparams.png"), 20cm, 15cm), plot_avgparams)
draw(PNG(string(figures_path, file_suffix, "_avgparams_zoomed.png"), 20cm, 15cm), plot_avgparams_zoomed)
println("$((time() - start_time)/60) minutes\n")