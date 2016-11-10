using JLD

data_name = ["mnist_deep_postserver_nworkers", 
            "mnist_deep_downpour_nworkers",
            "mnist_deep_downpour_nworkers_prelearn",
            "mnist_deep_easgd_nworkers_noise",
            "mnist_deep_easgd_nworkers_noise_prelearn"]

# Includes
# NOTE: Notice relative path below, e.g. works when in \results folder
include("../scripts/paths.jl")
push!(LOAD_PATH, "$(source_path)")
include("$(source_path)models.jl")

# Read the experiment data
file = jldopen("$(save_result_path)$(data_name[1])_all.jld", "r")
snep_acc = read(file, "acc")
snep_times = read(file, "times")
snep_iters = read(file, "iters")
close(file);

file = jldopen("$(save_result_path)$(data_name[2])_all.jld", "r")
asgd_acc = read(file, "acc")
asgd_times = read(file, "times")
asgd_iters = read(file, "iters")
close(file);

file = jldopen("$(save_result_path)$(data_name[3])_all.jld", "r")
asgd_pl_acc = read(file, "acc")
asgd_pl_times = read(file, "times")
asgd_pl_iters = read(file, "iters")
close(file);

file = jldopen("$(save_result_path)$(data_name[4])_all.jld", "r")
easgd_acc = read(file, "acc")
easgd_times = read(file, "times")
easgd_iters = read(file, "iters")
close(file);

file = jldopen("$(save_result_path)$(data_name[5])_all.jld", "r")
easgd_pl_acc = read(file, "acc")
easgd_pl_times = read(file, "times")
easgd_pl_iters = read(file, "iters")
close(file);


# Most of specs
datasize = 60000
ymin = 1.
ymax = 5.

println("Loading libraries...")
start_time = time()
using MLUtilities
println("$((time() - start_time)/60) minutes\n")

println("Loading Gadfly and Colors")
start_time = time()
using Gadfly
using Compose
using Colors
println("$((time() - start_time)/60) minutes\n")

# Load in data files
#unique_varyvalues =  convert(Array{Float64, 1}, unique(varyvalues))

println("Started plotting...")
start_time = time()

#colors = Colors.linspace(colorant"red", colorant"blue", length(unique_varyvalues))
#colors = Colors.linspace(colorant"black", colorant"black", length(unique_varyvalues))

# Create layers for graphs of all runs of 8 workers
layers_snep_8workers = cell(0)
layers_asgd_8workers = cell(0)
layers_easgd_8workers = cell(0)

vv = 8
this_times = snep_times[vv]
this_acc = snep_acc[vv]
this_iters = snep_iters[vv]
(xvals, yvals) = MLUtilities.interpolatedAverage(this_iters, this_acc)
push!(layers_snep_8workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"black", line_width=1pt)))
for kk = 1:length(this_times)
	push!(layers_snep_8workers, layer(x = this_iters[kk], y = (1.0-this_acc[kk])*100, Geom.line, Theme(default_color = colorant"grey", line_width=0.1pt)))
end

this_times = asgd_pl_times[vv]
this_acc = asgd_pl_acc[vv]
this_iters = asgd_pl_iters[vv]
(xvals, yvals) = MLUtilities.interpolatedAverage(this_iters, this_acc)
push!(layers_asgd_8workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"black", line_width=1pt)))
for kk = 1:length(this_times)
    push!(layers_asgd_8workers, layer(x = this_iters[kk], y = (1.0-this_acc[kk])*100, Geom.line, Theme(default_color = colorant"grey", line_width=0.1pt)))
end

this_times = easgd_pl_times[vv]
this_acc = easgd_pl_acc[vv]
this_iters = easgd_pl_iters[vv]
(xvals, yvals) = MLUtilities.interpolatedAverage(this_iters, this_acc)
push!(layers_easgd_8workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"black", line_width=1pt)))
for kk = 1:length(this_times)
    push!(layers_easgd_8workers, layer(x = this_iters[kk], y = (1.0-this_acc[kk])*100, Geom.line, Theme(default_color = colorant"grey", line_width=0.1pt)))
end

# Create layers for graphs for separate numbers of workers
layers_2workers = cell(0)
layers_4workers = cell(0)
layers_8workers = cell(0)
layers_16workers = cell(0)

# 2 workers
(xvals, yvals) = MLUtilities.interpolatedAverage(snep_iters[2], snep_acc[2])
push!(layers_2workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"red", line_width=1pt)))

(xvals, yvals) = MLUtilities.interpolatedAverage(asgd_iters[2], asgd_acc[2])
push!(layers_2workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"blue", line_width=1pt, line_style = [1 * Compose.mm, 1 * Compose.mm])))
(xvals, yvals) = MLUtilities.interpolatedAverage(asgd_pl_iters[2], asgd_pl_acc[2])
push!(layers_2workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"blue", line_width=1pt)))

(xvals, yvals) = MLUtilities.interpolatedAverage(easgd_iters[2], easgd_acc[2])
push!(layers_2workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"green", line_width=1pt, line_style = Gadfly.get_stroke_vector(:dot))))
(xvals, yvals) = MLUtilities.interpolatedAverage(easgd_pl_iters[2], easgd_pl_acc[2])
push!(layers_2workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"green", line_width=1pt)))

# 4 workers
(xvals, yvals) = MLUtilities.interpolatedAverage(snep_iters[4], snep_acc[4])
push!(layers_4workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"red", line_width=1pt)))

(xvals, yvals) = MLUtilities.interpolatedAverage(asgd_iters[4], asgd_acc[4])
push!(layers_4workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"blue", line_width=1pt, line_style = Gadfly.get_stroke_vector(:dot))))
(xvals, yvals) = MLUtilities.interpolatedAverage(asgd_pl_iters[4], asgd_pl_acc[4])
push!(layers_4workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"blue", line_width=1pt)))

(xvals, yvals) = MLUtilities.interpolatedAverage(easgd_iters[4], easgd_acc[4])
push!(layers_4workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"green", line_width=1pt, line_style = Gadfly.get_stroke_vector(:dot))))
(xvals, yvals) = MLUtilities.interpolatedAverage(easgd_pl_iters[4], easgd_pl_acc[4])
push!(layers_4workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"green", line_width=1pt)))

# 8 workers
(xvals, yvals) = MLUtilities.interpolatedAverage(snep_iters[8], snep_acc[8])
push!(layers_8workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"red", line_width=1pt)))

(xvals, yvals) = MLUtilities.interpolatedAverage(asgd_iters[8], asgd_acc[8])
push!(layers_8workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"blue", line_width=1pt, line_style = Gadfly.get_stroke_vector(:dot))))
(xvals, yvals) = MLUtilities.interpolatedAverage(asgd_pl_iters[8], asgd_pl_acc[8])
push!(layers_8workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"blue", line_width=1pt)))

(xvals, yvals) = MLUtilities.interpolatedAverage(easgd_iters[8], easgd_acc[8])
push!(layers_8workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"green", line_width=1pt, line_style = Gadfly.get_stroke_vector(:dot))))
(xvals, yvals) = MLUtilities.interpolatedAverage(easgd_pl_iters[8], easgd_pl_acc[8])
push!(layers_8workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"green", line_width=1pt)))

# 16 workers
(xvals, yvals) = MLUtilities.interpolatedAverage(snep_iters[16], snep_acc[16])
push!(layers_16workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"red", line_width=1pt)))

(xvals, yvals) = MLUtilities.interpolatedAverage(asgd_iters[16], asgd_acc[16])
push!(layers_16workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"blue", line_width=1pt, line_style = Gadfly.get_stroke_vector(:dot))))
(xvals, yvals) = MLUtilities.interpolatedAverage(asgd_pl_iters[16], asgd_pl_acc[16])
push!(layers_16workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"blue", line_width=1pt)))

(xvals, yvals) = MLUtilities.interpolatedAverage(easgd_iters[16], easgd_acc[16])
push!(layers_16workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"green", line_width=1pt, line_style = Gadfly.get_stroke_vector(:dot))))
(xvals, yvals) = MLUtilities.interpolatedAverage(easgd_pl_iters[16], easgd_pl_acc[16])
push!(layers_16workers, layer(x = xvals, y = (1.0 - yvals)*100, Geom.line, Theme(default_color = colorant"green", line_width=1pt)))

Gadfly.set_default_plot_size(20cm,15cm)
colorkey = Guide.manual_color_key("", ["ASGD", "EASGD", "p-SNEP"], [colorant"blue", colorant"green", colorant"red"])

# Plot 8 worker curves
plot_snep_8workers = plot(
				layers_snep_8workers...,
				Theme(background_color = colorant"white"),
				Guide.xlabel("epochs per worker"),
				Guide.ylabel("test error in %"),
				Guide.title("p-SNEP"),
				Coord.Cartesian(ymin = 1., ymax = 5.))

plot_asgd_8workers = plot(
                layers_asgd_8workers...,
                Theme(background_color = colorant"white"),
                Guide.xlabel("epochs per worker"),
                Guide.ylabel("test error in %"),
                Guide.title("ASGD"),
                Coord.Cartesian(ymin = 1., ymax = 5.))

plot_easgd_8workers = plot(
                layers_easgd_8workers...,
                Theme(background_color = colorant"white"),
                Guide.xlabel("epochs per worker"),
                Guide.ylabel("test error in %"),
                Guide.title("EASGD"),
                Coord.Cartesian(ymin = 1., ymax = 5.))

# Plot 2 workers
plot_2workers = plot(
                layers_2workers...,
                Theme(background_color = colorant"white"),
                Guide.xlabel("epochs per worker"),
                Guide.ylabel("test error in %"),
                Guide.title("2 workers"),
                colorkey,
                Coord.Cartesian(ymin = 1., ymax = 5.))

plot_4workers = plot(
                layers_4workers...,
                Theme(background_color = colorant"white"),
                Guide.xlabel("epochs per worker"),
                Guide.ylabel("test error in %"),
                Guide.title("4 workers"),
                colorkey,
                Coord.Cartesian(ymin = 1., ymax = 5.))

plot_8workers = plot(
                layers_8workers...,
                Theme(background_color = colorant"white"),
                Guide.xlabel("epochs per worker"),
                Guide.ylabel("test error in %"),
                Guide.title("8 workers"),
                colorkey,
                Coord.Cartesian(ymin = 1., ymax = 5.))

plot_16workers = plot(
                layers_16workers...,
                Theme(background_color = colorant"white"),
                Guide.xlabel("epochs per worker"),
                Guide.ylabel("test error in %"),
                Guide.title("16 workers"),
                colorkey,
                Coord.Cartesian(ymin = 1., ymax = 5.))


draw(PDF(string("$(figures_path)2workers.pdf"), 12.5cm, 10cm), plot_2workers)
draw(PDF(string("$(figures_path)4workers.pdf"), 12.5cm, 10cm), plot_4workers)
draw(PDF(string("$(figures_path)8workers.pdf"), 12.5cm, 10cm), plot_8workers)
draw(PDF(string("$(figures_path)16workers.pdf"), 12.5cm, 10cm), plot_16workers)

draw(PDF(string("$(figures_path)snep_8workers.pdf"), 8.33cm, 6.67cm), plot_snep_8workers)
draw(PDF(string("$(figures_path)asgd_8workers.pdf"), 8.33cm, 6.67cm), plot_asgd_8workers)
draw(PDF(string("$(figures_path)easgd_8workers.pdf"), 8.33cm, 6.67cm), plot_easgd_8workers)

println("$((time() - start_time)/60) minutes\n")