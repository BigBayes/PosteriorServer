push!(LOAD_PATH,"../src")
include("../src/read_distbayes_output.jl")
include("../src/read_downpour_output.jl")
include("../src/read_easgd_output.jl")
using Gadfly
using Colors


n = 10

results = cell(n)
results_beta = cell(n)
results_downpour = cell(n)
results_easgd = cell(n)
results_adam = cell(10)

layers_init = cell(n)
layers_avgnatpost = cell(n)
layers_natpost = cell(n)
layers_avgparams = cell(n)

layers_beta_init = cell(n)
layers_beta_avgnatpost = cell(n)
layers_beta_natpost = cell(n)
layers_beta_avgparams = cell(n)

layers_downpour = cell(n)
layers_easgd = cell(n)
layers_adam = cell(10)

path = realpath("../results/")
cd(path)

results[1] = read_distbayes_output("$(path)mnist_500x300_postserver_snep8_1_1_nworkers=8.jld")
results[2] = read_distbayes_output("$(path)mnist_500x300_postserver_snep8_2_1_nworkers=8.jld")
results[3] = read_distbayes_output("$(path)mnist_500x300_postserver_snep8_3_1_nworkers=8.jld")
results[4] = read_distbayes_output("$(path)mnist_500x300_postserver_snep8_4_1_nworkers=8.jld")
results[5] = read_distbayes_output("$(path)mnist_500x300_postserver_snep8_5_1_nworkers=8.jld")

results[6] = read_distbayes_output("$(path)mnist_500x300_postserver_snep8_6_1_nworkers=8.jld")
results[7] = read_distbayes_output("$(path)mnist_500x300_postserver_snep8_7_1_nworkers=8.jld")
results[8] = read_distbayes_output("$(path)mnist_500x300_postserver_snep8_8_1_nworkers=8.jld")
results[9] = read_distbayes_output("$(path)mnist_500x300_postserver_snep8_9_1_nworkers=8.jld")
results[10] = read_distbayes_output("$(path)mnist_500x300_postserver_snep8_10_1_nworkers=8.jld")


results_beta[1] = read_distbayes_output("$(path)mnist_500x300_postserver_nworkers=8_beta=nworkersinv_1_1_beta=0.125.jld")
results_beta[2] = read_distbayes_output("$(path)mnist_500x300_postserver_nworkers=8_beta=nworkersinv_2_1_beta=0.125.jld")
results_beta[3] = read_distbayes_output("$(path)mnist_500x300_postserver_nworkers=8_beta=nworkersinv_3_1_beta=0.125.jld")
results_beta[4] = read_distbayes_output("$(path)mnist_500x300_postserver_nworkers=8_beta=nworkersinv_4_1_beta=0.125.jld")
results_beta[5] = read_distbayes_output("$(path)mnist_500x300_postserver_nworkers=8_beta=nworkersinv_5_1_beta=0.125.jld")

results_beta[6] = read_distbayes_output("$(path)mnist_500x300_postserver_nworkers=8_beta=nworkersinv_6_1_beta=0.125.jld")
results_beta[7] = read_distbayes_output("$(path)mnist_500x300_postserver_nworkers=8_beta=nworkersinv_7_1_beta=0.125.jld")
results_beta[8] = read_distbayes_output("$(path)mnist_500x300_postserver_nworkers=8_beta=nworkersinv_8_1_beta=0.125.jld")
results_beta[9] = read_distbayes_output("$(path)mnist_500x300_postserver_nworkers=8_beta=nworkersinv_9_1_beta=0.125.jld")
results_beta[10] = read_distbayes_output("$(path)mnist_500x300_postserver_nworkers=8_beta=nworkersinv_10_1_beta=0.125.jld")


results_downpour[1] = read_downpour_output("$(path)mnist_500x300_downpour_downpour8_1_1_niterspersync=10.jld")
results_downpour[2] = read_downpour_output("$(path)mnist_500x300_downpour_downpour8_2_1_niterspersync=10.jld")
results_downpour[3] = read_downpour_output("$(path)mnist_500x300_downpour_downpour8_3_1_niterspersync=10.jld")
results_downpour[4] = read_downpour_output("$(path)mnist_500x300_downpour_downpour8_4_1_niterspersync=10.jld")
results_downpour[5] = read_downpour_output("$(path)mnist_500x300_downpour_downpour8_5_1_niterspersync=10.jld")

results_downpour[6] = read_downpour_output("$(path)mnist_500x300_downpour_downpour8_6_1_niterspersync=10.jld")
results_downpour[7] = read_downpour_output("$(path)mnist_500x300_downpour_downpour8_7_1_niterspersync=10.jld")
results_downpour[8] = read_downpour_output("$(path)mnist_500x300_downpour_downpour8_8_1_niterspersync=10.jld")
results_downpour[9] = read_downpour_output("$(path)mnist_500x300_downpour_downpour8_9_1_niterspersync=10.jld")
results_downpour[10] = read_downpour_output("$(path)mnist_500x300_downpour_downpour8_10_1_niterspersync=10.jld")


results_easgd[1] = read_elastic_output("$(path)mnist_500x300_easgd_easgd8_1_1_niterspersync=10.jld")
results_easgd[2] = read_elastic_output("$(path)mnist_500x300_easgd_easgd8_2_1_niterspersync=10.jld")
results_easgd[3] = read_elastic_output("$(path)mnist_500x300_easgd_easgd8_3_1_niterspersync=10.jld")
results_easgd[4] = read_elastic_output("$(path)mnist_500x300_easgd_easgd8_4_1_niterspersync=10.jld")
results_easgd[5] = read_elastic_output("$(path)mnist_500x300_easgd_easgd8_5_1_niterspersync=10.jld")

results_easgd[6] = read_elastic_output("$(path)mnist_500x300_easgd_easgd8_6_1_niterspersync=10.jld")
results_easgd[7] = read_elastic_output("$(path)mnist_500x300_easgd_easgd8_7_1_niterspersync=10.jld")
results_easgd[8] = read_elastic_output("$(path)mnist_500x300_easgd_easgd8_8_1_niterspersync=10.jld")
results_easgd[9] = read_elastic_output("$(path)mnist_500x300_easgd_easgd8_9_1_niterspersync=10.jld")
results_easgd[10] = read_elastic_output("$(path)mnist_500x300_easgd_easgd8_10_1_niterspersync=10.jld")


for i in 1:10
    results_adam[i] = read_downpour_output("mnist_500x300_downpour_adam_$(i)_1_nworkers=1.jld")
end

colors_beta = [repmat([colorant"red"],10)]
colors_snep = [repmat([colorant"grey"],10)]
colors_downpour = [repmat([colorant"blue"],10)]
colors_easgd = [repmat([colorant"green"],10)]


average_8 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 1:10],[100*(1-results[i][:acc_natpost]) for i in 1:10])

average_beta_8 = MLUtilities.interpolatedAverage([results_beta[i][:worker_iters_dist]/600 for i in 1:10],[100*(1-results_beta[i][:acc_natpost]) for i in 1:10])

average_downpour_8 = MLUtilities.interpolatedAverage([0.33+results_downpour[i][:worker_iters]/600 for i in 1:10],[100*(1-results_downpour[i][:acc]) for i in 1:10])

average_easgd_8 = MLUtilities.interpolatedAverage([0.33+results_easgd[i][:worker_iters]/600 for i in 1:10],[100*(1-results_easgd[i][:acc]) for i in 1:10])

average_adam =  MLUtilities.interpolatedAverage([results_adam[i][:iters]/600 for i in 1:10],[100*(1-results_adam[i][:accuracy]) for i in 1:10])


layer_avg_8 = layer(x=average_8[1],y=average_8[2],Geom.line,Theme(default_color = colors_snep[1]))

layer_avg_beta_8 = layer(x=average_beta_8[1],y=average_beta_8[2],Geom.line,Theme(default_color = colors_beta[1]))

layer_avg_dp_8 = layer(x=average_downpour_8[1],y=average_downpour_8[2],Geom.line,Theme(default_color = colors_downpour[1]))

layer_avg_ea_8 = layer(x=average_easgd_8[1],y=average_easgd_8[2],Geom.line,Theme(default_color = colors_easgd[1]))

layer_avg_adam = layer(x=average_adam[1],y=average_adam[2],Geom.line,Theme(default_color = colorant"pink"))

p = plot(layer_avg_8,layer_avg_beta_8,layer_avg_dp_8,layer_avg_ea_8,layer_avg_adam,Guide.xlabel("epochs per worker"),Guide.ylabel("test error in %"),
Guide.title("Comparison of distributed methods (8 workers)"),colorkey,Coord.Cartesian(xmin = 0.0, xmax = 20.33, ymin=1, ymax=2.5))

draw(PDF("../results/mnist_dist_comp_8workers.pdf",12.5cm,10cm),p)
