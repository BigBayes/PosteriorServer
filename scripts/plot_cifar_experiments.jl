push!(LOAD_PATH,"../src")
include("../src/read_distbayes_output.jl")
include("../src/read_downpour_output.jl")
include("../src/read_easgd_output.jl")

using Gadfly
using Colors


n = 3

results = cell(n)
layers_init = cell(n)
layers_avgnatpost = cell(n)

layers_natpost = cell(n)
layers_avgparams = cell(n)

n_beta = 3

results_beta = cell(n)
layers_init_beta = cell(n)
layers_beta_natpost = cell(n)

n_downpour = 3

results_downpour = cell(n_downpour)
layers_downpour = cell(n_downpour)

n_easgd = 3

results_easgd = cell(n_easgd)
layers_easgd = cell(n_easgd)

n_adam = 3
results_adam = cell(n_adam)
layers_adam = cell(n_adam)

path = "../results/"
cd(path)

results[1] = read_distbayes_output("$(path)cifar_10_gcn_zca_alex_postserver_base_decreasing_1_1_nworkers=8.jld")
results[2] = read_distbayes_output("$(path)cifar_10_gcn_zca_alex_postserver_base_decreasing_2_1_nworkers=8.jld")
results[3] = read_distbayes_output("$(path)cifar_10_gcn_zca_alex_postserver_base_decreasing_3_1_nworkers=8.jld")

results_beta[1] = read_distbayes_output("$(path)cifar_10_gcn_zca_alex_postserver_beta_nworkers=8_1_1_beta=0.125.jld")
results_beta[2] = read_distbayes_output("$(path)cifar_10_gcn_zca_alex_postserver_beta_nworkers=8_2_1_beta=0.125.jld")
results_beta[3] = read_distbayes_output("$(path)cifar_10_gcn_zca_alex_postserver_beta_nworkers=8_3_1_beta=0.125.jld")

results_downpour[1] = read_downpour_output("$(path)cifar_10_gcn_zca_alex_downpour_base_1_1_averagegradworker=true.jld")
results_downpour[2] = read_downpour_output("$(path)cifar_10_gcn_zca_alex_downpour_base_2_1_averagegradworker=true.jld")
results_downpour[3] = read_downpour_output("$(path)cifar_10_gcn_zca_alex_downpour_base_3_1_averagegradworker=true.jld")

results_easgd[1] = read_elastic_output("$(path)cifar_10_gcn_zca_alex_easgd_movingrate_1_1_nworkers=8.jld")
results_easgd[2] = read_elastic_output("$(path)cifar_10_gcn_zca_alex_easgd_movingrate_2_1_nworkers=8.jld")
results_easgd[3] = read_elastic_output("$(path)cifar_10_gcn_zca_alex_easgd_movingrate_3_1_nworkers=8.jld")


results_adam[1] = read_downpour_output("$(path)cifar_10_gcn_zca_alex_downpour_adam_1_1_nworkers=1.jld")
results_adam[2] = read_downpour_output("$(path)cifar_10_gcn_zca_alex_downpour_adam_2_1_nworkers=1.jld")
results_adam[3] = read_downpour_output("$(path)cifar_10_gcn_zca_alex_downpour_adam_3_1_nworkers=1.jld")

colors_snep = repmat([colorant"red"],3)
colors_beta =  repmat([colorant"orange"],3)
colors_downpour = repmat([colorant"blue"],3)
colors_easgd = repmat([colorant"black"],3)
colors_adam = repmat([colorant"green"],3)

nworkers = [8]
for i in 1:n
    layers_avgnatpost[i] = layer(x=results[i][:worker_iters_dist]/500,y=100*(1-results[i][:acc_avgnatpost]),Theme(default_color = colors_snep[i]), Geom.line)
    layers_avgparams[i] = layer(x=results[i][:worker_iters_dist]/500,y=100*(1-results[i][:acc_avgparams]),Theme(default_color = colors_snep[i]), Geom.line)
    layers_natpost[i] = layer(x=results[i][:worker_iters_dist]/500,y=100*(1-results[i][:acc_natpost]),Theme(default_color = colors_snep[i]), Geom.line)
end

for i in 1:n_beta
    layers_beta_natpost[i] = layer(x=results_beta[i][:worker_iters_dist]/500,y=100*(1-results[i][:acc_natpost]),Theme(default_color = colors_beta[i]), Geom.line)
end

for i in 1:n_downpour
    layers_downpour[i] = layer(x=0.33+results_downpour[i][:worker_iters]/500,y=100*(1-results_downpour[i][:acc]),Theme(default_color = colors_downpour[i]), Geom.line)
end

for i in 1:n_easgd
    layers_easgd[i] = layer(x=0.33+results_easgd[i][:worker_iters]/500,y=100*(1-results_easgd[i][:acc]),Theme(default_color = colors_easgd[i]), Geom.line)
end

for j in 1:n_adam
        layers_adam[j] = layer(x=results_adam[j][:iters]/500,y=100*(1-results_adam[j][:acc]),Theme(default_color = colors_adam[j]), Geom.line)
end

Gadfly.set_default_plot_size(20cm,15cm)

colorkey = Guide.manual_color_key("Methods",["SNEP","p-SNEP","A-SGD","EASGD","Adam"],[colors_snep[1],colors_beta[1],colors_downpour[1],colors_easgd[1],colors_adam[1]])


average_snep = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/500 for i in 1:n],[100*(1-results[i][:acc_natpost]) for i in 1:n])

average_beta = MLUtilities.interpolatedAverage([results_beta[i][:worker_iters_dist]/500 for i in 1:n_beta],[100*(1-results_beta[i][:acc_natpost]) for i in 1:n_beta])

average_downpour = MLUtilities.interpolatedAverage([results_downpour[i][:worker_iters]/500 for i in 1:n_downpour],[100*(1-results_downpour[i][:acc]) for i in 1:n_downpour])

average_easgd = MLUtilities.interpolatedAverage([results_easgd[i][:worker_iters]/500 for i in 1:n_easgd],[100*(1-results_easgd[i][:acc]) for i in 1:n_easgd])


average_adam = MLUtilities.interpolatedAverage([results_adam[i][:iters]/500 for i in 1:n_adam],[100*(1-results_adam[i][:acc]) for i in 1:n_adam])


layer_avg_snep = layer(x=average_snep[1],y=average_snep[2],Geom.line,Theme(default_color = colors_snep[1]))
layer_avg_beta = layer(x=average_beta[1],y=average_beta[2],Geom.line,Theme(default_color = colors_beta[1]))
layer_avg_downpour = layer(x=average_downpour[1],y=average_downpour[2],Geom.line,Theme(default_color = colors_downpour[1]))
layer_avg_easgd = layer(x=average_easgd[1],y=average_easgd[2],Geom.line,Theme(default_color = colors_easgd[1]))
layer_avg_adam = layer(x=average_adam[1],y=average_adam[2],Geom.line,Theme(default_color = colors_adam[1]))


p = plot(layer_avg_snep,layer_avg_beta,layer_avg_downpour,layer_avg_easgd,layer_avg_adam,Guide.xlabel("Epochs per worker"),Guide.ylabel("Test error in %"),Guide.title("Comparison of distributed methods (8 workers)"),colorkey,Coord.Cartesian(ymin=20, ymax=40))


draw(PDF("../results/cifar_experiments.pdf",12.5cm,10cm),p)
