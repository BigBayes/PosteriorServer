push!(LOAD_PATH,"../src")
include("../src/read_distbayes_output.jl")
include("../src/read_downpour_output.jl")
using Gadfly
using Colors


n = 40

results = cell(n)
results_adam = cell(10)

layers_init = cell(n)
layers_avgnatpost = cell(n)
layers_natpost = cell(n)
layers_avgparams = cell(n)

layers_adam = cell(10)

path = realpath("../results/")

results[1] = read_distbayes_output("$(path)mnist_500x300_postserver_base_1_1_nworkers=2.jld")
results[2] = read_distbayes_output("$(path)mnist_500x300_postserver_base_2_1_nworkers=2.jld")
results[3] = read_distbayes_output("$(path)mnist_500x300_postserver_base_3_1_nworkers=2.jld")
results[4] = read_distbayes_output("$(path)mnist_500x300_postserver_base_4_1_nworkers=2.jld")
results[5] = read_distbayes_output("$(path)mnist_500x300_postserver_base_5_1_nworkers=2.jld")

results[6] = read_distbayes_output("$(path)mnist_500x300_postserver_base_6_1_nworkers=2.jld")
results[7] = read_distbayes_output("$(path)mnist_500x300_postserver_base_7_1_nworkers=2.jld")
results[8] = read_distbayes_output("$(path)mnist_500x300_postserver_base_8_1_nworkers=2.jld")
results[9] = read_distbayes_output("$(path)mnist_500x300_postserver_base_9_1_nworkers=2.jld")
results[10] = read_distbayes_output("$(path)mnist_500x300_postserver_base_10_1_nworkers=2.jld")

results[11] = read_distbayes_output("$(path)mnist_500x300_postserver_base_11_1_nworkers=4.jld")
results[12] = read_distbayes_output("$(path)mnist_500x300_postserver_base_12_1_nworkers=4.jld")
results[13] = read_distbayes_output("$(path)mnist_500x300_postserver_base_13_1_nworkers=4.jld")
results[14] = read_distbayes_output("$(path)mnist_500x300_postserver_base_14_1_nworkers=4.jld")
results[15] = read_distbayes_output("$(path)mnist_500x300_postserver_base_15_1_nworkers=4.jld")

results[16] = read_distbayes_output("$(path)mnist_500x300_postserver_base_16_1_nworkers=4.jld")
results[17] = read_distbayes_output("$(path)mnist_500x300_postserver_base_17_1_nworkers=4.jld")
results[18] = read_distbayes_output("$(path)mnist_500x300_postserver_base_18_1_nworkers=4.jld")
results[19] = read_distbayes_output("$(path)mnist_500x300_postserver_base_19_1_nworkers=4.jld")
results[20] = read_distbayes_output("$(path)mnist_500x300_postserver_base_20_1_nworkers=4.jld")

results[21] = read_distbayes_output("$(path)mnist_500x300_postserver_base_21_1_nworkers=8.jld")
results[22] = read_distbayes_output("$(path)mnist_500x300_postserver_base_22_1_nworkers=8.jld")
results[23] = read_distbayes_output("$(path)mnist_500x300_postserver_base_23_1_nworkers=8.jld")
results[24] = read_distbayes_output("$(path)mnist_500x300_postserver_base_24_1_nworkers=8.jld")
results[25] = read_distbayes_output("$(path)mnist_500x300_postserver_base_25_1_nworkers=8.jld")

results[26] = read_distbayes_output("$(path)mnist_500x300_postserver_base_26_1_nworkers=8.jld")
results[27] = read_distbayes_output("$(path)mnist_500x300_postserver_base_27_1_nworkers=8.jld")
results[28] = read_distbayes_output("$(path)mnist_500x300_postserver_base_28_1_nworkers=8.jld")
results[29] = read_distbayes_output("$(path)mnist_500x300_postserver_base_29_1_nworkers=8.jld")
results[30] = read_distbayes_output("$(path)mnist_500x300_postserver_base_30_1_nworkers=8.jld")


results[31] = read_distbayes_output("$(path)mnist_500x300_postserver_base_1_1_nworkers=1.jld")
results[32] = read_distbayes_output("$(path)mnist_500x300_postserver_base_2_1_nworkers=1.jld")
results[33] = read_distbayes_output("$(path)mnist_500x300_postserver_base_3_1_nworkers=1.jld")
results[34] = read_distbayes_output("$(path)mnist_500x300_postserver_base_4_1_nworkers=1.jld")
results[35] = read_distbayes_output("$(path)mnist_500x300_postserver_base_5_1_nworkers=1.jld")

results[36] = read_distbayes_output("$(path)mnist_500x300_postserver_base_6_1_nworkers=1.jld")
results[37] = read_distbayes_output("$(path)mnist_500x300_postserver_base_7_1_nworkers=1.jld")
results[38] = read_distbayes_output("$(path)mnist_500x300_postserver_base_8_1_nworkers=1.jld")
results[39] = read_distbayes_output("$(path)mnist_500x300_postserver_base_9_1_nworkers=1.jld")
results[40] = read_distbayes_output("$(path)mnist_500x300_postserver_base_10_1_nworkers=1.jld")

for i in 1:10
    results_adam[i] = read_downpour_output("mnist_500x300_downpour_adam_$(i)_1_nworkers=1.jld")
end


colors_snep = [repmat([colorant"red"],10),repmat([colorant"purple"],10),repmat([colorant"blue"],10),repmat([colorant"black"],10)]

nworkers= [2, 4, 8]



Gadfly.set_default_plot_size(20cm,15cm)
colorkey = Guide.manual_color_key("",["1","2","4","8","Adam"],[colors_snep[31],colors_snep[1],colors_snep[11],colors_snep[21],colorant"green"])
p_natpost = plot(layers_natpost[[1:15;17:40]]...,layers_adam..., Guide.xlabel("Epochs per worker"),Guide.ylabel("Test error in %"),
Guide.title("Varying N<sub>Workers</sub>"),colorkey,Coord.Cartesian(xmin = 0.0, xmax = 20.33, ymin=1, ymax=2.5))


average_2 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 1:10],[100*(1-results[i][:acc_natpost]) for i in 1:10])
average_4 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in [11:15;17:20]],[100*(1-results[i][:acc_natpost]) for i in [11:15;17:20]])
average_8 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 21:30],[100*(1-results[i][:acc_natpost]) for i in 21:30])

average_1 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 31:40],[100*(1-results[i][:acc_natpost]) for i in 31:40])

average_adam =  MLUtilities.interpolatedAverage([results_adam[i][:iters]/600 for i in 1:10],[100*(1-results_adam[i][:acc]) for i in 1:10])

layer_avg_2 = layer(x=average_2[1],y=average_2[2],Geom.line,Theme(default_color = colors_snep[1]))
layer_avg_4 = layer(x=average_4[1],y=average_4[2],Geom.line,Theme(default_color = colors_snep[11]))
layer_avg_8 = layer(x=average_8[1],y=average_8[2],Geom.line,Theme(default_color = colors_snep[21]))

layer_avg_1 = layer(x=average_1[1],y=average_1[2],Geom.line,Theme(default_color = colors_snep[31]))

layer_avg_adam = layer(x=average_adam[1],y=average_adam[2],Geom.line,Theme(default_color = colorant"green"))

p = plot(layer_avg_1,layer_avg_2,layer_avg_4,layer_avg_8,layer_avg_adam,Guide.xlabel("epochs per worker"),Guide.ylabel("test error in %"),
Guide.title("Varying the number of workers"),colorkey,Coord.Cartesian(xmin = 0.0, xmax = 20.33, ymin=1, ymax=2.5))


draw(PDF("../results/mnist_base_adam.pdf",12.5cm,10cm),p)
