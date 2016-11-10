push!(LOAD_PATH,"../src")
include("../src/read_distbayes_output.jl")

using Gadfly
using Colors


n = 40

results = cell(n)
layers_init = cell(n)
layers_avgnatpost = cell(n)

layers_natpost = cell(n)
layers_avgparams = cell(n)

path = realpath("../results/")
cd(path)

results[1] = read_distbayes_output("$(path)mnist_500x300_postserver_1_1_nitersperdamp=1.jld")
results[2] = read_distbayes_output("$(path)mnist_500x300_postserver_2_1_nitersperdamp=1.jld")
results[3] = read_distbayes_output("$(path)mnist_500x300_postserver_3_1_nitersperdamp=1.jld")
results[4] = read_distbayes_output("$(path)mnist_500x300_postserver_4_1_nitersperdamp=1.jld")
results[5] = read_distbayes_output("$(path)mnist_500x300_postserver_5_1_nitersperdamp=1.jld")
results[6] = read_distbayes_output("$(path)mnist_500x300_postserver_6_1_nitersperdamp=1.jld")
results[7] = read_distbayes_output("$(path)mnist_500x300_postserver_7_1_nitersperdamp=1.jld")
results[8] = read_distbayes_output("$(path)mnist_500x300_postserver_8_1_nitersperdamp=1.jld")
results[9] = read_distbayes_output("$(path)mnist_500x300_postserver_9_1_nitersperdamp=1.jld")
results[10] = read_distbayes_output("$(path)mnist_500x300_postserver_10_1_nitersperdamp=1.jld")

results[11] = read_distbayes_output("$(path)mnist_500x300_postserver_11_1_nitersperdamp=5.jld")
results[12] = read_distbayes_output("$(path)mnist_500x300_postserver_12_1_nitersperdamp=5.jld")
results[13] = read_distbayes_output("$(path)mnist_500x300_postserver_13_1_nitersperdamp=5.jld")
results[14] = read_distbayes_output("$(path)mnist_500x300_postserver_14_1_nitersperdamp=5.jld")
results[15] = read_distbayes_output("$(path)mnist_500x300_postserver_15_1_nitersperdamp=5.jld")
results[16] = read_distbayes_output("$(path)mnist_500x300_postserver_16_1_nitersperdamp=5.jld")
results[17] = read_distbayes_output("$(path)mnist_500x300_postserver_17_1_nitersperdamp=5.jld")
results[18] = read_distbayes_output("$(path)mnist_500x300_postserver_18_1_nitersperdamp=5.jld")
results[19] = read_distbayes_output("$(path)mnist_500x300_postserver_19_1_nitersperdamp=5.jld")
results[20] = read_distbayes_output("$(path)mnist_500x300_postserver_20_1_nitersperdamp=5.jld")

results[21] = read_distbayes_output("$(path)mnist_500x300_postserver_21_1_nitersperdamp=10.jld")
results[22] = read_distbayes_output("$(path)mnist_500x300_postserver_22_1_nitersperdamp=10.jld")
results[23] = read_distbayes_output("$(path)mnist_500x300_postserver_23_1_nitersperdamp=10.jld")
results[24] = read_distbayes_output("$(path)mnist_500x300_postserver_24_1_nitersperdamp=10.jld")
results[25] = read_distbayes_output("$(path)mnist_500x300_postserver_25_1_nitersperdamp=10.jld")
results[26] = read_distbayes_output("$(path)mnist_500x300_postserver_26_1_nitersperdamp=10.jld")
results[27] = read_distbayes_output("$(path)mnist_500x300_postserver_27_1_nitersperdamp=10.jld")
results[28] = read_distbayes_output("$(path)mnist_500x300_postserver_28_1_nitersperdamp=10.jld")
results[29] = read_distbayes_output("$(path)mnist_500x300_postserver_29_1_nitersperdamp=10.jld")
results[30] = read_distbayes_output("$(path)mnist_500x300_postserver_30_1_nitersperdamp=10.jld")


results[31] = read_distbayes_output("$(path)mnist_500x300_postserver_31_1_nitersperdamp=20.jld")
results[32] = read_distbayes_output("$(path)mnist_500x300_postserver_32_1_nitersperdamp=20.jld")
results[33] = read_distbayes_output("$(path)mnist_500x300_postserver_33_1_nitersperdamp=20.jld")
results[34] = read_distbayes_output("$(path)mnist_500x300_postserver_34_1_nitersperdamp=20.jld")
results[35] = read_distbayes_output("$(path)mnist_500x300_postserver_35_1_nitersperdamp=20.jld")
results[36] = read_distbayes_output("$(path)mnist_500x300_postserver_36_1_nitersperdamp=20.jld")
results[37] = read_distbayes_output("$(path)mnist_500x300_postserver_37_1_nitersperdamp=20.jld")
results[38] = read_distbayes_output("$(path)mnist_500x300_postserver_38_1_nitersperdamp=20.jld")
results[39] = read_distbayes_output("$(path)mnist_500x300_postserver_39_1_nitersperdamp=20.jld")
results[40] = read_distbayes_output("$(path)mnist_500x300_postserver_40_1_nitersperdamp=20.jld")

colors = [repmat([colorant"red"],10), repmat([colorant"orange"],10), repmat([colorant"blue"],10),repmat([colorant"black"],10)]

nitersperdamp = [1,5,10,20]
Gadfly.set_default_plot_size(20cm,15cm)


average_1 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 1:10],[100*(1-results[i][:acc_natpost]) for i in 1:10])
average_2 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 11:20],[100*(1-results[i][:acc_natpost]) for i in 11:20])
average_4 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 21:30],[100*(1-results[i][:acc_natpost]) for i in 21:30])
average_8 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 31:40],[100*(1-results[i][:acc_natpost]) for i in 31:40])

layer_avg_1 = layer(x=average_1[1],y=average_1[2],Geom.line,Theme(default_color = colors[1]))
layer_avg_2 = layer(x=average_2[1],y=average_2[2],Geom.line,Theme(default_color = colors[11]))
layer_avg_4 = layer(x=average_4[1],y=average_4[2],Geom.line,Theme(default_color = colors[21]))
layer_avg_8 = layer(x=average_8[1],y=average_8[2],Geom.line,Theme(default_color = colors[31]))

p = plot(layer_avg_1,layer_avg_2,layer_avg_4,layer_avg_8,Guide.xlabel("epochs per worker"),Guide.ylabel("test error in %"),Guide.title("Varying the number of outer iterations"),Guide.manual_color_key("",["1","5","10","20"],[colors[1],colors[11],colors[21],colors[31]]),Coord.Cartesian(ymin=1, ymax=2.5))



draw(PDF("mnist_n_outer.pdf",12.5cm,10cm),p)
