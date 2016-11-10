push!(LOAD_PATH,"../src")
include("../src/read_distbayes_output.jl")

using Gadfly
using Colors


n = 80

results = cell(n)
layers_init = cell(n)
layers_avgnatpost = cell(n)

layers_natpost = cell(n)
layers_avgparams = cell(n)

path = realpath("../results/")
cd(path)

results[1] = read_distbayes_output("$(path)mnist_500x300_postserver_1_1_niterspersync=1.jld")
results[2] = read_distbayes_output("$(path)mnist_500x300_postserver_2_1_niterspersync=1.jld")
results[3] = read_distbayes_output("$(path)mnist_500x300_postserver_3_1_niterspersync=1.jld")
results[4] = read_distbayes_output("$(path)mnist_500x300_postserver_4_1_niterspersync=1.jld")
results[5] = read_distbayes_output("$(path)mnist_500x300_postserver_5_1_niterspersync=1.jld")
results[6] = read_distbayes_output("$(path)mnist_500x300_postserver_6_1_niterspersync=1.jld")
results[7] = read_distbayes_output("$(path)mnist_500x300_postserver_7_1_niterspersync=1.jld")
results[8] = read_distbayes_output("$(path)mnist_500x300_postserver_8_1_niterspersync=1.jld")
results[9] = read_distbayes_output("$(path)mnist_500x300_postserver_9_1_niterspersync=1.jld")
results[10] = read_distbayes_output("$(path)mnist_500x300_postserver_10_1_niterspersync=1.jld")

results[11] = read_distbayes_output("$(path)mnist_500x300_postserver_11_1_niterspersync=5.jld")
results[12] = read_distbayes_output("$(path)mnist_500x300_postserver_12_1_niterspersync=5.jld")
results[13] = read_distbayes_output("$(path)mnist_500x300_postserver_13_1_niterspersync=5.jld")
results[14] = read_distbayes_output("$(path)mnist_500x300_postserver_14_1_niterspersync=5.jld")
results[15] = read_distbayes_output("$(path)mnist_500x300_postserver_15_1_niterspersync=5.jld")
results[16] = read_distbayes_output("$(path)mnist_500x300_postserver_16_1_niterspersync=5.jld")
results[17] = read_distbayes_output("$(path)mnist_500x300_postserver_17_1_niterspersync=5.jld")
results[18] = read_distbayes_output("$(path)mnist_500x300_postserver_18_1_niterspersync=5.jld")
results[19] = read_distbayes_output("$(path)mnist_500x300_postserver_19_1_niterspersync=5.jld")
results[20] = read_distbayes_output("$(path)mnist_500x300_postserver_20_1_niterspersync=5.jld")

results[21] = read_distbayes_output("$(path)mnist_500x300_postserver_21_1_niterspersync=10.jld")
results[22] = read_distbayes_output("$(path)mnist_500x300_postserver_22_1_niterspersync=10.jld")
results[23] = read_distbayes_output("$(path)mnist_500x300_postserver_23_1_niterspersync=10.jld")
results[24] = read_distbayes_output("$(path)mnist_500x300_postserver_24_1_niterspersync=10.jld")
results[25] = read_distbayes_output("$(path)mnist_500x300_postserver_25_1_niterspersync=10.jld")
results[26] = read_distbayes_output("$(path)mnist_500x300_postserver_26_1_niterspersync=10.jld")
results[27] = read_distbayes_output("$(path)mnist_500x300_postserver_27_1_niterspersync=10.jld")
results[28] = read_distbayes_output("$(path)mnist_500x300_postserver_28_1_niterspersync=10.jld")
results[29] = read_distbayes_output("$(path)mnist_500x300_postserver_29_1_niterspersync=10.jld")
results[30] = read_distbayes_output("$(path)mnist_500x300_postserver_30_1_niterspersync=10.jld")

results[31] = read_distbayes_output("$(path)mnist_500x300_postserver_31_1_niterspersync=20.jld")
results[32] = read_distbayes_output("$(path)mnist_500x300_postserver_32_1_niterspersync=20.jld")
results[33] = read_distbayes_output("$(path)mnist_500x300_postserver_33_1_niterspersync=20.jld")
results[34] = read_distbayes_output("$(path)mnist_500x300_postserver_34_1_niterspersync=20.jld")
results[35] = read_distbayes_output("$(path)mnist_500x300_postserver_35_1_niterspersync=20.jld")
results[36] = read_distbayes_output("$(path)mnist_500x300_postserver_36_1_niterspersync=20.jld")
results[37] = read_distbayes_output("$(path)mnist_500x300_postserver_37_1_niterspersync=20.jld")
results[38] = read_distbayes_output("$(path)mnist_500x300_postserver_38_1_niterspersync=20.jld")
results[39] = read_distbayes_output("$(path)mnist_500x300_postserver_39_1_niterspersync=20.jld")
results[40] = read_distbayes_output("$(path)mnist_500x300_postserver_40_1_niterspersync=20.jld")


results[41] = read_distbayes_output("$(path)mnist_500x300_postserver_41_1_niterspersync=30.jld")
results[42] = read_distbayes_output("$(path)mnist_500x300_postserver_42_1_niterspersync=30.jld")
results[43] = read_distbayes_output("$(path)mnist_500x300_postserver_43_1_niterspersync=30.jld")
results[44] = read_distbayes_output("$(path)mnist_500x300_postserver_44_1_niterspersync=30.jld")
results[45] = read_distbayes_output("$(path)mnist_500x300_postserver_45_1_niterspersync=30.jld")
results[46] = read_distbayes_output("$(path)mnist_500x300_postserver_46_1_niterspersync=30.jld")
results[47] = read_distbayes_output("$(path)mnist_500x300_postserver_47_1_niterspersync=30.jld")
results[48] = read_distbayes_output("$(path)mnist_500x300_postserver_48_1_niterspersync=30.jld")
results[49] = read_distbayes_output("$(path)mnist_500x300_postserver_49_1_niterspersync=30.jld")
results[50] = read_distbayes_output("$(path)mnist_500x300_postserver_50_1_niterspersync=30.jld")

results[51] = read_distbayes_output("$(path)mnist_500x300_postserver_51_1_niterspersync=40.jld")
results[52] = read_distbayes_output("$(path)mnist_500x300_postserver_52_1_niterspersync=40.jld")
results[53] = read_distbayes_output("$(path)mnist_500x300_postserver_53_1_niterspersync=40.jld")
results[54] = read_distbayes_output("$(path)mnist_500x300_postserver_54_1_niterspersync=40.jld")
results[55] = read_distbayes_output("$(path)mnist_500x300_postserver_55_1_niterspersync=40.jld")
results[56] = read_distbayes_output("$(path)mnist_500x300_postserver_56_1_niterspersync=40.jld")
results[57] = read_distbayes_output("$(path)mnist_500x300_postserver_57_1_niterspersync=40.jld")
results[58] = read_distbayes_output("$(path)mnist_500x300_postserver_58_1_niterspersync=40.jld")
results[59] = read_distbayes_output("$(path)mnist_500x300_postserver_59_1_niterspersync=40.jld")
results[60] = read_distbayes_output("$(path)mnist_500x300_postserver_60_1_niterspersync=40.jld")

results[61] = read_distbayes_output("$(path)mnist_500x300_postserver_61_1_niterspersync=50.jld")
results[62] = read_distbayes_output("$(path)mnist_500x300_postserver_62_1_niterspersync=50.jld")
results[63] = read_distbayes_output("$(path)mnist_500x300_postserver_63_1_niterspersync=50.jld")
results[64] = read_distbayes_output("$(path)mnist_500x300_postserver_64_1_niterspersync=50.jld")
results[65] = read_distbayes_output("$(path)mnist_500x300_postserver_65_1_niterspersync=50.jld")
results[66] = read_distbayes_output("$(path)mnist_500x300_postserver_66_1_niterspersync=50.jld")
results[67] = read_distbayes_output("$(path)mnist_500x300_postserver_67_1_niterspersync=50.jld")
results[68] = read_distbayes_output("$(path)mnist_500x300_postserver_68_1_niterspersync=50.jld")
results[69] = read_distbayes_output("$(path)mnist_500x300_postserver_69_1_niterspersync=50.jld")
results[70] = read_distbayes_output("$(path)mnist_500x300_postserver_70_1_niterspersync=50.jld")


results[71] = read_distbayes_output("$(path)mnist_500x300_postserver_71_1_niterspersync=100.jld")
results[72] = read_distbayes_output("$(path)mnist_500x300_postserver_72_1_niterspersync=100.jld")
results[73] = read_distbayes_output("$(path)mnist_500x300_postserver_73_1_niterspersync=100.jld")
results[74] = read_distbayes_output("$(path)mnist_500x300_postserver_74_1_niterspersync=100.jld")
results[75] = read_distbayes_output("$(path)mnist_500x300_postserver_75_1_niterspersync=100.jld")
results[76] = read_distbayes_output("$(path)mnist_500x300_postserver_76_1_niterspersync=100.jld")
results[77] = read_distbayes_output("$(path)mnist_500x300_postserver_77_1_niterspersync=100.jld")
results[78] = read_distbayes_output("$(path)mnist_500x300_postserver_78_1_niterspersync=100.jld")
results[79] = read_distbayes_output("$(path)mnist_500x300_postserver_79_1_niterspersync=100.jld")
results[80] = read_distbayes_output("$(path)mnist_500x300_postserver_80_1_niterspersync=100.jld")

colors = Colors.linspace(colorant"red",colorant"blue",8)

niterspersync = [1,5,10,20,30,40,50,100]

Gadfly.set_default_plot_size(20cm,15cm)
colorkey = Guide.manual_color_key(["1","5","10","20","30","40","50","100"],colors)

average_1 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 1:10],[100*(1-results[i][:acc_natpost]) for i in 1:10])
average_5 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 11:20],[100*(1-results[i][:acc_natpost]) for i in 11:20])
average_10 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 21:30],[100*(1-results[i][:acc_natpost]) for i in 21:30])
average_20 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 31:40],[100*(1-results[i][:acc_natpost]) for i in 31:40])
average_30 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 41:50],[100*(1-results[i][:acc_natpost]) for i in 41:50])
average_40 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 51:60],[100*(1-results[i][:acc_natpost]) for i in 51:60])
average_50 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 61:70],[100*(1-results[i][:acc_natpost]) for i in 61:70])
average_100 = MLUtilities.interpolatedAverage([results[i][:worker_iters_dist]/600 for i in 71:80],[100*(1-results[i][:acc_natpost]) for i in 71:80])


layer_avg_1 = layer(x=average_1[1],y=average_1[2],Geom.line,Theme(default_color = colors[1]))
layer_avg_5 = layer(x=average_5[1],y=average_5[2],Geom.line,Theme(default_color = colors[2]))
layer_avg_10 = layer(x=average_10[1],y=average_10[2],Geom.line,Theme(default_color = colors[3]))
layer_avg_20 = layer(x=average_20[1],y=average_20[2],Geom.line,Theme(default_color = colors[4]))
layer_avg_30 = layer(x=average_30[1],y=average_30[2],Geom.line,Theme(default_color = colors[5]))
layer_avg_40 = layer(x=average_40[1],y=average_40[2],Geom.line,Theme(default_color = colors[6]))
layer_avg_50 = layer(x=average_50[1],y=average_50[2],Geom.line,Theme(default_color = colors[7]))
layer_avg_100 = layer(x=average_100[1],y=average_100[2],Geom.line,Theme(default_color = colors[8]))



p = plot(layer_avg_1,layer_avg_5,layer_avg_10,layer_avg_20,layer_avg_30,layer_avg_40,layer_avg_50,layer_avg_100,Guide.xlabel("epochs per worker"),Guide.ylabel("test error in %"),Guide.title("Varying the communication interval"),colorkey,Coord.Cartesian(ymin=1, ymax=2.5))

draw(PDF("../results/mnist_niterspersync.pdf",12.5cm,10cm),p)
