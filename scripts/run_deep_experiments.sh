#!/bin/bash
# Reproduce deep model experiments and produce figures
julia run.jl deep/mnist_asgd_deep_nworkers
julia run.jl deep/mnist_asgd_deep_nworkers_prelearn
julia run.jl deep/mnist_deep_nworkers
julia run.jl deep/mnist_easgd_deep_nworkers_noise
julia run.jl deep/mnist_easgd_deep_nworkers_noise_prelearn
julia extract.jl deep/mnist_asgd_deep_nworkers
julia extract.jl deep/mnist_asgd_deep_nworkers_prelearn
julia extract.jl deep/mnist_deep_nworkers
julia extract.jl deep/mnist_easgd_deep_nworkers_noise
julia extract.jl deep/mnist_easgd_deep_nworkers_noise_prelearn
julia plot_deep_figures.jl