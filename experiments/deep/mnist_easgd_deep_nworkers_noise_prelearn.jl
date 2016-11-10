# User experiment setting variables
method = "easgd"	# sgd distbayes sgld
dataset = "mnist" # mnist omniglot cifar-100
run_suffix = "nworkers_noise_prelearn"
(modelfactory, model_name) = make_dense_nn([50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50],10)
include("$(scripts_path)mnist_defaults.jl")

specs[:model] = "deep"
specs[:nworkers] = 8
specs[:initparams] = :Xavier
specs[:sampler] = :Adam
specs[:batchsize] = 100

specs[:nitersinitial] = 200
specs[:stepsizeinitial] = 0.00065
specs[:injectnoiseinitial] = 0.
specs[:averagegradinitial] = true

specs[:niters] = 600 * 50
specs[:stepsize] = 0.00065
specs[:injectnoise] = 1.0
specs[:averagegrad] = true

specs[:moving_rate] = 0.25

varyparam = :nworkers
varyvalues = repmat([2 4 6 8 16], 10)[:]