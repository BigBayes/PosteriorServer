# User experiment setting variables
method = "easgd"	# sgd distbayes sgld
dataset = "mnist" # mnist omniglot cifar-100
run_suffix = "easgd8"
(modelfactory, model_name) = make_dense_nn([500,300],10)
#(modelfactory, model_name) = make_dense_nn([100],10)
include("$(scripts_path)mnist_defaults.jl")
specs[:model] = "500x300"
specs[:nworkers] = 8
specs[:initparams] = :Xavier
specs[:sampler] = :Adam
specs[:batchsize] = 100

specs[:nitersinitial] = 200
specs[:stepsizeinitial] = 1e-3
specs[:injectnoiseinitial] = 0.
specs[:averagegradinitial] = true

specs[:niters] = 600 * 20
specs[:stepsize] = 5e-3
specs[:niterspersync] = 1000
specs[:injectnoise] = 0.
specs[:averagegrad] = true

specs[:moving_rate] = 0.25
specs[:use_aws] = false

# what parameters to vary for this experiment

varyparam = :niterspersync
varyvalues = repmat([10], 10)
