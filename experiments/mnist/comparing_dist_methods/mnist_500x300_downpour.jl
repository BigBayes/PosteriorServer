# User experiment setting variables
method = "downpour"	# sgd distbayes sgld
dataset = "mnist" # mnist omniglot cifar-100
run_suffix = "downpour8"
(modelfactory, model_name) = make_dense_nn([500,300],10)
include("$(scripts_path)mnist_defaults.jl")

specs[:model] = "500x300"
specs[:nworkers] = 8
specs[:initparams] = :Xavier
specs[:sampler] = :Adam
specs[:sampler_workers] = :Adam
specs[:batchsize] = 100

specs[:nitersinitial] = 200
specs[:stepsizeinitial] = 1e-3
specs[:injectnoiseinitial] = 0.
specs[:averagegradinitial] = true

specs[:niters] = 600*20
specs[:stepsize] = 5e-3
specs[:masterstepsize] = 0.00075
specs[:injectnoise] = 0.
specs[:averagegrad] = true

specs[:injectnoiseworker] = 0.
specs[:averagegradworker] = false

specs[:use_aws] = false
varyparam = :niterspersync
#varyvalues = repmat([1 40 200 600 1200], 5)
varyvalues = repmat([10], 10)
