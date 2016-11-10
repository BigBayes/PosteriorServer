# User experiment setting variables
method = "downpour"	# sgd distbayes sgld
dataset = "mnist" # mnist omniglot cifar-100
run_suffix = "adam"
(modelfactory, model_name) = make_dense_nn([500,300],10)
include("$(scripts_path)mnist_defaults.jl")

specs[:model] = "mnist"
specs[:nworkers] = 1
specs[:initparams] = :Xavier
specs[:sampler] = :Adam
specs[:sampler_workers] = :Adam
specs[:batchsize] = 100

specs[:nitersinitial] = 600 * 20
specs[:stepsizeinitial] = 0.001
specs[:injectnoiseinitial] = 1.0
specs[:averagegradinitial] = true

specs[:niters] = 0
specs[:stepsize] = 5e-3
specs[:masterstepsize] = 0.00075
specs[:injectnoise] = 0.
specs[:averagegrad] = true

specs[:injectnoiseworker] = 0.
specs[:averagegradworker] = false

varyparam = :nworkers
varyvalues = repmat([1], 10)[:]
