# User experiment setting variables
method = "downpour"	# sgd distbayes sgld
dataset = "cifar_10_gcn_zca" # mnist omniglot cifar-100
run_suffix = "adam"
(modelfactory, model_name) = alex_cifar_tutorial_nn(10)

include("$(scripts_path)cifar10_alex_defaults.jl")
specs[:model] = "alex"

specs[:nworkers] = 1
specs[:initparams] = :Xavier
specs[:sampler] = :Adam
specs[:sampler_workers] = :Adam
specs[:batchsize] = 100

specs[:nitersinitial] = 600 * 300
specs[:stepsizeinitial] = 0.001
specs[:injectnoiseinitial] = 0.0
specs[:averagegradinitial] = true

specs[:niters] = 0
specs[:stepsize] = 5e-3
specs[:masterstepsize] = 0.00075
specs[:injectnoise] = 0.
specs[:averagegrad] = true

specs[:injectnoiseworker] = 0.
specs[:averagegradworker] = false
specs[:use_aws] = false
varyparam = :nworkers
varyvalues = repmat([1], 3)[:]
