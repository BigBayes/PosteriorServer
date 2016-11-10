# User experiment setting variables
method = "downpour"	# sgd distbayes sgld
dataset = "cifar_10_gcn_zca" # mnist omniglot cifar-100
run_suffix = "base"
(modelfactory, model_name) = alex_cifar_tutorial_nn(10)

include("$(scripts_path)cifar10_alex_defaults.jl")
specs[:model] = "alex"

specs[:nworkers] = 8	# NOTE: This line doesn't have any effect when varyparam is :nworkers => change varyvalues instead!
specs[:initparams] = :Xavier
specs[:sampler] = :Adam
specs[:sampler_workers] = :Adam
specs[:batchsize] = 100

specs[:nitersinitial] = 200
specs[:stepsizeinitial] = 1e-3
specs[:injectnoiseinitial] = 0.
specs[:averagegradinitial] = true

specs[:niters] = 500*30
specs[:stepsize] = 0.0025
specs[:masterstepsize] = 0.0025 / 8
specs[:injectnoise] = 0.
specs[:averagegrad] = true

specs[:injectnoiseworker] = 0.
specs[:averagegradworker] = true

specs[:regu_coef] = 0.
specs[:regu_coef_initial] = 0.

specs[:use_aws] = false
# what parameters to vary for this experiment
varyparam = :averagegradworker
varyvalues = repmat([true], 3)[:]
