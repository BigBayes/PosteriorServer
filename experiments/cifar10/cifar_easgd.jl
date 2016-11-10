# User experiment setting variables
method = "easgd"	# sgd distbayes sgld
dataset = "cifar_10_gcn_zca" # mnist omniglot cifar-100
run_suffix = "movingrate"
(modelfactory, model_name) = alex_cifar_tutorial_nn(10)

include("$(scripts_path)cifar10_alex_defaults.jl")
specs[:model] = "alex"

specs[:nworkers] = 8	# NOTE: This line doesn't have any effect when varyparam is :nworkers => change varyvalues instead!
specs[:initparams] = :Xavier
specs[:sampler] = :Adam
specs[:batchsize] = 100

specs[:nitersinitial] = 200
specs[:stepsizeinitial] = 1e-3
specs[:injectnoiseinitial] = 0.
specs[:averagegradinitial] = true
specs[:regu_coef_initial] = 0.

specs[:niters] = 500*15
specs[:stepsize] = 0.0025
specs[:masterstepsize] = 0.
specs[:injectnoise] = 0.
specs[:averagegrad] = true

# what parameters to vary for this experiment
specs[:moving_rate] = 0.3 # or should it be 0.25?
specs[:niters] = 30*500
specs[:regu_coef] = 0.
specs[:use_aws] = false

varyparam = :nworkers
#varyvalues = repmat([0.1 0.2 0.25 0.3 0.35 0.5], 5)
#varyvalues = repmat([0.1 0.1 0.1 0.1 0.1 0.2 0.2 0.2], 1)
varyvalues = repmat([8], 3)[:]
