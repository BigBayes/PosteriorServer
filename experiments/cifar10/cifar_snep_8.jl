# User experiment setting variables
method = "postserver"    # sgd distbayes sgld
dataset = "cifar_10_gcn_zca" # mnist omniglot cifar-100 cifar-10
experiment_type = "base" # base minibatch workers
run_suffix = "base_decreasing" # <= Use if you want to do several runs of same experiment to s$
(modelfactory, model_name) = alex_cifar_tutorial_nn(10) # make_dense_nn([100])   # make_$
include("$(scripts_path)cifar10_alex_defaults.jl")
specs[:model] = "alex"
specs[:priorvar] = 100.0
specs[:nworkers] = 8
specs[:initparams] = :Xavier
specs[:initvar] = .01
specs[:initlikvar] = .01
specs[:sampler] = :Adam
specs[:batchsize] = 100
specs[:nitersperdamp] = 10
specs[:synchronizedamping] = false
specs[:temperature] = 1.0
specs[:averagepredprobs] = false

specs[:nitersinitialopt] = 0
specs[:learnrateinitialopt] = 0.0
specs[:stepsizeinitialopt] = 1e-3

specs[:nitersinitialmcmc] = 0
specs[:learnrateinitialmcmc] = 1e-2
specs[:stepsizeinitialmcmc] = 1e-3
specs[:injectnoisemcmc] = 0.1
specs[:averagegradmcmc] = false
specs[:saveinitstate] = ""

specs[:niters] = 500*30
specs[:learnrate] =  niters::Int -> if niters<=5000 s=2e-2 else s=2e-3 end
specs[:stepsize] = niters::Int -> if niters<=5000 s=1e-3 else s=1e-4 end
specs[:injectnoise] = 1.0
specs[:averagegrad] = false

specs[:nitersstartaveraging] = 500*20
specs[:adaptation] = :none
specs[:use_aws] = false
# what parameters to vary for this experiment
varyparam = :nworkers
varyvalues = repmat([8],3)
