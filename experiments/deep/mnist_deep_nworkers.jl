# User experiment setting variables
method = "postserver"   # sgd distbayes sgld
dataset = "mnist" # mnist omniglot cifar-100
run_suffix = "nworkers"
(modelfactory, model_name) = make_dense_nn([50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50],10)
include("$(scripts_path)mnist_defaults.jl")
specs[:model] = "deep"
specs[:nworkers] = 8
specs[:initparams] = :Xavier
specs[:initlikvar] = 0.01
specs[:priorvar] = .025 #.05
specs[:sampler] = :Adam
specs[:psnep] = true

specs[:batchsize] = 100
specs[:averagepredprobs] = false

specs[:nitersinitialopt] = 0
specs[:stepsizeinitialopt] = 0.00065
specs[:injectnoise] = 0.
specs[:averagegradinitial] = true

specs[:nitersinitialmcmc] = 0
specs[:learnrateinitialmcmc] = 2e-2
specs[:stepsizeinitialmcmc] = 1e-3
specs[:injectnoisemcmc] = 0.1
specs[:averagegradmcmc] = false

specs[:niters] = 600 * 50
specs[:learnrate] = 0.04
specs[:stepsize] = 0.00065
specs[:injectnoise] = 1.0
specs[:averagegrad] = false
specs[:nitersstartaveraging]  = 600*50
specs[:synchronizedamping] = false
specs[:adaptation] = :none
specs[:beta] = 1.0 / 8.0

# what parameters to vary for this experiment
varyparam = :nworkers
varyvalues = repmat([2 4 6 8 16], 10)[:]