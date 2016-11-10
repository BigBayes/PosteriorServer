# User experiment setting variables
method = "postserver"	# sgd distbayes sgld
dataset = "mnist" # mnist omniglot cifar-100
run_suffix = "test"
(modelfactory, model_name) = make_dense_nn([100],10)
include("$(scripts_path)mnist_defaults.jl")
specs[:model] = "100"
specs[:nworkers] = 2
specs[:initparams] = :Xavier
specs[:initlikvar] = 0.1
specs[:priorvar] = 100.
specs[:sampler] = :Adam
specs[:batchsize] = 100
specs[:averagepredprobs] = false

specs[:nitersinitialopt] = 0
specs[:learnrateinitialopt] = 1e-2
specs[:stepsizeinitialopt] = 1e-3

specs[:nitersinitial] = 0
specs[:learnrateinitial] = 1e-2
specs[:stepsizeinitial] = 1e-3
specs[:injectnoise] = 0.

specs[:nitersinitialmcmc] = 0
specs[:learnrateinitialmcmc] = 1e-2
specs[:stepsizeinitialmcmc] = 1e-3
specs[:injectnoisemcmc] = 0.1
specs[:averagegradmcmc] = false

specs[:niters] = 1200
specs[:niterspersync] = 1
specs[:learnrate] =  0.02
specs[:stepsize] = 1e-3
specs[:injectnoise] = 1.0
specs[:niterssample] = 1
specs[:nitersperexamine] = 1

#specs[:averagegrad] = false
specs[:varlimits] = (1e-2,1e2)
specs[:nitersstartaveraging]     = 60
specs[:synchronizedamping] = false
specs[:adaptation] = :none
specs[:default_plot] = true

# what parameters to vary for this experiment
varyparam = :nworkers
varyvalues = [4]
