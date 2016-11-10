# NOTE: Assumes called from run.jl, which sets up necessary variables
# Start workers
# NOTE: We now create an additional process to do the validation

using Compat

addprocs(max(0,specs[:nworkers]+4 - nprocs()))
snapperid         = 1
initworkerid      = 1
w                 = workers()
workerids         = w[1:specs[:nworkers]]
workerexaminerids = Nullable{Array{Int}}()
avgprobmasterid   = Nullable{Int}()
examinerid        = w[specs[:nworkers]+1]
masterid          = w[specs[:nworkers]+2]
trainexaminerid   = w[specs[:nworkers]+3]

# Note @everywhere does not do implicit variable copying to other processes
@everywhere begin
  include("paths.jl")
  push!(LOAD_PATH, "$(source_path)")
  using Compat
end

# Mocha somehow complains a lot if loaded with other stuff
@everywhere begin
  using SGMCMC
  using MLUtilities
  using Utilities
  using DataModel
end
@everywhere begin
  using LogRegDataModel
  using PosteriorServer
  using AbstractGaussian
  using DiagGaussian
  using FullCovGaussian
end
using HDF5

snapper = PosteriorServer.Snapper(snapperid)

println("=== Setting up workers & examiners ===")
dms = LogRegDataModel.startup(
                      logReg_datX,
                      logReg_datY,
                      examinerid,
                      trainexaminerid,
                      initworkerid,
                      workerids,
                      workerexaminerids
                      )


# determine number of parameters
nparams = fetchnparams(dms[:initworker])

# set up prior and worker states
if specs[:fullcov]
  prior = ones(FullCovGaussian.NatParam, nparams) / specs[:priorvar]
  initworkerstate = PosteriorServer.WorkerState(
    ones(FullCovGaussian.NatParam,nparams)/specs[:initlikvar])
else
  prior = ones(DiagGaussian.NatParam, nparams) / specs[:priorvar]
  initworkerstate = PosteriorServer.WorkerState(
    ones(DiagGaussian.NatParam,nparams)/specs[:initlikvar])
end

if specs[:initparams] == :Gaussian
  init_gaussian(dms[:initworker], specs[:initvar])
elseif specs[:initparams] == :Xavier
  do_nothing = true
elseif specs[:initparams] == :Uniform
  init_uniform(dms[:initworker], specs[:initvar])
elseif specs[:initparams] == :Simple_Fanin
  init_simple_fanin(dms[:initworker])
else
  error("Unknown method of initialising params")
end

x = mean(prior)

if specs[:sampler] == :Adam
  initsamplerstate = AdamState(x,
    averagegrad = true)
else
  error("Unknown sampler")
end

initworkerstate.likapprox = prior

results = PosteriorServer.run(
    experiment_name,
    "$(save_result_path)$(file_name)",
    prior,
    initworkerstate,
    initsamplerstate,
    masterid,
    avgprobmasterid,
    snapper,
    dms[:examiner],
    dms[:trainexaminer],
    dms[:initworker],
    dms[:workers],
    dms[:workerexaminers],
    [0.],
    specs = specs)

JLD.save("$(save_result_path)$(file_name).jld",
    "specs",specs,
    "results",results)
println("=== Results saved ===")

@everywhere try
    interrupt(myid())
end
rmprocs(workers())

#wait for processes to shut down.
while workers() != [1]
    sleep(1)
end
