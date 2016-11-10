# NOTE: Assumes called from run.jl, which sets up necessary variables
# Start workers

using Compat

if haskey(specs, :psnep) && specs[:psnep] == true
    print("Enabling p-SNEP")
    specs[:beta] = 1. / specs[:nworkers]
end

# add additional processes
if specs[:averagepredprobs]
    addprocs(max(0,2*specs[:nworkers]+5 - nprocs()))
    snapperid = 1
    initworkerid = 1
    w = workers()
    workerids = w[1:specs[:nworkers]]
    workerexaminerids = Nullable(w[(specs[:nworkers]+1):(2*specs[:nworkers])])
    examinerid = w[2*specs[:nworkers]+1]
    masterid = w[2*specs[:nworkers]+2]
    avgprobmasterid = Nullable(w[2*specs[:nworkers]+3])
    trainexaminerid = w[2*specs[:nworkers]+4]
else
    addprocs(max(0,specs[:nworkers]+4 - nprocs()))
    snapperid = 1
    initworkerid = 1
    w = workers()
    workerids = w[1:specs[:nworkers]]
    workerexaminerids = Nullable{Array{Int}}()
    avgprobmasterid = Nullable{Int}()
    examinerid = w[specs[:nworkers]+1]
    masterid = w[specs[:nworkers]+2]
    trainexaminerid = w[specs[:nworkers]+3]
end


# Note @everywhere does not do implicit variable copying to other processes
@everywhere begin
  include("paths.jl")
  push!(LOAD_PATH, "$(source_path)")
  use_gpu = false
  if use_gpu
    ENV["MOCHA_USE_CUDA"] = "true"
    error("GPU use is not currently implemented")
  else
    ENV["MOCHA_USE_NATIVE_EXT"] = "true"
    ENV["OMP_NUM_THREADS"] = 1
  end

end

@everywhere using Mocha

@everywhere begin
  using SGMCMC
  using MochaWrapper
  using MLUtilities
  using Utilities
  using DataModel
end
@everywhere begin
  using MochaDataModel
  using PosteriorServer
  include( "$(source_path)models.jl")
  using AbstractGaussian
  using DiagGaussian
  using FullCovGaussian
end
using HDF5

# initialise snapper
snapper = PosteriorServer.Snapper(snapperid)

# Load data
trainFile = h5open( "$(data_path)$(dataset)_train.hdf5", "r" )
testFile  = h5open( "$(data_path)$(dataset)_test.hdf5", "r" )
images    = convert(Array{Float64,4},trainFile["data"][:,:,:,:])
dlabel    = convert(Array{Float64,2},trainFile["label"][:,:])
timages   = convert(Array{Float64,4},testFile["data"][:,:,:,:])
tdlabel   = convert(Array{Float64,2},testFile["label"][:,:])

println("=== Setting up workers & examiners ===")
dms = MochaDataModel.startup(
                      images,
                      dlabel,
                      timages,
                      tdlabel,
                      modelfactory,
                      specs[:batchsize],
                      examinerid,
                      trainexaminerid,
                      initworkerid,
                      workerids,
                      workerexaminerids,
                      temperature = specs[:temperature],
                      averagepredprobs = specs[:averagepredprobs])

# determine number of parameters
nparams = fetchnparams(dms[:initworker])
# set up prior and worker states
if specs[:fullcov]
  prior = ones(FullCovGaussian.NatParam,nparams) / specs[:priorvar]
  initworkerstate = PosteriorServer.WorkerState(
    ones(FullCovGaussian.NatParam,nparams)/specs[:initlikvar])
else
  prior = ones(DiagGaussian.NatParam,nparams) / specs[:priorvar]
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

x = fetchparams(dms[:initworker])


if specs[:sampler] == :Adam
  initsamplerstate = AdamState(x,
    averagegrad = true)
else
  error("Unknown sampler")
end

# set up initial likelihood approximation
initworkerstate.likapprox = DiagGaussian.NatParam(mean=x, var= ones(nparams)*specs[:initlikvar])

# run PosteriorServer
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
    tdlabel[:],
    specs = specs)


# save results
JLD.save("$(save_result_path)$(file_name).jld",
  "specs",specs,
  "results",results)
println("=== Results saved ===")

# remove processes
@everywhere try
    interrupt(myid())
end
rmprocs(workers())

#wait for processes to shut down.
while workers() != [1]
    sleep(1)
end
