# NOTE: Assumes called from run.jl, which sets up necessary variables
# Start workers

using Compat

addprocs(max(0,specs[:nworkers]+4 - nprocs()))
snapperid = 1
initworkerid = 1
w = workers()
workerids = w[1:specs[:nworkers]]
workerexaminerids = Nullable{Array{Int}}()
examinerid = w[specs[:nworkers]+1]
masterid = w[specs[:nworkers]+2]
trainexaminerid = w[specs[:nworkers]+3]


# Note @everywhere does not do implicit variable copying to other processes
@everywhere begin
  include("paths.jl")
  push!(LOAD_PATH, "$(source_path)")
  use_gpu = false
  ENV["MOCHA_USE_NATIVE_EXT"] = "true"
  ENV["OMP_NUM_THREADS"] = 1
end

@everywhere using Mocha

# Mocha somehow complains a lot if loaded with other stuff
@everywhere begin
  using SGMCMC
  using MochaWrapper
  using MLUtilities
  using Utilities
  using DataModel
end

@everywhere begin
  using MochaDataModel
  using DownpourServer
  include( "$(source_path)models.jl")
end
using HDF5

snapper = DownpourServer.Snapper(snapperid)

# Load data
trainFile = h5open( "$(data_path)$(dataset)_train.hdf5", "r" )
testFile  = h5open( "$(data_path)$(dataset)_test.hdf5", "r" )
images    = convert(Array{Float64, 4},trainFile["data"][:,:,:,:])
dlabel    = convert(Array{Float64, 2},trainFile["label"][:,:])
timages   = convert(Array{Float64, 4},testFile["data"][:,:,:,:])
tdlabel   = convert(Array{Float64, 2},testFile["label"][:,:])

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

nparams = fetchnparams(dms[:initworker])

# SW: I moved initialization to MochaWrapper2
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

if specs[:sampler] == :Adagrad
  initsamplerstate = AdagradState(x,
    injectnoise = 0.,
    minmass = specs[:minmass])
elseif specs[:sampler] == :Adam
  initsamplerstate = AdamState(x,
    injectnoise = 0.,
    averagegrad = true)
elseif specs[:sampler] == :Constant
  initsamplerstate = ConstantState(x)
else
  error("Unknown sampler")
end

if specs[:sampler_workers] == :Adagrad
  initsamplerstate_workers = AdagradState(x,
    injectnoise = 0.,
    minmass = specs[:minmass])
elseif specs[:sampler_workers] == :Adam
  initsamplerstate_workers = AdamState(x,
    injectnoise = 0.,
    averagegrad = true)
elseif specs[:sampler_workers] == :Constant
  initsamplerstate_workers = ConstantState(x)
else
  error("Unknown sampler")
end

results = DownpourServer.run(
    experiment_name,
    "$(save_result_path)$(file_name)",
    initsamplerstate,
    initsamplerstate_workers,
    masterid,
    snapper,
    dms[:examiner],
    dms[:initworker],
    dms[:workers],
    dms[:workerexaminers],
    tdlabel[:],
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
