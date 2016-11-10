module MochaDataModel

### Implements a data model for use with Mocha.jl for neural networks for
### classification.

using Compat
using Mocha
using MochaWrapper
using SGMCMC
using AbstractGaussian
using MLUtilities
using Utilities
using DataModel

export MochaSGMCMCDataModel, RemoteMochaSGMCMCDataModel, startup

### startup sets up the required data models for PosteriorServer.run
function startup(
  trainx,                   # training images
  trainc,                   # training classes
  testx,                    # test images
  testc,                    # test classes
  modelfactory,             # function that construct the Mocha Neural Network
  batchsize,
  examinerid,               # process id of global examiner
  trainexaminerid,          # process id of global training examiner
  initworkerid,             # process id of initial worker (not used in experiments)
  workerids,                # worker process ids
  workerexaminerids;        # worker examiner ids
  allworkeralldata = false,
  averagepredprobs = false,
  temperature = 1.0,
  use_gpu = false)

  ntrain    = length(trainc)
  ntest     = length(testc)
  nworkers  = length(workerids)

  println("=== Set up master examiner ===")
  examiner = Remote{AbstractDataModel}(@spawnat examinerid begin
    try
        examinerbackend = initMochaBackend(use_gpu)
        MochaSGMCMCDataModel(
          testx,testc,
          modelfactory,
          examinerbackend,
          batchsize = batchsize,
          temperature = temperature,
          do_shuffle = false,
          do_accuracy = true)
      catch
          error("Examiner set up failed.")
      end
  end)

  trainexaminer = Remote{AbstractDataModel}(@spawnat trainexaminerid begin
    try
        examinerbackend = initMochaBackend(use_gpu)
        MochaSGMCMCDataModel(
          trainx,trainc,
          modelfactory,
          examinerbackend,
          batchsize = batchsize,
          temperature = temperature,
          do_shuffle = false,
          do_accuracy = true)
      catch
          error("Examiner set up failed.")
      end
  end)

  println("=== Set up worker examiners ===")
  if averagepredprobs
      workerexaminers = Array(Remote{AbstractDataModel},nworkers)
      for i in 1:nworkers
          workerexaminers[i] = Remote{AbstractDataModel}(@spawnat get(workerexaminerids)[i] begin
            try
                examinerbackend = initMochaBackend(use_gpu)
                MochaSGMCMCDataModel(
                  testx,testc,
                  modelfactory,
                  examinerbackend,
                  batchsize = batchsize,
                  temperature = temperature,
                  do_shuffle = false,
                  do_accuracy = false,
                  do_predprob = true)
              catch
                  error("Examiner $(myid()) set up failed.")
              end
          end)
      end
      workerexaminers = Nullable(workerexaminers)
  else
      # no worker examiners required
      workerexaminers = Nullable{Array{Remote{AbstractDataModel},1}}()
  end

  # randomly assign data to workers
  assignments = shuffle(rem(convert(Array,1:ntrain),nworkers)+1)
  workers = Array(Remote{AbstractDataModel},nworkers)


  println("=== Set up initial worker === ")
  initworker = Remote{AbstractDataModel}(@spawnat initworkerid begin
    try
        initworkerbackend = initMochaBackend(use_gpu)
        MochaSGMCMCDataModel(
          allworkeralldata ? trainx : trainx[:,:,:,assignments.==1],
          allworkeralldata ? trainc : trainc[:,assignments.==1],
          modelfactory,
          initworkerbackend,
          ntrain = float(ntrain), #float(ntrain/nworkers),
          batchsize = batchsize,
          temperature = temperature)
      catch
          error("Initworker set up failed.")
      end
  end)

  println("=== Set up workers === ")
  for worker = 1:nworkers
      # process id for worker
    workerid = workerids[worker]

    # assign random subset
    wtrainx = allworkeralldata ? trainx : trainx[:,:,:,assignments.==worker]
    wtrainc = allworkeralldata ? trainc : trainc[:,assignments.==worker]
    workers[worker] = Remote{AbstractDataModel}(@spawnat workerid begin
        try
          workerbackend = initMochaBackend(use_gpu)
          MochaSGMCMCDataModel(
            wtrainx,wtrainc,
            modelfactory,
            workerbackend,
            ntrain = float(allworkeralldata ? ntrain/nworkers : length(wtrainc)),
            batchsize = batchsize,
            temperature = temperature)
        catch
            error("Worker $(myid()) set up failed." )
        end
    end)
  end
  # return dictionary of remote references
  return @dict(examiner,trainexaminer,initworker,workers,workerexaminers)
end

# Data model type
type MochaSGMCMCDataModel <: AbstractDataModel
  backend::Backend                  # Backend (required for Mocha)
  mochaNet::MochaWrapper.MWrap     # Neural Net wrapper object
  labels::Array{Int64,1}            # labels
  ntrain::Float64                   # effective training set size
  batchsize::Int64
  temperature::Float64
end


function RemoteMochaSGMCMCDataModel(where::Int, args...; keyargs...)
  Remote{AbstractDataModel}(@spawnat where MochaSGMCMCDataModel(args...; keyargs...))
end

# constructor
function MochaSGMCMCDataModel(
  datax,datac,                          # images and labels
  modelfactory::Function,               # function to construct neural net
  backend::Backend;
  ntrain::Float64 = float(length(datac)),
  batchsize::Int = 100,
  temperature::Float64 = 1.0,
  do_shuffle::Bool = true,              # shuffle mini batches
  do_accuracy::Bool = false,
  do_predprob::Bool = false
  )

  # set up Mocha MemoryDataLayer
  data_layer = MemoryDataLayer(name = "data",
                               data = Array[datax,datac],
                               batch_size = batchsize,
                               shuffle = do_shuffle)

    # set up Mocha Wrapper
  mochaNet = MochaWrapper.MWrap(data_layer,
                                 modelfactory,
                                 "MochaSGMCMCNet",
                                 do_accuracy,
                                 do_predprob,
                                 backend)
  MochaSGMCMCDataModel(backend,mochaNet,datac[:],ntrain,batchsize,temperature)
end

function Base.show(io::IO, x::MochaSGMCMCDataModel)
  print(io, "MochaSGMCMCDataModel($(x.ntrain),$(x.batchsize))")
end

# fetch parameters
function DataModel.fetchparams(dms::MochaSGMCMCDataModel)
  MochaWrapper.getparams(dms.mochaNet)
end

# fetch number of parameters
function DataModel.fetchnparams(dms::MochaSGMCMCDataModel)
  MochaWrapper.getnparams(dms.mochaNet)
end

# various initialisation
function DataModel.init_xavier(dms::MochaSGMCMCDataModel)
  MochaWrapper.init_xavier(dms.mochaNet)
end
function DataModel.init_simple_fanin(dms::MochaSGMCMCDataModel)
  MochaWrapper.init_simple_fanin(dms.mochaNet)
end
function DataModel.init_gaussian(dms::MochaSGMCMCDataModel, initvar::Float64)
  MochaWrapper.init_gaussian(dms.mochaNet, initvar)
end
function DataModel.init_uniform(dms::MochaSGMCMCDataModel, initvar::Float64)
  MochaWrapper.init_uniform(dms.mochaNet, initvar)
end

# sample one step using MCMC method and stochastic gradients from data model.
function DataModel.sample!(dms::MochaSGMCMCDataModel,
  state::SamplerState,
  prior::AbstractGaussian.NatParam,
  beta::Float64
  )

  function learngrad(x::Vector{Float64})
    (llik,grad) = MochaWrapper.evaluateNN(dms.mochaNet, x)
    return vec(grad) * dms.ntrain / beta / dms.temperature + gradloglik(prior,x)
  end

  SGMCMC.sample!(state, learngrad)
end

# return average gradient for minibatch
function DataModel.evaluateGrad(dms::MochaSGMCMCDataModel,
                                params::Vector{Float64};
                                regu_coef::Float64 = 0.)
  (llik, grad) = MochaWrapper.evaluateNN(dms.mochaNet, params, regu_coef = regu_coef)
  return vec(grad)
end

# evaluate accuracy based on predictive probabilities.
function DataModel.evaluate(dms::MochaSGMCMCDataModel,
  predprobs::Array{Float64,2}
  )
  prediction =  (findmax(predprobs,1)[2]-1.) % size(predprobs)[1]
  accuracy = sum(dms.labels .== prediction[:])./(length(dms.labels)+0.0)
  @dict(accuracy)
end

# evaluate accuracy and loglikelihood
function DataModel.evaluate(dms::MochaSGMCMCDataModel,
  x::Vector{Float64}
  )
  (accuracy, loglikelihood) = MochaWrapper.evaluateTestNN(dms.mochaNet, x, dms.batchsize)
  @dict(accuracy, loglikelihood)
end
function DataModel.evaluate(dms::MochaSGMCMCDataModel,
  posterior::AbstractGaussian.NatParam
  )
  evaluate(dms,mean(posterior))
end
function DataModel.evaluate(dms::MochaSGMCMCDataModel,
  posterior::AbstractGaussian.MeanParam
  )
  evaluate(dms,mean(posterior))
end

# access class probabilities from Mocha.
function DataModel.evaluatePredProb(dms::MochaSGMCMCDataModel,
  x::Vector{Float64}
  )
  predictiveprobs = MochaWrapper.evaluateTestNNPredProb(dms.mochaNet, x, dms.batchsize)
  @dict(predictiveprobs)
end
function DataModel.evaluatePredProb(dms::MochaSGMCMCDataModel,
  posterior::AbstractGaussian.NatParam
  )
  DataModel.evaluatePredProb(dms,mean(posterior))
end
function DataModel.evaluatePredProb(dms::MochaSGMCMCDataModel,
  posterior::AbstractGaussian.MeanParam
  )
  DataModel.evaluatePredProb(dms,mean(posterior))
end


end
