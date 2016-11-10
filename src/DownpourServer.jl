module DownpourServer

using Compat
using ExpFamily
using AbstractGaussian
using MLUtilities
using Utilities
using SGMCMC
using JLD
using DataModel
using Formatting

typealias Float Float64

type Snapshot
  time::Float
  action::Symbol
  data::Any
  Snapshot(time,action) = new(time,action,())
  Snapshot(time,action,data...) = new(time,action,data)
end

type Snapper
  inittime::Float
  io::IO
  snapshots::Array{Snapshot}
  tostream::Array{Symbol}
end

abstract AbstractExaminer

type Examiner <: AbstractExaminer
  status::Symbol
  dm::AbstractDataModel
  snapper::Nullable{Remote{Snapper}}
  train::Bool
end

type WorkerExaminer <: AbstractExaminer
    status::Symbol
    dm::AbstractDataModel
    snapper::Nullable{Remote{Snapper}}
end

type ExamFacility
  bouncer::Bouncer
  examiner::Remote{Examiner}
  waiting::Bool
  results::RemoteRef
end

type WorkerExamFacility
  bouncer::Bouncer
  examiner::Remote{WorkerExaminer}
  waiting::Bool
  results::RemoteRef
end

type Master
  name::AbstractString
  status::Symbol
  samplerstate::SamplerState
  itersync::Int

  workers::Array{Remote}
  snapper::Remote{Snapper}
  examiner::Remote{Examiner}
  examfac::ExamFacility

  exception::Any
  go::Condition
  itersworkers::Dict{AbstractString,Int}
end

abstract SyncFacility

type SyncGradFacility <: SyncFacility
  bouncer::Bouncer
  waiting::Bool
  master::Remote{Master}
  results::RemoteRef
end

type Worker
  name::AbstractString
  status::Symbol
  iter::Int

  dmworker::AbstractDataModel

  # State information
  samplerstate::SamplerState
  master::Nullable{Remote{Master}}
  syncfac::Nullable{SyncFacility}
  snapper::Nullable{Remote{Snapper}}
  examfac::Nullable{ExamFacility}

end


include("Snapper.jl")
include("Examiner.jl")
include("WorkerExaminer.jl")
include("ExamFacility.jl")
include("WorkerExamFacility.jl")
include("DownpourMaster.jl")
include("SyncDownpourFacility.jl")
include("DownpourWorker.jl")

function Specs(;
  precision::Type = Float64,
  nworkers::Int = 8,
  sampler::Symbol = :Adam,
  sampler_workers::Symbol = :Adam,
  initparams::Symbol = :Gaussian,
  initvar::Float = 1.0,

  loadinitstate::AbstractString = "",
  saveinitstate::AbstractString = "",

  batchsize::Int = 100,

  stepsizeinitial::Float = 1e-2,
  nitersinitial::Int64 = 200,

  stepsize::Float = 1e-2,
  niters::Int64 = 10000,

  niterspersync::Int64 = 10,

  nitersperexamineworker::Int64 = 0,
  nitersperexamineinitial::Int64 = 10,

  nsyncsperexaminemaster::Int64 = 10,
  regu_coef::Float = 0.0005
  )

  return @dict(
    precision,
    nworkers,
    batchsize,
    loadinitstate,
    saveinitstate,
    sampler,
    initparams,
    initvar,
    stepsizeinitial,
    nitersinitial,
    stepsize,
    niters,
    niterspersync,
    nsyncsperexaminemaster,
    nitersperexamineinitial,
    nitersperexamineworker,
    regu_coef)
end

function run(
  name::AbstractString,
  filename::AbstractString,
  samplerstate::SamplerState,
  samplerstate_workers::SamplerState,
  masterid::Int64,
  snapper::Remote{Snapper},
  dmexaminer::Remote{AbstractDataModel},
  dminitworker::Remote{AbstractDataModel},
  dmworkers::Array{Remote{AbstractDataModel}},
  dmworkerexaminers::Nullable{Array{Remote{AbstractDataModel},1}},
  testclasses::Array{Float64,1}; # Refs of DataModelWorker
  specs = DownpourServer.Specs(keyargs...),
  keyargs...)

  nparams = fetchnparams(dminitworker)
  nworkers = length(dmworkers)
  println("=== Specs ===")
  @show specs
  println("=== $name: Starting up, #params = $(nparams) ===")
  examiner = RemoteExaminer(dmexaminer,snapper = snapper)

  setStepSize(samplerstate,specs[:stepsizeinitial])
  setInjectNoise(samplerstate,specs[:injectnoiseinitial])
  setAverageGrad(samplerstate,specs[:averagegradinitial])

  if specs[:nitersinitial] > 0
    initworker = RemoteWorker("$(name)_initworker",dminitworker, samplerstate,
      snapper = Nullable(snapper),
      examiner = Nullable(examiner),
      nitersperexamine = specs[:nitersperexamineinitial]
      )
  end

  initial_learning_time = time()

  if specs[:nitersinitial] > 0
    println("=== $name: Initial OPT learning phase ===")
    runfetch(initworker,
      specs[:nitersinitial],
      regu_coef = specs[:regu_coef_initial]
      )
    samplerstate = fetchstate(initworker)
    println("=== $name: Initial OPT learning phase done! === ")
  end

  if specs[:nitersinitialopt] > 0
    shutdown(initworker)
  end

  initial_learning_time = time() - initial_learning_time

  # Distributed phase of learning

  if specs[:niters] == 0
    snapshots = fetchsnapshots(snapper)
    results = @dict(snapshots, initial_learning_time)
  else
    setStepSize(samplerstate,specs[:masterstepsize])
    setInjectNoise(samplerstate,specs[:injectnoise])
    setAverageGrad(samplerstate,specs[:averagegrad])

    master = Nullable(Remote{Master}(@spawnat masterid begin
      Master(
        "$(name)_master",
        samplerstate,
        nworkers,
        snapper,
        examiner,
        nsyncsperexamine = specs[:nsyncsperexaminemaster])
    end))

    result = cell(nworkers)
    for worker = 1:nworkers
      workerid = where(dmworkers[worker])

      if specs[:sampler] == specs[:sampler_workers]
        samplerstate_workers = samplerstate
      end


      println("=== $name: Spawning worker $worker ===")
      result[worker] = @spawnat workerid begin
        dmworker = fetch(dmworkers[worker])::AbstractDataModel

        setStepSize(samplerstate_workers,specs[:stepsize])
        setInjectNoise(samplerstate_workers,specs[:injectnoiseworker])
        setAverageGrad(samplerstate_workers,specs[:averagegradworker])

        worker = Worker(
                  "$(name)_worker$worker",
                  dmworker,
                  samplerstate_workers,
                  master = master,
                  snapper = snapper,
                  niterspersync = specs[:niterspersync]
                )

        run(worker,
          specs[:niters],
          regu_coef = specs[:regu_coef]
          )
      end # spawnat
    end #for worker


    @sync begin
      numdone = 0
      incdone() = numdone += 1
      for id = 1:nworkers
        @async begin
          fetch(result[id])
          println("=== $name: Completed worker $id ($(incdone())/$nworkers done) ===")
          sleep(.1)
        end
      end
    end
    snapshots = fetchsnapshots(snapper)
    results = @dict(snapshots, initial_learning_time)
    shutdown(get(master))
  end

  shutdown(examiner)
  for i in 0:length(snapshots)-1
      if snapshots[end-i].action == :endExam
          println("=== Final accuracy ",snapshots[end-i].data[1][:params][:accuracy])
          break
      end
  end
  println("=== $name: Done ===")
  return results
end

end # module PosteriorServer
