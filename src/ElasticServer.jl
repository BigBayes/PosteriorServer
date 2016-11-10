module ElasticServer

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
  params::Array{Float}
  itersync::Int

  workers::Array{Remote}
  snapper::Remote{Snapper}
  examiner::Remote{Examiner}
  examfac::ExamFacility

  exception::Any
  go::Condition
  itersworkers::Dict{AbstractString,Int}
  specs::Dict{Symbol,Any}
end

abstract SyncFacility

type ElasticSyncFacility <: SyncFacility
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
  specs::Dict{Symbol,Any}
end


include("Snapper.jl")
include("Examiner.jl")
include("WorkerExaminer.jl")
include("ExamFacility.jl")
include("WorkerExamFacility.jl")
include("ElasticMaster.jl")
include("ElasticSyncFacility.jl")
include("ElasticWorker.jl")

function Specs(;
  precision::Type = Float64,
  nworkers::Int = 8,
  sampler::Symbol = :Adam,
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

  moving_rate::Float = 0.5,
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
    moving_rate,
    regu_coef)
end

function run(
  name::AbstractString,
  filename::AbstractString,
  samplerstate::SamplerState,
  masterid::Int64,
  snapper::Remote{Snapper},
  dmexaminer::Remote{AbstractDataModel},
  dminitworker::Remote{AbstractDataModel},
  dmworkers::Array{Remote{AbstractDataModel}},
  dmworkerexaminers::Nullable{Array{Remote{AbstractDataModel},1}},
  testclasses::Array{Float64,1}; # Refs of DataModelWorker
  specs = ElasticServer.Specs(keyargs...),
  keyargs...)

  nparams = fetchnparams(dminitworker)
  nworkers = length(dmworkers)
  println("=== Specs ===")
  @show specs
  println("=== $name: Starting up, #params = $(nparams) ===")
  examiner = RemoteExaminer(dmexaminer,snapper = snapper)

  setStepSize(samplerstate,specs[:stepsizeinitial])
  setInjectNoise(samplerstate,0.0)

  if specs[:nitersinitial] > 0
    initworker = RemoteWorker("$(name)_initworker",dminitworker, samplerstate, specs,
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
    setStepSize(samplerstate,specs[:stepsize])

    master = Nullable(Remote{Master}(@spawnat masterid begin
      Master(
        "$(name)_master",
        getState(samplerstate),
        nworkers,
        snapper,
        examiner,
        specs,
        nsyncsperexamine = specs[:nsyncsperexaminemaster])
    end))

    setInjectNoise(samplerstate,specs[:injectnoise])

    result = cell(nworkers)
    for worker = 1:nworkers
      workerid = where(dmworkers[worker])

      println("=== $name: Spawning worker $worker ===")
      result[worker] = @spawnat workerid begin
        dmworker = fetch(dmworkers[worker])::AbstractDataModel

        setStepSize(samplerstate,specs[:stepsize])

        worker = Worker(
                  "$(name)_worker$worker",
                  dmworker,
                  samplerstate,
                  specs,
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