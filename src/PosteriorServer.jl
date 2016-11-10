module PosteriorServer

### Implements the PosteriorServer

    using Compat
    using ExpFamily
    using AbstractGaussian
    using DiagGaussian
    using MLUtilities
    using Utilities
    using SGMCMC
    using JLD
    using DataModel
    using MochaDataModel
    using Formatting


    # declare types

    # used for generic saving during learning
    type Snapshot
      time::Float64             # time event occured
      action::Symbol            # action taken
      data::Any                 # data
      Snapshot(time,action) = new(time,action,())
      Snapshot(time,action,data...) = new(time,action,data)
    end

    # facility to save snapshots
    type Snapper
      inittime::Float64
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

    # facility to request exams
    type ExamFacility
      bouncer::Bouncer
      examiner::Remote{Examiner}
      waiting::Bool
      results::RemoteRef
    end

    # facility to request exams on worker
    type WorkerExamFacility
      bouncer::Bouncer
      examiner::Remote{WorkerExaminer}
      waiting::Bool
      results::RemoteRef
    end

    # Posterior Server
    type Master
      name::AbstractString
      status::Symbol
      natposterior::AbstractGaussian.NatParam       # global posterior approx (nat param)
      meanposterior::AbstractGaussian.MeanParam     # global posterior approx (mean param)
      avgnatposterior::AbstractGaussian.NatParam    # averaged global approximation
      avgparams::Array{Float64}                     # averaged parameters
      itersync::Int                                 # number of synchronizations

      workers::Array{Remote}                        # workers linked to master
      snapper::Remote{Snapper}
      examiner::Remote{Examiner}
      trainexaminer::Remote{Examiner}
      examfac::ExamFacility
      trainexamfac::ExamFacility

      nitersstartaveraging::Int                     # when to start averaging
      exception::Any
      go::Condition

      specs::Dict{Symbol,Any}
      itersworkers::Dict{AbstractString,Int}        # iterations on workers
    end

    # Server to average predictive probabilities
    type PredProbMaster
      name::AbstractString
      status::Symbol
      workerpredprobs::Array{Array{Float64},1}
      numpredprobs::Int
      testclasses::Array{Float64,1}
      master::Remote{Master}

      dimensions::Tuple

      avgpredprobs::Array{Float64}

      nitersstartaveraging::Int
      exception::Any
      snapper::Remote{Snapper}
    end

    # synchronization facilities
    abstract SyncFacility

    type SyncDampFacility <: SyncFacility
      bouncer::Bouncer
      waiting::Bool
      synchronizedamping::Bool
      lastpushedlikapprox::AbstractGaussian.NatParam
      lastpushedmeanpost::AbstractGaussian.MeanParam
      master::Remote{Master}
      results::RemoteRef

      specs::Dict{Symbol,Any}
      synctimes::Array{Float64,1}
    end

    # synchronizing predictive class probabilities
    type SyncPredProbFacility <: SyncFacility
        bouncer::Bouncer
        waiting::Bool
        lastpushed::Nullable{Array{Float64}}
        master::Remote{PredProbMaster}
        results::RemoteRef
    end

    type WorkerState
      likapprox::AbstractGaussian.NatParam
    end

    # worker object
    type Worker
      name::AbstractString
      status::Symbol
      iter::Int

      dmworker::AbstractDataModel               # data model
      condprior::AbstractGaussian.NatParam      # cavity distribution
      meanpost::AbstractGaussian.MeanParam      # local posterior approximation (mean param)
      damper::AbstractGaussian.NatParam         # auxiliary variables
      workerstate::WorkerState
      samplerstate::SamplerState                # MCMC state
      predprobs::Nullable{Array{Float64}}

      master::Nullable{Remote{Master}}
      avgprobmaster::Nullable{Remote{PredProbMaster}}
      syncfac::Nullable{SyncFacility}
      syncfacPredProb::Nullable{SyncPredProbFacility}
      snapper::Nullable{Remote{Snapper}}
      examfac::Nullable{ExamFacility}
      workerexamfac::Nullable{WorkerExamFacility}

      nitersperdamp::Int

      specs::Dict{Symbol,Any}
    end


    # include additional code
    include("Snapper.jl")
    include("Examiner.jl")
    include("WorkerExaminer.jl")
    include("ExamFacility.jl")
    include("WorkerExamFacility.jl")
    include("Master.jl")
    include("PredProbMaster.jl")
    include("SyncDampFacility.jl")
    include("SyncPredProbFacility.jl")
    include("Worker.jl")

    # default parameters
    function Specs(;
      use_gpu::Bool = false,
      nworkers::Int = 8,
      beta::Float64 = 1.0,
      priorvar::Float64 = 100.,
      fullcov::Bool = false,
      sampler::Symbol = :Adam,
      initparams::Symbol = :Gaussian,
      initvar::Float64 = 1.0,
      initlikvar::Float64 = 1.0,

      loadinitstate::AbstractString = "",
      saveinitstate::AbstractString = "",

      temperature::Float64 = 1.0,
      batchsize::Int = 100,

      learnrateinitialopt::Float64 = 1e-3,
      stepsizeinitialopt::Float64 = 1e-3,
      nitersinitialopt::Int64 = 200,

      learnrateinitialmcmc::Float64 = 1e-3,
      stepsizeinitialmcmc::Float64 = 1e-3,
      nitersinitialmcmc::Int64 = 200,
      injectnoisemcmc::Float64 = .1,
      averagegradmcmc::Bool = true,

      learnrate::Float64 = 1e-2,
      stepsize::Float64 = 1e-2,
      niters::Int64 = 10000,
      injectnoise::Float64 = 1.0,
      averagegrad::Bool = false,

      niterspersync::Int64 = 10,
      nitersperdamp::Int64 = 50,
      niterssample::Int64 = 1,
      synchronizedamping::Bool = false,

      nitersperexamineworker::Int64 = 0,
      nitersperexamineinitial::Int64 = 10,

      nsyncsperexaminemaster::Int64 = 10,
      nitersstartaveraging::Int64 = 1000,
      averagepredprobs::Bool = false,

      messageform = :standard,
      sep_alpha = 1. / 8.,

      meanlimits = (-10.0,10.0),
      varlimits = (1e-2,1e2),
      minmass = 1e-10,
      logReg = false,
      algep  = :snep,
      default_plot = true,
      use_aws = false,
      regu_coef::Float64 = 0.,
      regu_coef_initial::Float64 = 0.,
      shiftstates = true,
      psnep::Bool = false)

      return @dict(
        use_gpu,
        nworkers,
        beta,
        priorvar,
        fullcov,
        batchsize,
        temperature,
        loadinitstate,
        saveinitstate,
        initparams,
        initvar,
        initlikvar,
        learnrateinitialopt,
        stepsizeinitialopt,
        nitersinitialopt,
        learnrateinitialmcmc,
        stepsizeinitialmcmc,
        nitersinitialmcmc,
        injectnoisemcmc,
        averagegradmcmc,
        learnrate,
        stepsize,
        niters,
        injectnoise,
        averagegrad,
        niterspersync,
        nitersperdamp,
        niterssample,
        nitersstartaveraging,
        averagepredprobs,
        nsyncsperexaminemaster,
        nitersperexamineinitial,
        nitersperexamineworker,
        synchronizedamping,
        meanlimits,
        varlimits,
        minmass,
        logReg,
        default_plot,
        use_aws,
        regu_coef,
        regu_coef_initial,
        shiftstates,
        psnep)
    end

    # running PosteiorServer code.
    # assumes that data models are already set up
    function run(
      name::AbstractString,
      filename::AbstractString,
      prior::AbstractGaussian.NatParam,
      workerstate::WorkerState,
      samplerstate::SamplerState,
      masterid::Int64,
      avgprobmasterid::Nullable{Int64},
      snapper::Remote{Snapper},
      dmexaminer::Remote{AbstractDataModel},                            # data models
      dmtrainexaminer::Remote{AbstractDataModel},                       #
      dminitworker::Remote{AbstractDataModel},                          #
      dmworkers::Array{Remote{AbstractDataModel}},                      #
      dmworkerexaminers::Nullable{Array{Remote{AbstractDataModel},1}},  #
      testclasses::Array{Float64,1};
      specs = PosteriorServer.Specs(keyargs...),
      keyargs...)

      nparams = fetchnparams(dminitworker)
      nworkers = length(dmworkers)
      println("=== Specs ===")
      @show specs
      println("=== $name: Starting up, #params = $(nparams) ===")
      examiner = RemoteExaminer(dmexaminer,snapper = snapper)
      trainexaminer = RemoteExaminer(dmtrainexaminer, snapper = snapper, train = true)

      if !isempty(specs[:loadinitstate])
        println("=== $name: Loading learning state ===")
        d = JLD.load(specs[:loadinitstate])
        workerstate = d["workerstate"]
        samplerstate = d["samplerstate"]
      end

      # set up worker for initial optimisation or MCMC if specified
      if specs[:nitersinitialopt] + specs[:nitersinitialmcmc] > 0
        initworker = RemoteWorker("$(name)_initworker",dminitworker,
          workerstate,
          samplerstate,
          specs,
          condprior = prior,
          snapper = Nullable(snapper),
          examiner = Nullable(examiner),
          nitersperexamine = specs[:nitersperexamineinitial],
          nitersperdamp = specs[:nitersperdamp],
          synchronizedamping = false
          )
      end

      # initial optimisation phase if specified
      if specs[:nitersinitialopt] > 0
        println("=== $name: Initial OPT learning phase ===")
        setStepSize(samplerstate,specs[:stepsizeinitialopt])
        setInjectNoise(samplerstate,0.0)
        setAverageGrad(samplerstate,true)
        runfetch(initworker,
          specs[:nitersinitialopt],
          beta = specs[:beta],
          learnrate = specs[:learnrateinitialopt],
          niterssample = specs[:niterssample],
          meanlimits = specs[:meanlimits],
          varlimits = specs[:varlimits],
          posstep = specs[:posstep],
          negstep = specs[:negstep],
          minstep = specs[:minstep]
          )
        (workerstate, samplerstate) = fetchstate(initworker)
        println("=== $name: Initial OPT learning phase done! === ")
      end
      # initial MCMC phase if specified
      if specs[:nitersinitialmcmc] > 0
        println("=== $name: Initial MCMC learning phase ===")
        setStepSize(samplerstate,specs[:stepsizeinitialmcmc])
        setInjectNoise(samplerstate,specs[:injectnoisemcmc])
        setAverageGrad(samplerstate,specs[:averagegradmcmc])
        runfetch(initworker,
          specs[:nitersinitialmcmc],
          beta = specs[:beta],
          learnrate = specs[:learnrateinitialmcmc],
          niterssample = specs[:niterssample],
          meanlimits = specs[:meanlimits],
          varlimits = specs[:varlimits],
          posstep = specs[:posstep],
          negstep = specs[:negstep],
          minstep = specs[:minstep]
          )
        (workerstate, samplerstate) = fetchstate(initworker)
        println("=== $name: Initial MCMC learning phase done! === ")
      end

      if specs[:nitersinitialopt] + specs[:nitersinitialmcmc] > 0
        shutdown(initworker)
      end

      if !isempty(specs[:saveinitstate])
        println("=== $name: Saving learning state ===")
        JLD.save(specs[:saveinitstate],
          "workerstate",workerstate,
          "samplerstate",samplerstate)
      end

      # Distributed phase of learning
      if specs[:niters] == 0
        state = getState(samplerstate)
        snapshots = fetchsnapshots(snapper)
        results = @dict(state,snapshots)
      else
          # set up master
        master = Nullable(Remote{Master}(@spawnat masterid begin
          Master("$(name)_master",prior,nworkers,snapper,examiner,trainexaminer,specs,
            nsyncsperexamine = specs[:nsyncsperexaminemaster],
            nitersstartaveraging = specs[:nitersstartaveraging])
        end))
        # set up server for averaging class probabilities
        if specs[:averagepredprobs]
            avgprobmasterid = get(avgprobmasterid)
            avgprobmaster = Nullable(Remote{PredProbMaster}(@spawnat avgprobmasterid begin
                try
                    PredProbMaster("$(name)predprobmaster",snapper,testclasses,
                    get(master),specs[:nitersstartaveraging])
                catch e
                    snap(snapper,:error,e)
                end
            end))
        else
            avgprobmaster = Nullable{Remote{PredProbMaster}}()
        end

        setStepSize(samplerstate,specs[:stepsize])
        setInjectNoise(samplerstate,specs[:injectnoise])
        setAverageGrad(samplerstate,specs[:averagegrad])
        workerstate.likapprox /= nworkers ## all data on initworker

        result = cell(nworkers)
        for worker = 1:nworkers
          workerid = where(dmworkers[worker])

          println("=== $name: Spawning worker $worker ===")
          result[worker] = @spawnat workerid begin
            dmworker = fetch(dmworkers[worker])::AbstractDataModel
            if specs[:averagepredprobs]
                workerexaminer = Nullable(RemoteWorkerExaminer(get(dmworkerexaminers)[worker]))
            else
                workerexaminer = Nullable{Remote{WorkerExaminer}}()
            end
            # set up worker on process workerid
            worker = Worker("$(name)_worker$worker",dmworker,
              workerstate,
              samplerstate,
              specs,
              master = master,
              avgprobmaster = avgprobmaster,
              snapper = snapper,
              workerexaminer = workerexaminer,
              niterspersync = specs[:niterspersync],
              nitersperdamp = specs[:nitersperdamp],
              synchronizedamping = specs[:synchronizedamping]
              )
              # run worker code
            run(worker,
              specs[:niters],
              beta = specs[:beta],
              learnrate = specs[:learnrate],
              niterssample = specs[:niterssample],
              meanlimits = specs[:meanlimits],
              varlimits = specs[:varlimits]
              )
            (workerstate,samplerstate) = fetchstate(worker)
          end # spawnat
        end #for worker

        workerstates = Array(WorkerState,nworkers)
        samplerstates = Array(SamplerState,nworkers)
        @sync begin
          numdone = 0
          incdone() = numdone += 1
          for id = 1:nworkers
            @async begin
              (workerstates[id],samplerstates[id]) = fetch(result[id])
              println("=== $name: Completed worker $id ($(incdone())/$nworkers done) ===")
              sleep(.1)
            end
          end
        end
        # collect posterior, snapshots and results
        posterior = fetchnatposterior(get(master))
        snapshots = fetchsnapshots(snapper)
        results = @dict(posterior,workerstates,samplerstates,snapshots)
        shutdown(get(master))
      end

      shutdown(examiner)
      shutdown(trainexaminer)

      println("=== $name: Done ===")

      return results
    end

end
