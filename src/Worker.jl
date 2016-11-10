### Worker process
export Worker, WorkerState, fetchstate, run, runEP, workerstatus


const workerstatus = [:startup,:running,:stopped,:interrupt,:shutdown]

# constructor
function RemoteWorker(name::AbstractString,
                dmworker::Remote{AbstractDataModel},
                workerstate::WorkerState,
                samplerstate::SamplerState,
                specs::Dict; keyargs...
                )
  Remote{Worker}(@spawnat where(dmworker) Worker(name,fetch(dmworker),workerstate,samplerstate,specs;keyargs...))
end


# constructor
function Worker(name::AbstractString,
                dmworker::AbstractDataModel,
                workerstate::WorkerState,
                samplerstate::SamplerState,
                specs::Dict;
                condprior::AbstractGaussian.NatParam = 0.0*workerstate.likapprox,
                master::Nullable{Remote{Master}} = Nullable{Remote{Master}}(),
                avgprobmaster::Nullable{Remote{PredProbMaster}} = Nullable{Remote{PredProbMaster}}(),
                snapper = Nullable{Remote{Snapper}}(),
                examiner::Nullable{Remote{Examiner}} = Nullable{Remote{Examiner}}(),
                workerexaminer::Nullable{Remote{WorkerExaminer}} = Nullable{Remote{WorkerExaminer}}(),
                nitersperexamine::Int64 = 1,
                niterspersync::Int64 = 10,
                nitersperdamp::Int64 = 50,
                synchronizedamping::Bool = false
                )
  snap(snapper,:startup,name)
  # set up exam facilities
  examfac = isnull(examiner) ?
            Nullable{ExamFacility}() :
            Nullable(ExamFacility(get(examiner),nitersperexamine=nitersperexamine))
  workerexamfac = isnull(workerexaminer) ?
              Nullable{WorkerExamFacility}() :
              Nullable(WorkerExamFacility(get(workerexaminer),nitersperexamine=nitersperexamine))
  # initialise variables
  damper = condprior+workerstate.likapprox
  meanpost = meanparam(damper)

  # set up worker
  worker = Worker(name,
                    :startup,
                    0,
                    dmworker,
                    condprior,
                    meanpost, # tmp
                    damper, # tmp
                    workerstate,
                    samplerstate,
                    Nullable{Array{Float64}}(),
                    master,
                    avgprobmaster,
                    Nullable{SyncFacility}(),
                    Nullable{SyncFacility}(),
                    snapper,
                    examfac,
                    workerexamfac,
                    nitersperdamp,
                    specs)
  # register Worker with corresponding Master
  if !isnull(master)
    (natpost,meanpost,masterstatus) = register(get(master),Remote(worker),
      workerstate.likapprox,meanpost)
    worker.condprior = natpost - workerstate.likapprox
    worker.meanpost = meanpost
    worker.damper = natparam(meanpost)
    if masterstatus != :running
      worker.status = masterstatus
    end
    # set up synchronization facility
    worker.syncfac = Nullable(SyncDampFacility(get(master),
      workerstate.likapprox,
      worker.meanpost,
      specs,
      niterspersync = niterspersync,
      synchronizedamping = synchronizedamping))
  end

  if !isnull(avgprobmaster)
    worker.syncfacPredProb = Nullable(SyncPredProbFacility(get(avgprobmaster),Nullable{Array{Float64}}(),niterspersync = niterspersync))
  else
    worker.syncfacPredProb = Nullable{SyncPredProbFacility}()
  end

  worker
end

function fetchstate(w::Worker)
  (w.workerstate, w.samplerstate)
end

function fetchstate(worker::Remote{Worker})
  callfetch(fetchstate,worker)
end

function shutdown(worker::Remote{Worker}; status=:shutdown)
  call(shutdown,worker,status = status)
end
function shutdown(worker::Worker; status=:shutdown)
  worker.status = status
  snap(worker.snapper,status,worker.name)
  worker
end


function run(worker::Remote{Worker},args...; keyargs...)
  call(run,worker,args...;keyargs...)
end
function runfetch(worker::Remote{Worker},args...; keyargs...)
  callfetch(run,worker,args...;keyargs...)
end
function runEP(worker::Remote{Worker},args...; keyargs...)
  call(runEP,worker,args...;keyargs...)
end
function runEPfetch(worker::Remote{Worker},args...; keyargs...)
  callfetch(runEP,worker,args...;keyargs...)
end

# SNEP and SMS code
function run(worker::Worker,niters::Int;
             beta = 1.0,
             learnrate = 1e-3,
             niterssample = 1,
             meanlimits = (-10.0,10.0),
             varlimits = (1e-2,1e2),
             algep = :snep,
             doproject = true,
             showallcov = false)

  if !(worker.status in [:startup,:stopped])
    error("Worker status $(worker.status) not runnable.")
  end

  workerstate  = worker.workerstate
  samplerstate = worker.samplerstate

  snap(worker.snapper,:run,worker.name)
  worker.status = :running

  nparams  = fetchnparams(worker.dmworker)      # number of parameters
  likmean  = meanparam(workerstate.likapprox)   #

  # expected value of sufficient statistics
  stats = zeros(typeof(likmean),nparams)

  startiter = worker.iter
  # main learning loop
  for worker.iter = startiter+1:startiter+niters
    # LEARNER STATE: worker.condprior, worker.likapprox, worker.state, posterior
    natpost = worker.condprior + workerstate.likapprox

    if worker.status != :running
      snap(worker.snapper,:interrupt,worker.name,worker.iter,worker.status)
      return nothing
    end

    # synchronization
    if !isnull(worker.syncfac)
      syncfac = get(worker.syncfac)
      # checks whether there is return sync from master
      (newcondprior,newdamper,status) = request_results(syncfac)
      if status == :running
        syncfac.synctimes[end] = time() - syncfac.synctimes[end]
          # move samplerstate taking into account new conditional prior. cf v1.0 equation 46.
        snap(worker.snapper,:endSync,worker.name,worker.iter)
        #
        if worker.specs[:shiftstates]
            (curm,curv) = AbstractGaussian.meancov(natpost)
            relstate    = sqrtm(curv)\(getState(worker.samplerstate)-curm)
            #
            worker.condprior = newcondprior
            natpost          = newcondprior + workerstate.likapprox
            (newm,newv)      = AbstractGaussian.meancov(natpost)
            #
            setState(samplerstate,real(newm + sqrtm(newv)*relstate))
            #
        else
            worker.condprior = newcondprior
            natpost          = newcondprior + workerstate.likapprox
        end
        if newdamper != nothing
          worker.damper = natparam(newdamper)
        end

      elseif status != :waiting
        snap(worker.snapper,:interrupt,worker.name,worker.iter,status)
        worker.status = :interrupt
        return nothing
      end
      # request sync
      if syncfac.synchronizedamping
        request_sync(syncfac,
          workerstate.likapprox,
          worker.meanpost,
          worker.name,worker.iter)
      else
          request_sync(syncfac,
          workerstate.likapprox,
          worker.name,worker.iter)
      end
    end
    if isnull(worker.syncfac) || !get(worker.syncfac).synchronizedamping
      # damping posterior for convergence.
      if rem(worker.iter,worker.nitersperdamp)==0
        worker.damper = natpost
      end
    end

    if !isnull(worker.syncfacPredProb)
      syncfacPredProb = get(worker.syncfacPredProb)
      # checks whether there is return sync from master
      request_results(syncfacPredProb)

      request_sync(syncfacPredProb,
          worker.predprobs)
    end

    # -----------------------
    # core learning algorithm
    # -----------------------

    # update worker state
    worker.meanpost = meanparam(natpost)
    cavity          = worker.damper - workerstate.likapprox/beta
    #
    # compute noisy gradient (estimator of expected value of sufficient statistics)
    stats          *= 0.0
    for iter = 1:niterssample
      DataModel.sample!(worker.dmworker,samplerstate, cavity, beta)
      stats += suffstats(typeof(cavity),getState(worker.samplerstate))
    end
    if niterssample != 1
        stats /= niterssample
    end

    if algep==:snep
        #
        # compute step in Mean Parameter space
        step = stats - worker.meanpost

        # update
        #likmean = likmean + learnrate*step
        # inplace update
        lr = learnrate
        if isa(learnrate,Function)
            lr = learnrate(worker.iter)
        end
        Base.axpy!(lr,step,likmean)

        # projection step / inflation of variance in case of collapse
        if doproject
          likmean = project(likmean; minmu = meanlimits[1], maxmu = meanlimits[2],
                                     minvar = varlimits[1], maxvar = varlimits[2])
        end
        #
        # update corresponding natural parameters
        workerstate.likapprox = natparam(likmean)

        if showallcov
          @show stats.negPrec[:]'
          @show cov(likmean)[:]'
          @show cov(workerstate.likapprox)[:]'
        end
    #
    elseif algep==:sms
        statsNP   = natparam(stats)
        # c
        likapprox = (statsNP - cavity) * beta
        # damping
        likapprox = (1-learnrate) * workerstate.likapprox + learnrate * likapprox
        #
        if doproject
          workerstate.likapprox = project(likapprox,minmu=meanlimits[1],maxmu=meanlimits[2],
                                                    minvar=varlimits[1],maxvar=varlimits[2])
        else
          workerstate.likapprox = likapprox
        end
        if showallcov
          @show stats.negPrec[:]'
          @show cov(stats)[:]'
          @show cov(statsNP)[:]'
          @show cov(likapprox)[:]'
        end
    end
    # ----------------------------
    # done core learning algorithm
    # ----------------------------

    if !isnull(worker.examfac)
      if results_ready(get(worker.examfac))
        # get results of previous exam
        # but discard it, assume exam results saved by .
         fetch_results(get(worker.examfac))
      end
      # examine state
      request_exam(get(worker.examfac),worker.name,worker.iter,0.,
                   state=getState(worker.samplerstate),posterior=worker.meanpost)
    end


    if !isnull(worker.workerexamfac)
      if results_ready(get(worker.workerexamfac))
        # get results of previous exam
        # but discard it, assume exam results saved by .
        worker.predprobs =Nullable(fetch_results(get(worker.workerexamfac))[:state][:predictiveprobs])

        #@show [sum(v[:predictiveprobs],1) for (k,v) in get(worker.workerexamfac).results]
      end
      # examine state
      request_exam(get(worker.workerexamfac),worker.name,worker.iter,0.,
                   state=getState(worker.samplerstate),posterior=worker.meanpost)
    end

    # allow asynchronous tasks to run
    sleep(.00001)

  end # learning loop

  worker.status = :stopped
  if !isnull(worker.syncfac)
      snap(worker.snapper,:synctimes,worker.name,get(worker.syncfac).synctimes[1:(end-1)])
  end
  snap(worker.snapper,:stop,worker.name)
  nothing

end

function runEP(worker::Worker,niters::Int;
             beta = 1.0,
             niterspersync::Int = 1000,
             dampfactor::Float64 = .9,
             dampboundary::Float64 = .9,
             meanlimits = (-10.0,10.0),
             varlimits = (1e-2,1e2))

  if !(worker.status in [:startup,:stopped])
    error("Worker status not runnable.")
  end

  try

  workerstate = worker.workerstate
  samplerstate = worker.samplerstate

  if !isnull(worker.master)
    syncfac = SyncFacility(worker.master,workerstate.likapprox)
  else
    syncfac = Nullable{SyncFacility}()
  end

  snap(worker.snapper,:run,worker.name)
  worker.status = :running

  nparams = fetchnparams(worker.dmworker)
  posterior = worker.condprior + workerstate.likapprox
  cavity = posterior - workerstate.likapprox/beta
  stats = 0.0*meanparam(posterior)

  # main learning loop
  for iter = 1:niters
    worker.iter += 1

    if worker.status != :running
      snap(worker.snapper,:interrupt,worker.name,worker.iter,worker.status)
      return worker
    end

    # EP step and synchronize
    if rem(iter,niterspersync)==0
      tilted = natparam(stats/niterspersync)
      workerstate.likapprox = dampedUpdate(worker.likapprox,beta*(tilted-cavity),
                                       dampfactor=dampfactor, dampboundary=dampboundary)
      stats = 0.0*stats
      posterior = worker.condprior + workerstate.likapprox
      if !isnull(syncfac)
        # sync
        snap(worker.snapper,:startSync,worker.name,worker.iter)
        wait_sync(worker.syncfac,workerstate.likapprox,worker.name,worker.iter)
        (newcondprior,status) = wait_results(worker.syncfac)
        if status == :running
          snap(worker.snapper,:endSync,worker.name,worker.iter)
          (curm,curv) = AbstractGaussian.meancov(posterior)
          relstate = sqrtm(curv)\(getState(samplerstate)-curm)
          worker.condprior = newcondprior
          posterior = newcondprior + workerstate.likapprox
          (newm,newv) = AbstractGaussian.meanvar(posterior)
          setState(samplerstate,newm + sqrtm(newv)*relstate)
        elseif status != :none
          snap(worker.snapper,:interrupt,worker.name,worker.iter,status)
          worker.status = :interrupt
          return worker
        end
      end
      cavity = posterior - workerstate.likapprox/beta
    end

    sample!(worker.dmworker, samplerstate, cavity, beta)
    stats += suffstats(typeof(workerstate.liapprox),getState(samplerstate))

    if !isnull(worker.examfac)
      if results_ready(get(worker.examfac))
        # get results of previous exam
        # but discard it, assume exam results saved by .
        fetch_results(get(worker.examfac))
      end
      # examine state
      request_exam(get(worker.examfac),worker.name,worker.iter,
                   state=getState(samplerstate),posterior=posterior)
    end

  end # learning loop

  worker.status = :stopped
  snap(worker.snapper,:stop,worker.name)
  worker

  catch err
    @show err
    worker.status = :interrupt
    snap(worker.snapper,:interrupt,worker.name)
    shutdown(worker.master,Symbol(worker.name),err)
    rethrow(err)
  end

end


function dampedUpdate(likapprox,newapprox; dampfactor = .9, dampboundary = .9)
  ii = newapprox.negPrec .>= 0.0
  if !any(ii)
    step = dampfactor
  else
    step = dampboundary * minimum(-likapprox.negPrec[ii]./(newapprox.negPrec[ii]-likapprox.negPrec[ii]))
  end
  return (1.0-step)*likapprox + step*newapprox
end

function Base.show(io::IO,worker::Worker)
  print(io,"Worker($(worker.name),$(worker.status))")
end
function Base.show(io::IO,workerstate::WorkerState)
  print(io,"WorkerState($(length(workerstate.likapprox)))")
end
