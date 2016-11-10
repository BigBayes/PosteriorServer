export Worker, fetchstate, run, runEP, workerstatus

const workerstatus = [:startup,:running,:stopped,:interrupt,:shutdown]

function RemoteWorker(name::AbstractString,
                      dmworker::Remote{AbstractDataModel},
                      samplerstate::SamplerState,
                      specs::Dict{Symbol,Any};
                      keyargs...
                      )
  Remote{Worker}(@spawnat where(dmworker) Worker(name, fetch(dmworker), samplerstate, specs; keyargs...))
end


function Worker(name::AbstractString,
                dmworker::AbstractDataModel,
                samplerstate::SamplerState,
                specs::Dict{Symbol,Any};
                master::Nullable{Remote{Master}} = Nullable{Remote{Master}}(),
                snapper = Nullable{Remote{Snapper}}(),
                examiner::Nullable{Remote{Examiner}} = Nullable{Remote{Examiner}}(),
                nitersperexamine::Int64 = 1,
                niterspersync::Int64 = 10
                )

  # Log worker startup and create examination facility
  snap(snapper,:startup,name)
  examfac = isnull(examiner) ?
            Nullable{ExamFacility}() :
            Nullable(ExamFacility(get(examiner),nitersperexamine=nitersperexamine))
  
  # Create the worker
  worker = Worker(name,
                  :startup,
                  0,
                  dmworker,
                  samplerstate,
                  master,
                  Nullable{SyncFacility}(),
                  snapper,
                  examfac,
                  specs)

  # Register the worker with the master and create the synchronization facility
  if !isnull(master)
    masterstatus = register(get(master), Remote(worker))
    
    if masterstatus != :running
      worker.status = masterstatus
    end

    worker.syncfac = Nullable(ElasticSyncFacility(get(master), niterspersync = niterspersync))
  end

  worker
end

function fetchstate(w::Worker)
  w.samplerstate
end

function fetchstate(worker::Remote{Worker})
  callfetch(fetchstate, worker)
end

function shutdown(worker::Remote{Worker}; status=:shutdown)
  call(shutdown, worker, status = status)
end
function shutdown(worker::Worker; status=:shutdown)
  worker.status = status
  snap(worker.snapper, status, worker.name)
  worker
end


function run(worker::Remote{Worker}, args...; keyargs...)
  call(run, worker, args...; keyargs...)
end

function runfetch(worker::Remote{Worker}, args...; keyargs...)
  callfetch(run, worker, args...; keyargs...)
end

function run(worker::Worker,
            niters::Int;
            regu_coef = 0.0005)

  if !(worker.status in [:startup,:stopped])
    error("Worker status $(worker.status) not runnable.")
  end

  snap(worker.snapper,:run,worker.name)
  worker.status = :running

  nparams = fetchnparams(worker.dmworker)

  startiter = worker.iter
  # main learning loop
  for worker.iter = startiter+1:startiter+niters

    if worker.status != :running
      snap(worker.snapper,:interrupt,worker.name,worker.iter,worker.status)
      return nothing
    end

    # core learning algorithm
    SGMCMC.sample!(worker.samplerstate, x -> DataModel.evaluateGrad(worker.dmworker, x, regu_coef = regu_coef))
    
    # done core learning algorithm

    if !isnull(worker.syncfac)
      syncfac = get(worker.syncfac)

      # Request sync with master (in distributed case)
      status = request_sync(syncfac, getState(worker.samplerstate), worker.name, worker.iter)
      
      (new_params, status) = request_results(syncfac)
      
      if status == :running
        #params = copy(new_params)
        setState(worker.samplerstate, (1 - worker.specs[:moving_rate]) * getState(worker.samplerstate) + worker.specs[:moving_rate] * new_params)

        snap(worker.snapper,:endSync,worker.name,worker.iter)
      elseif status != :waiting
        snap(worker.snapper,:interrupt,worker.name,worker.iter,status)
        worker.status = :interrupt
        return nothing
      end
    end

    if !isnull(worker.examfac)
      if results_ready(get(worker.examfac))
        # get results of previous exam
         fetch_results(get(worker.examfac))
      end
      # examine state
      request_exam(get(worker.examfac),
                  worker.name,
                  worker.iter,
                  0.,
                  params = getState(worker.samplerstate))
    end

    # allow asynchronous tasks to run
    sleep(.00001)
  end # learning loop

  worker.status = :stopped
  snap(worker.snapper, :stop, worker.name)
  nothing
end

function Base.show(io::IO,worker::Worker)
  print(io,"Worker($(worker.name),$(worker.status))")
end

function Base.show(io::IO, params::Array{Float})
  print(io,"Params($(length(params)))")
end