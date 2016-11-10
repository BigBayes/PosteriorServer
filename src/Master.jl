export Master, shutdown, time, nworkers, masterstatus

### Master implements the PosteriorServer

# possible option for master status
const masterstatus = [:startup,:running,:interrupt,:shutdown]

# Constructor
function Master(name::AbstractString,
                prior::AbstractGaussian.NatParam,
                nworkers::Int,
                snapper::Remote{Snapper},
                examiner::Remote{Examiner},
                trainexaminer::Remote{Examiner},
                specs::Dict;
                nsyncsperexamine::Int = nworkers,
                nitersstartaveraging::Int = 1000)
  status = :startup
  # initalise master variables
  natposterior = deepcopy(prior)
  meanposterior = 0.0*meanparam(natposterior)
  avgnatposterior = natposterior
  avgparams = mean(natposterior)
  itersync = 0

  # array of workers linked to the master
  workers = Array(Remote,nworkers)

  # set up exam facilities
  examfac = ExamFacility(examiner,nitersperexamine = nsyncsperexamine)
  trainexamfac = ExamFacility(trainexaminer,nitersperexamine = nsyncsperexamine)
  exception = :none

  # construct master
  master = Master(name,status,natposterior,meanposterior,avgnatposterior,avgparams,itersync,workers,snapper,examiner,trainexaminer,examfac,trainexamfac,nitersstartaveraging,exception,Condition(),specs,Dict{AbstractString,Int}())

  # save snapshot
  snap(snapper,:startup,name)
  master
end

# shutdown
function shutdown(master::Remote{Master},args...; keyargs...)
  callfetch(shutdown,master,args...; keyargs...)
end


function shutdown(master::Master; which = :none, err = :none)
  master.status = err == :none ? :shutdown : :interrupt
  master.exception = err
  [shutdown(w,status = master.status) for w = master.workers]
  snap(master.snapper,master.status,master.name,which,string(err))
  master
end

# interrupt master
function interrupt(master::Remote{Master})
  call(interrupt,master)
end
function interrupt(master::Master)
  shutdown(master,:user,InterruptException())
end


# wait till all workers have registered.
function waitToStart(master::Master)
  if isdefined(master.workers,length(master.workers))
    master.status = :running
    snap(master.snapper,:run,master.name)
    notify(master.go,all=true)
  else
    wait(master.go)
  end
end

function register(master::Remote{Master},worker::Remote{Worker},
                        likapprox::AbstractGaussian.NatParam,
                        meanpostapprox::AbstractGaussian.MeanParam)
  callfetch(register,master,worker,likapprox,meanpostapprox)
end

function register(master::Master,worker::Remote{Worker},
                        likapprox::AbstractGaussian.NatParam,
                        meanpostapprox::AbstractGaussian.MeanParam)
  for i=1:length(master.workers)
    if !isdefined(master.workers,i)
      master.workers[i] = worker
      break
    end
  end

  master.natposterior += likapprox
  # wait for all other workers to start
  waitToStart(master)
  master.meanposterior = meanparam(master.natposterior)
  return (master.natposterior,master.meanposterior,master.status)
end

function fetchstatus(master::Master,snapdata...)
  snap(master.snapper,:fetchstatus,snapdata...)
  return (master.natposterior, master.status)
end

# return the number of synchronizations on the master
function fetchitersync(master::Remote{Master})
  callfetch(fetchitersync,master)
end
function fetchitersync(master::Master)
  return master.itersync
end

# return the mean number of iterations on the workers
function fetchmeaniter(master::Remote{Master})
  callfetch(fetchmeaniter,master)
end
function fetchmeaniter(master::Master)
    t = 0
    if !isempty(values(master.itersworkers))
      t = mean(values(master.itersworkers))
    end
    return t
end

# return current global approximation to the posterior
function fetchnatposterior(master::Remote{Master})
  callfetch(fetchnatposterior,master)
end
function fetchnatposterior(master::Master)
  return master.natposterior
end

# synchronize
function synchronizefetch(master::Remote{Master},
  likdelta::AbstractGaussian.NatParam,
  snapdata...)
  callfetch(synchronize,master,likdelta,snapdata...)
end
function synchronizefetch(master::Remote{Master},
  likdelta::AbstractGaussian.NatParam,
  meanpostdelta::AbstractGaussian.MeanParam,
  snapdata...)
  callfetch(synchronize,master,likdelta,meanpostdelta,snapdata...)
end

# synchronize
function synchronize(master::Master,
  likdelta::AbstractGaussian.NatParam,
  worker_name::AbstractString,
  worker_iter::Int,
  snapdata...)

  # update global posterior
  master.natposterior += likdelta

  # increase synchronization counter
  master.itersync += 1

  # Store the number of iterations this worker has done
  master.itersworkers[worker_name] = worker_iter

  # examination code
  snap(master.snapper,:startSync,snapdata...)
  # mean number of iterations on the workers
  niterseffective = mean(values(master.itersworkers))

  # average parameters
  eps = 1.0/max(1.0,niterseffective-master.nitersstartaveraging-1)
  master.avgparams = (1.0-eps)*master.avgparams + eps*mean(master.natposterior)
  master.avgnatposterior = (1.0-eps)*master.avgnatposterior + eps*master.natposterior

  # test error
  if results_ready(master.examfac)
    # get results of previous exam
    # but discard it, assume exam results saved by examiner.
    fetch_results(master.examfac)
  end
  # examine state
  request_exam(master.examfac,master.name,master.itersync,mean(values(master.itersworkers)),
               natposterior=master.natposterior,
               avgparams=master.avgparams,
               avgnatposterior=master.avgnatposterior)

    #train error
    if results_ready(master.trainexamfac)
     # get results of previous exam
     # but discard it, assume exam results saved by examiner.
     fetch_results(master.trainexamfac)
    end
    # examine only natpost to save time.
    request_exam(master.trainexamfac,master.name,master.itersync,mean(values(master.itersworkers)),
                natposterior=master.natposterior)

  return (master.natposterior, nothing, master.status)
end

function synchronize(master::Master,
  likdelta::AbstractGaussian.NatParam,
  meanpostdelta::AbstractGaussian.MeanParam,
  worker_name::AbstractString,
  worker_iter::Int,
  snapdata...)

  master.natposterior += likdelta
  master.meanposterior += meanpostdelta/nworkers(master)
  master.itersync += 1

  # Store the number of iterations this worker has done
  master.itersworkers[worker_name] = worker_iter

  # examination code
  snap(master.snapper,:startSync,snapdata...)

  niterseffective = int((master.specs[:niterspersync]+0.0)/master.specs[:nworkers]*master.itersync)

  eps = 1.0/max(1.0,niterseffective-master.nitersstartaveraging-1)

  master.avgparams = (1.0-eps)*master.avgparams + eps*mean(master.natposterior)
  master.avgnatposterior = (1.0-eps)*master.avgnatposterior + eps*master.natposterior

  # test error
  if results_ready(master.examfac)
    # get results of previous exam
    # but discard it, assume exam results saved by examiner.
    fetch_results(master.examfac)
  end
  # examine state
  request_exam(master.examfac,master.name,master.itersync,mean(values(master.itersworkers)),
               natposterior=master.natposterior,
               avgparams=master.avgparams,
               avgnatposterior=master.avgnatposterior)

    #train error.
    if results_ready(master.trainexamfac)
     # get results of previous exam
     # but discard it, assume exam results saved by examiner.
     fetch_results(master.trainexamfac)
    end
    # examine state
    request_exam(master.trainexamfac,master.name,master.itersync,mean(values(master.itersworkers)),
                natposterior=master.natposterior,
                avgparams=master.avgparams,
                avgnatposterior=master.avgnatposterior)


  return (master.natposterior, master.meanposterior, master.status)
end

function nworkers(master::Master)
  length(master.workers)
end

function Base.show(io::IO,master::Master)
  print(io,"Master($(master.name),$(master.status))")
end
