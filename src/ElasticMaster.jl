export Master, shutdown, time, nworkers, masterstatus

const masterstatus = [:startup, :running, :interrupt, :shutdown]

function Master(name::AbstractString,
                params::Array{Float},
                nworkers::Int,
                snapper::Remote{Snapper},
                examiner::Remote{Examiner},
                specs::Dict{Symbol,Any};
                nsyncsperexamine::Int = nworkers)
  status = :startup
  itersync = 0
  workers = Array(Remote, nworkers)
  examfac = ExamFacility(examiner, nitersperexamine = nsyncsperexamine)
  exception = :none
  master = Master(name, status, copy(params), itersync, workers, snapper, examiner, examfac, exception, Condition(), Dict{AbstractString,Int}(),specs)
  snap(snapper, :startup, name)
  master
end


function shutdown(master::Remote{Master},args...; keyargs...)
  callfetch(shutdown,master,args...; keyargs...)
end


function shutdown(master::Master; which = :none, err = :none)
  master.status = err == :none ? :shutdown : :interrupt
  master.exception = err
  [shutdown(w,status = master.status) for w = master.workers]
  snap(master.snapper, master.status, master.name, which, string(err))
  master
end

function interrupt(master::Remote{Master})
  call(interrupt,master)
end
function interrupt(master::Master)
  shutdown(master,:user,InterruptException())
end

function waitToStart(master::Master)
  if isdefined(master.workers,length(master.workers))
    master.status = :running
    snap(master.snapper,:run,master.name)
    notify(master.go,all=true)
  else
    wait(master.go)
  end
end

# Register worker with master via remote on master
function register(master::Remote{Master},
                  worker::Remote{Worker})
  callfetch(register, master, worker)
end

# Register worker with master
function register(master::Master,
                  worker::Remote{Worker})
  for i=1:length(master.workers)
    if !isdefined(master.workers,i)
      master.workers[i] = worker
      break
    end
  end

  waitToStart(master)
  return master.status
end

function fetchstatus(master::Master, snapdata...)
  snap(master.snapper, :fetchstatus, snapdata...)
  return master.status
end

function fetchitersync(master::Remote{Master})
  callfetch(fetchitersync, master)
end

function fetchitersync(master::Master)
  return master.itersync
end

function synchronizefetch(master::Remote{Master},
                          delta::Array{Float, 1},
                          snapdata...)
  callfetch(synchronize, master, delta, snapdata...)
end

function synchronize(master::Master,
                    delta::Array{Float, 1},
                    worker_name::AbstractString,
                    worker_iter::Int,
                    snapdata...)

  # Perform a single Adagrad/ADAM step
  master.params = (1. - master.specs[:moving_rate]) * master.params + master.specs[:moving_rate] * delta

  #master.params += delta
  master.itersync += 1

  # Store the number of iterations this worker has done
  master.itersworkers[worker_name] = worker_iter;

  # examination code
  snap(master.snapper, :startSync, snapdata...)

  if results_ready(master.examfac)
    # get results of previous exam
    fetch_results(master.examfac)
  end
  # examine state
  request_exam(master.examfac,
              master.name,
              master.itersync,
              mean(values(master.itersworkers)),
              params = master.params)

  return (master.params, master.status)
end

function nworkers(master::Master)
  length(master.workers)
end

function Base.show(io::IO,master::Master)
  print(io,"Master($(master.name),$(master.status))")
end