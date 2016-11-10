
function SyncGradFacility(master::Remote{Master};
                          niterspersync = typemax(Int),
                          timepersync = Inf)

  bouncer = Bouncer(nitersperentry = niterspersync, timeperentry = timepersync)
  SyncGradFacility(bouncer,
                  false,
                  master,
                  RemoteRef())
end


function request_sync(fac::SyncGradFacility,
                      delta::Array{Float, 1},
                      args...)
  if request(fac.bouncer) && !fac.waiting
    do_sync(fac, delta, args...)
    return :running
  else
    return :waiting
  end
end

# wait until last sync done, do sync, and return result of last sync
function wait_sync(fac::SyncGradFacility,
                  delta::Array{Float, 1},
                  args...)
  if fac.waiting
    (params, status) = wait_results(fac)
  end
  do_sync(fac,likapprox,args...)
  (params, status)
end

# check if results is ready
function results_ready(fac::SyncGradFacility)
  isready(fac.results)
end

# wait to get results
function fetch_results(fac::SyncGradFacility)
  (params, status) = take!(fac.results)
  fac.waiting = false
  (params, status)
end

# if results are ready, return it, if not, return nothing
function request_results(fac::SyncGradFacility)
  if isready(fac.results)
    fetch_results(fac)
  else
    (nothing, :waiting)
  end
end

# internal function, do not use
function do_sync(fac::SyncGradFacility,
                grad::Array{Float, 1},
                args...)
  fac.waiting = true

  new_grad = copy(grad)
  #grad = zeros(length(grad))

  @schedule begin
    # NOTE: synchronizefetch is in DownpourMaster.jl
    # The following call is synchronous, i.e., doesn't return until finished
    put!(fac.results, synchronizefetch(fac.master, new_grad, args...))
  end
  reset(fac.bouncer)
end

function Base.show(io::IO, fac::SyncGradFacility)
  print(io,"SyncDampFacility($(fac.waiting?"waiting":"available"))")
end