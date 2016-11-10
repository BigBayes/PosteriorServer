### SyncDampFacility handles the request for SNEP synchronizations.
function SyncDampFacility(master::Remote{Master},
                      likapprox::AbstractGaussian.NatParam,
                      meanpost::AbstractGaussian.MeanParam,
                      specs::Dict;
                      niterspersync = typemax(Int),
                      timepersync = Inf,
                      synchronizedamping = false)
  bouncer = Bouncer(nitersperentry=niterspersync,timeperentry = timepersync)
  SyncDampFacility(bouncer,
    false,synchronizedamping,
    likapprox,meanpost,master,RemoteRef(),specs,Float64[])
end

# request sync
function request_sync(fac::SyncDampFacility,
  likapprox::AbstractGaussian.NatParam,
  args...)
  if request(fac.bouncer) && !fac.waiting
    push!(fac.synctimes,time())
    do_sync(fac,likapprox,args...)
  end
end
function request_sync(fac::SyncDampFacility,
  likapprox::AbstractGaussian.NatParam,
  meanpost::AbstractGaussian.MeanParam,
  args...)
  if request(fac.bouncer) && !fac.waiting
    push!(fac.synctimes,time())
    do_sync(fac,likapprox,meanpost,args...)
  end
end

# wait until last sync done, do sync, and return result of last sync
function wait_sync(fac::SyncDampFacility,
  likapprox::AbstractGaussian.NatParam,
  args...)
  if fac.waiting
    (condprior,meanpost,status) = wait_results(fac)
  end
  do_sync(fac,likapprox,args...)
  (condprior,meanpost,status)
end
function wait_sync(fac::SyncDampFacility,
  likapprox::AbstractGaussian.NatParam,
  meanpost::AbstractGaussian.MeanParam,
  args...)
  if fac.waiting
    (condprior,meanpost,status) = wait_results(fac)
  end
  do_sync(fac,likapprox,meanpost,args...)
  (condprior,meanpost,status)
end

# check if results is ready
function results_ready(fac::SyncDampFacility)
  isready(fac.results)
end

# wait to get results
function fetch_results(fac::SyncDampFacility)
  (natpost,meanpost,status) = take!(fac.results)
  fac.waiting = false
  condprior = natpost - fac.lastpushedlikapprox
  (condprior,meanpost,status)
end

# if results are ready, return it, if not, return nothing
function request_results(fac::SyncDampFacility)
  if isready(fac.results)
    fetch_results(fac)
  else
    (nothing,nothing,:waiting)
  end
end

# internal function
function do_sync(fac::SyncDampFacility,
  likapprox::AbstractGaussian.NatParam,
  args...)
  fac.waiting = true
  likdelta = likapprox - fac.lastpushedlikapprox
  fac.lastpushedlikapprox = likapprox
  @schedule begin
    put!(fac.results,synchronizefetch(fac.master,likdelta,args...))
  end
  reset(fac.bouncer)
end

# internal function
function do_sync(fac::SyncDampFacility,
  likapprox::AbstractGaussian.NatParam,
  meanpost::AbstractGaussian.NatParam,
  args...)
  fac.waiting = true
  likdelta = likapprox - fac.lastpushedlikapprox
  meanpostdelta = meanpost - fac.lastpushedmeanpost
  fac.lastpushedlikapprox = likapprox
  fac.lastpushedmeanpost = meanpost
  @schedule begin
    put!(fac.results,synchronizefetch(fac.master,likdelta,meanpostdelta,args...))
  end
  reset(fac.bouncer)
end

function Base.show(io::IO,fac::SyncDampFacility)
  print(io,"SyncDampFacility($(fac.waiting?"waiting":"available"))")
end
