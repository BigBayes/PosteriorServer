# facility to send class probabilities from Worker to PredProbMaster for averaging.

function SyncPredProbFacility(master::Remote{PredProbMaster},pred_probs::Nullable{Array{Float64}};
                      niterspersync = typemax(Int),
                      timepersync = Inf)
  bouncer = Bouncer(nitersperentry=niterspersync,timeperentry = timepersync)
  SyncPredProbFacility(bouncer,false,pred_probs,master,RemoteRef())
end


# check with bouncer and whether last sync done.  If ok, do sync.
function request_sync(fac::SyncPredProbFacility,pred_probs::Nullable{Array{Float64}},args...)
  if request(fac.bouncer) && !fac.waiting
    do_sync(fac,pred_probs,args...)
  end
end

# wait until last sync done, do sync, and return result of last sync
function wait_sync(fac::SyncPredProbFacility,pred_probs::Nullable{Array{Float64}},args...)
  if fac.waiting
    wait_results(fac)
  end
  do_sync(fac,pred_probs,args...)
end

# check if results is ready
function results_ready(fac::SyncPredProbFacility)
  isready(fac.results)
end

# wait to get results
function fetch_results(fac::SyncPredProbFacility)
  fac.waiting = false
end

# if results are ready, return it, if not, return nothing
function request_results(fac::SyncPredProbFacility)
  if isready(fac.results)
    fetch_results(fac)
  else
    (nothing,:waiting)
  end
end

# internal function, do not use
function do_sync(fac::SyncPredProbFacility,pred_probs::Nullable{Array{Float64}},args...)
  fac.waiting = true
  fac.lastpushed = pred_probs
  @schedule begin
    put!(fac.results,synchronizefetch(fac.master,pred_probs,args...))
  end
  reset(fac.bouncer)
end

function Base.show(io::IO,fac::SyncPredProbFacility)
  print(io,"SyncPredProbFacility($(fac.waiting?"waiting":"available"))")
end
