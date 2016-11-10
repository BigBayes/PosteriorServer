
export Remote, call, callfetch, localcall
export Bouncer, request, isready, reset, where

import Base.reset, Base.fetch

# A wrapper around RemoteRef that includes extra type-safety
immutable Remote{T}
  ref::RemoteRef
end

function Remote{T}(obj::T)
  ref = RemoteRef()
  put!(ref,obj)
  Remote{T}(ref)
end

function where{T}(remote::Remote{T})
  remote.ref.where
end

Base.fetch{T}(remote::Remote{T}) = fetch(remote.ref)

function localcall{T}(f::Function,remote::Remote{T},keyargs,args...)
  f(fetch(remote),args...;keyargs...)
end

# Calls a function locally or remotely depending on whether or not remote points to current worker
function call{T}(f::Function,remote::Remote{T},args...;keyargs...)
  if myid() == where(remote)
    # local call
    result = RemoteRef()
    put!(result,localcall(f,remote,keyargs,args...))
    result
  else
    # remotecall
    remotecall(where(remote),localcall,f,remote,keyargs,args...)
  end
end

# Calls a function locally or remotely, fetching if remotely, depending on whether or not remote points to current worker
function callfetch{T}(f::Function,remote::Remote{T},args...;keyargs...)
  if myid() == where(remote)
    # local call
    localcall(f,remote,keyargs,args...)
  else
    # remotecall
    remotecall_fetch(where(remote),localcall,f,remote,keyargs,args...)
  end
end


# returns true after at least a certain number of requests, or a certain
# amount of time.
type Bouncer
  curiter::Int                  # current iteration
  lastiter::Int                 # iteration of last entry
  lasttime::Float64             # time of last entry
  nitersperentry::Int           # number of iterations per entry
  timeperentry::Float64         # time per entry
end

function Bouncer(;nitersperentry = typemax(Int),timeperentry = Inf)
  if nitersperentry == typemax(Int) && timeperentry == Inf
    # by default, examine everytime pushed
    nitersperentry = 1
  end
  Bouncer(0,0,time(),nitersperentry,timeperentry)
end

# send request
function request(bouncer::Bouncer)
  bouncer.curiter += 1
  return isready(bouncer)
end

# returns true if request will return true.
function isready(bouncer::Bouncer)
  curtime = time()
  return bouncer.curiter+1 >= bouncer.lastiter + bouncer.nitersperentry ||
     curtime >= bouncer.lasttime + bouncer.timeperentry
end

# reset bouncers
function Base.reset(bouncer::Bouncer)
  bouncer.lastiter = bouncer.curiter
  bouncer.lasttime = time()
  nothing
end
