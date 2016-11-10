module SGMCMC

### This module implements a number of different MCMC and SGMCMC algorithms.
### The state of a sampler includes the actual state of the Markov chain as well
### as any adaptive parameters and hyperparameters.

abstract SamplerState

export SamplerState, AdamState, ConstantState, HMCState
export sample!, setStepSize, setInjectNoise, setAverageGrad
export getState, setState

type AdamState <: SamplerState
  x::Array{Float64}
  t::Int
  mt::Array{Float64}
  vt::Array{Float64}

  stepsize::Function
  beta1::Float64
  beta2::Float64
  minmass::Float64
  averagegrad::Bool
  injectnoise::Float64


  function AdamState(x::Array{Float64};
    stepsize::Function = niters::Int -> .001,beta1::Float64=.9,beta2::Float64=.999,minmass::Float64=1e-8,
    averagegrad::Bool=true,injectnoise::Float64=1.0)
    nparams = size(x)
    new(x,0,zeros(nparams),zeros(nparams),
    stepsize,beta1,beta2,minmass,
    averagegrad,injectnoise
    )
  end

  function AdamState(x::Array{Float64};
    stepsize::Float64=0.001,beta1::Float64=.9,beta2::Float64=.999,minmass::Float64=1e-8,
    averagegrad::Bool=true,injectnoise::Float64=1.0)
    nparams = size(x)
    stepsize = niters::Int -> stepsize
    new(x,0,zeros(nparams),zeros(nparams),
    stepsize,beta1,beta2,minmass,
    averagegrad,injectnoise
    )
  end
end


function sample!(s::AdamState, grad::Function)
  # stochastic gradient langevin dynamics
  # adapt mass and average gradient according to Adam

  gt = grad(s.x)
  s.t += 1
  s.mt = s.beta1*s.mt + (1-s.beta1)*gt
  s.vt = s.beta2*s.vt + (1-s.beta2)*(gt.*gt)
  mt = s.mt/(1.0-s.beta1^s.t) # debiased
  vt = s.vt/(1.0-s.beta2^s.t) # debiased
  alphat = s.stepsize(s.t)./(sqrt(vt)+s.minmass)
  s.x[:] += alphat .* (s.averagegrad ? mt : gt)
  if s.injectnoise>0.0 # cf experimental section
    noisestep = min(s.stepsize(s.t)*s.injectnoise,sqrt(2.0*alphat)).*randn(size(s.x))
    s.x[:] += noisestep
  end
  s
end

### SGD ###

type ConstantState <: SamplerState
  x::Array{Float64}
  stepsize::Float64
  t::Int

  function ConstantState(x::Array{Float64};
    stepsize::Float64 = .001)
    new(x, stepsize, 0)
  end

end

function sample!(s::ConstantState, grad::Function)
  g = grad(s.x)
  s.t += 1

  step = s.stepsize .* g
  s.x[:] +=  step
end

##### HMC sampler #####

type HMCState <: SamplerState
    x::Array{Float64}   # sampler state
    p::Array{Float64}   # sampler momentum
    stepsize::Float64   # stepsize
    niters::Int64       # number of leapfrog steps per MH step
    mass                # mass matrix
    function HMCState(x::Array{Float64};p=randn(length(x)),stepsize=0.001,niters=10,mass=1.0)
        if isa(mass,Number)
          mass = mass * ones(length(x))
        end
        new(x,p,stepsize,niters,mass)
    end
end

function sample!(s::HMCState,llik,grad)
  nparams = length(s.x)
  mass = s.mass
  stepsize = s.stepsize
  niters = s.niters
  s.p = sqrt(mass).*randn(nparams)
  curx = s.x
  curp = s.p
  s.p += .5*stepsize * grad(s.x)
  for iter = 1:niters
    s.x += stepsize * s.p./mass
    s.p += (iter<niters ? stepsize : .5*stepsize) * grad(s.x) # two leapfrog steps rolled in one unless at the end.
  end

  #calculate MH acceptance ratio in log space.
  logaccratio = llik(s.x) - llik(curx) -.5*sum((s.p.*s.p - curp.*curp)./mass)[1]

  if 0.0 > logaccratio[1] - log(rand())
      #reject
      s.x = curx
      s.p = curp
  else
      #accept
      #negate momentum for symmetric Metropolis proposal
      s.p = -s.p
      accrej = 1.
  end
  return s
end



# util functions
function Base.show(io::IO, state::AdamState)
  print(io,"AdamState($(length(state.x)),$(state.t),$(state.stepsize),$(state.beta1),$(state.beta2),$(state.averagegrad?"averagegrad":"currentgrad"),$(state.injectnoise > 0.0?:"injectnoise":"nonoise"))")
end

function getState(s::SamplerState)
  s.x
end
function setState(s::SamplerState,x)
  s.x = x
end

function setStepSize(s::SamplerState,stepsize)
  s.stepsize = stepsize
end

# Adam State uses a stepsize that depends on the iteration.
function setStepSize(s::AdamState,stepsize::Function)
  s.stepsize = stepsize
end
function setStepSize(s::AdamState,stepsize::Float64)
  s.stepsize = x::Int -> stepsize
end

function setInjectNoise(s::SamplerState,injectnoise::Float64)
end
function setInjectNoise(s::SamplerState,injectnoise::Bool)
end


# injectnoise control whether and how much noise is added cf appendix
function setInjectNoise(s::AdamState,injectnoise::Float64)
  s.injectnoise = injectnoise
end
function setInjectNoise(s::AdamState,injectnoise::Bool)
  s.injectnoise = injectnoise ? Inf : 0.0
end



function setAverageGrad(s::SamplerState,averagegrad::Bool)
end
# set momentum switch for Adam.
function setAverageGrad(s::AdamState,averagegrad::Bool)
  s.averagegrad = averagegrad
end

end
