module ExpFamily

### Implements abstract exponential family types

export NatParam, MeanParam
export rand, loglikelihood, suffstats
export meanparam, natparam, project

import Base.+, Base.-, Base.*, Base./, Base.==, Base.(.*), Base.size, Base.zeros, Base.ones, Base.mean

abstract NatParam
abstract MeanParam


# sufficient statistics
function suffstats(::Type{ExpFamily.MeanParam},x)
  error("Abstract function suffstats not defined")
end
function suffstats(::Type{ExpFamily.NatParam},x)
  error("Abstract function suffstats not defined")
end

function Base.zeros(::Type{NatParam},dims...)
  error("Abstract function zeros not defined")
end
function Base.zeros(::Type{MeanParam},dims...)
  error("Abstract function zeros not defined")
end

# projection function
function project(x::NatParam; kargs...)
  error("Abstract function project not defined")
end
function project(x::MeanParam; kargs...)
  error("Abstract function project not defined")
end


# conversion functions
function meanparam(x::NatParam)
  error("Abstract function meanparam not defined")
end
function natparam(x::MeanParam)
  error("Abstract function natparam not defined")
end

# random number generation
function rand(x::NatParam)
  error("Abstract function rand not defined")
end
function rand(x::MeanParam)
  error("Abstract function rand not defined")
end

# log density
function loglikelihood(g::NatParam,x)
  error("Abstract function loglikelihood not defined")
end
function loglikelihood(g::MeanParam,x)
  error("Abstract function loglikelihood not defined")
end


# arithmetic 
function +(x::MeanParam, y::MeanParam)
  error("Abstract function + not defined")
end
function +(x::NatParam, y::NatParam)
  error("Abstract function + not defined")
end

function -(x::MeanParam, y::MeanParam)
  error("Abstract function - not defined")
end
function -(x::NatParam, y::NatParam)
  error("Abstract function - not defined")
end

function *(x::Number, y::MeanParam)
  error("Abstract function * not defined")
end
function *(x::Number, y::NatParam)
  error("Abstract function * not defined")
end

function *(x::MeanParam, y::Number)
  error("Abstract function * not defined")
end
function *(x::NatParam, y::Number)
  error("Abstract function * not defined")
end
function /(x::MeanParam, y::Number)
  error("Abstract function / not defined")
end
function /(x::NatParam, y::Number)
  error("Abstract function / not defined")
end

function .*(x::Number, y::MeanParam)
  error("Abstract function .* not defined")
end
function .*(x::Number, y::NatParam)
  error("Abstract function .* not defined")
end

function .*(x::MeanParam, y::Number)
  error("Abstract function .* not defined")
end
function .*(x::NatParam, y::Number)
  error("Abstract function .* not defined")
end

function ==(x::MeanParam, y::MeanParam)
  error("Abstract function == not defined")
end
function ==(x::NatParam, y::NatParam)
  error("Abstract function == not defined")
end


end
