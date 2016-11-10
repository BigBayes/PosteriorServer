module AbstractGaussian

### Implements abstract type for gaussian exponential families.


# import type hierarchy
using ExpFamily

abstract NatParam <: ExpFamily.NatParam
abstract MeanParam <: ExpFamily.MeanParam

export meancov, gradloglik

function meancov(x::NatParam)
  (mean(x),cov(x))
end
function meancov(x::MeanParam)
  (mean(x),cov(x))
end

function gradloglik(g::NatParam,x::Vector{Float64})
  error("Abstract function gradloglik not defined")
end

end
