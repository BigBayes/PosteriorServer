module DataModel

### Implements an abstract data model for use by the workers.

using AbstractGaussian
using SGMCMC
using Utilities

export AbstractDataModel, sample!, evaluate, fetchparams, fetchnparams, shutdown, init_xavier, init_simple_fanin, init_gaussian, init_uniform

abstract AbstractDataModel

# parameters
function fetchparams(dm::Remote{AbstractDataModel})
  callfetch(fetchparams,dm)
end

# number of parameters
function fetchnparams(dm::Remote{AbstractDataModel})
  callfetch(fetchnparams,dm)
end

function init_gaussian(dm::Remote{AbstractDataModel}, initvar::Float64)
  callfetch(init_gaussian, dm, initvar)
end

function init_uniform(dm::Remote{AbstractDataModel}, initvar::Float64)
  callfetch(init_uniform, dm, initvar)
end

function init_xavier(dm::Remote{AbstractDataModel})
  callfetch(init_xavier, dm)
end

function init_simple_fanin(dm::Remote{AbstractDataModel})
  callfetch(init_simple_fanin, dm)
end

# parameters
function fetchparams(dm::AbstractDataModel)
  error("Function params not defined for abstract DataModel")
end
# number of parameters
function fetchnparams(dm::AbstractDataModel)
  error("Function nparams not defined for abstract DataModel")
end

function init_xavier(dm::AbstractDataModel)
  error("Function init_xavier not defined for abstract DataModel")
end

function init_simple_fanin(dm::AbstractDataModel)
  error("Function init_simple_fanin not defined for abstract DataModel")
end

function init_gaussian(dm::AbstractDataModel, initvar::Float64)
  error("Function init_gaussian not defined for abstract DataModel")
end

function init_uniform(dm::AbstractDataModel, initvar::Float64)
  error("Function init_uniform not defined for abstract DataModel")
end

function init_gaussian(dm::AbstractDataModel, initvar::Float32)
  error("Function init_gaussian not defined for abstract DataModel")
end

function init_uniform(dm::AbstractDataModel, initvar::Float32)
  error("Function init_uniform not defined for abstract DataModel")
end


# given params, does 1 step of MCMC sampling, returning params
function sample!(dm::AbstractDataModel,
  state::SamplerState,
  prior::AbstractGaussian.NatParam,
  beta::Float64)

  error("Function sample not defined for abstract DataModel")
end

function sample!(dm::AbstractDataModel,
  state::SamplerState,
  prior::AbstractGaussian.NatParam,
  beta::Float32)

  error("Function sample not defined for abstract DataModel")
end

# evaluate metrics
function evaluate(dm::AbstractDataModel,
  state::Array{Float64})

  error("Function evaluate not defined for abstract DataModel")
end

function evaluate(dm::AbstractDataModel,
  state::Array{Float32})

  error("Function evaluate not defined for abstract DataModel")
end


function evaluatePredProb(dm::AbstractDataModel,
  state::Array{Float64})

  error("Function evaluatePredProb not defined for abstract DataModel")
end

function evaluatePredProb(dm::AbstractDataModel,
  state::Array{Float32})

  error("Function examinePredProb not defined for abstract DataModel")
end

function evaluateGrad(dms::AbstractDataModel,
                                params::Vector{Float64})
  error("Function evaluateGrad not defined for abstract DataModel")
end

function evaluatePredProb(dm::AbstractDataModel,
    posterior::AbstractGaussian.NatParam)

    evaluatePredProb(dm,mean(posterior))
end



function evaluate(dm::AbstractDataModel,
  posterior::AbstractGaussian.NatParam)

  evaluate(dm,mean(posterior))
end

function evaluate(dm::AbstractDataModel; keyargs...)
  Dict([(k,examine(dm,v)) for (k,v) in keyargs])
end



function evaluatefetch(dm::Remote{AbstractDataModel},args...;keyargs...)
  callfetch(examine,dm,args...;keyargs...)
end

function shutdown(dms::Remote{AbstractDataModel})
  callfetch(shutdown,dms)
end

function shutdown(dms::AbstractDataModel)
end

end
