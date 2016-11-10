module DiagGaussian

### Implements diagonal covariance Gaussian distributions

# import type hierarchy
using ExpFamily
using AbstractGaussian

import Base.+, Base.-, Base.*, Base./, Base.==, Base.size, Base.zeros, Base.ones, Base.(.*),
       Base.mean, Base.var, Base.std, Base.axpy!



# Natural parameter
# parameterisation (mu/sigma^2, -1/sigma^2)
immutable NatParam <: AbstractGaussian.NatParam
  muPrec::Array{Float64}
  negPrec::Array{Float64}

  function NatParam(muprec::Array{Float64},negprec::Array{Float64})
    if any(negprec .> 0.0)
      #warn("Negative precision detected.")
    elseif any(isinf(negprec))
      #warn("Infinite precision detected.")
    elseif any(isnan(negprec)) | any(isnan(muprec))
      warn("NaN precision detected.")
    end
    new(muprec,negprec)
  end
  function NatParam(;mean::Array{Float64} = [0.0],var::Array{Float64} = [1.0])
    new(mean./var,-1.0./var)
  end
  function NatParam(muPrec::Number,negPrec::Number)
    new([convert(Float64,muPrec)],[convert(Float64,negPrec)])
  end
end

# mean parameter
# parameterisation (mu, 0.5*(mu^2 + sigma^2) )
immutable MeanParam <: AbstractGaussian.MeanParam
  mu::Array{Float64}
  mu2var::Array{Float64}

  function MeanParam(mu::Array{Float64},mu2var::Array{Float64})
    if any(2.0*mu2var .< mu.*mu)
      #warn("Negative variance detected.")
    elseif any(isinf(mu)) | any(isinf(mu2var))
      #warn("Infinite variance detected.")
    elseif any(isnan(mu)) | any(isnan(mu2var))
      error("NaN variance detected.")
    end
    new(mu,mu2var)
  end
  function MeanParam(;mean::Array{Float64} = [0.0],var::Array{Float64} = [1.0])
    new(mean,.5*(mean.*mean+var))
  end
  function MeanParam(mu::Number,mu2var::Number)
    new([convert(Float64,mu)],[convert(Float64,mu2var)])
  end
end

### conversion functions
function ExpFamily.meanparam(x::NatParam)
  mm = mean(x)
  vv = var(x)
  MeanParam(mm,.5*(mm.*mm+vv))
end
function ExpFamily.natparam(x::MeanParam)
  pp = 1.0./var(x)
  mm = mean(x)
  NatParam(mm.*pp, -pp)
end



### sufficient statistics
function ExpFamily.suffstats(::Type{MeanParam},x::Array{Float64})
  MeanParam(x,.5*(x.*x))
end
function ExpFamily.suffstats(::Type{NatParam},x::Array{Float64})
  MeanParam(x,.5*(x.*x))
end


Base.size(x::NatParam) = size(x.muPrec)
Base.size(x::MeanParam) = size(x.mu)

Base.length(x::NatParam) = length(x.muPrec)
Base.length(x::MeanParam) = length(x.mu)

Base.zeros(::Type{MeanParam},dims...) = MeanParam(zeros(Float64,dims),zeros(Float64,dims))
Base.zeros(::Type{NatParam},dims...)  = NatParam(zeros(Float64,dims),zeros(Float64,dims))
Base.ones(::Type{MeanParam},dims...) = MeanParam(zeros(Float64,dims),fill(0.5,dims))
Base.ones(::Type{NatParam},dims...)  = NatParam(zeros(Float64,dims),fill(-1.0,dims))


# projection function to enforce minimum and maximum variance and means
function ExpFamily.project(x::MeanParam; minmu = -Inf,maxmu = Inf,minvar = 0,maxvar = Inf)
  mm = max(minmu,min(maxmu,mean(x)))
  vv = max(minvar,min(maxvar,var(x)))
  MeanParam(mean = mm, var = vv)
end
function ExpFamily.project(x::NatParam; minmu = -Inf,maxmu = Inf,minvar = 0,maxvar = Inf)
  pp = max(1.0./maxvar,min(1.0./minvar,-x.negPrec))
  mp = max(minmu,min(maxmu,mean(x))).*pp
  NatParam(mp,-pp)
end

isvalid(x::NatParam) = x.negPrec .<= 0.0
isvalid(x::MeanParam) = 2.0*x.mu2var .>= x.mu.*x.mu


Base.mean(x::NatParam) = - x.muPrec./x.negPrec
Base.mean(x::MeanParam) = x.mu

Base.var(x::NatParam) = -1.0./x.negPrec
Base.var(x::MeanParam) = 2.0*x.mu2var - x.mu.*x.mu
Base.std(x::NatParam) = sqrt(var(x))
Base.std(x::MeanParam) = sqrt(var(x))

Base.cov(x::NatParam) = Diagonal(var(x))
Base.cov(x::MeanParam) = Diagonal(var(x))

# random samplers
ExpFamily.rand(x::NatParam) = mean(x) + std(x).*Base.randn(size(x.muPrec))
ExpFamily.rand(x::MeanParam) = mean(x) + std(x).*Base.randn(size(x.mu))

function ExpFamily.loglikelihood(g::NatParam,x::Array{Float64})
  const log2pi = log(2*pi)
  prec = -g.negPrec
  mu = g.muPrec./prec
  .5*sum(-log2pi + log(prec)) -.5*sum(prec.*(x-mu).*(x-mu))
end

function ExpFamily.loglikelihood(g::MeanParam,x::Array{Float64})
  const log2pi = log(2*pi)
  v = var(g)
  mu = g.mu
  -.5*sum(-log2pi - log(v) + (x-mu).*(x-mu)./v)
end


# log density gradient
function AbstractGaussian.gradloglik(g::NatParam,x::Vector{Float64})
  g.muPrec + g.negPrec.*x
end


+(x::MeanParam, y::MeanParam) = MeanParam(x.mu + y.mu, x.mu2var + y.mu2var)
-(x::MeanParam, y::MeanParam) = MeanParam(x.mu - y.mu, x.mu2var - y.mu2var)

+(x::NatParam, y::NatParam) = NatParam(x.muPrec + y.muPrec, x.negPrec + y.negPrec)
-(x::NatParam, y::NatParam) = NatParam(x.muPrec - y.muPrec, x.negPrec - y.negPrec)

function Base.axpy!(alpha::Float64,x::MeanParam,y::MeanParam)
    #in place addition.
    Base.axpy!(alpha,x.mu,y.mu)
    Base.axpy!(alpha,x.mu2var,y.mu2var)
end

function Base.axpy!(alpha::Float64,x::NatParam,y::NatParam)
    #in place addition.
    Base.axpy!(alpha,x.muPrec,y.muPrec)
    Base.axpy!(alpha,x.negPrec,y.negPrec)

end



*(x::Number, y::MeanParam) = MeanParam(x*y.mu, x*y.mu2var)
*(x::Number, y::NatParam)  = NatParam(x*y.muPrec, x*y.negPrec)

*(y::MeanParam, x::Number) = MeanParam(x*y.mu, x*y.mu2var)
*(y::NatParam, x::Number)  = NatParam(x*y.muPrec, x*y.negPrec)

.*(x::Array{Float64}, y::MeanParam) = MeanParam(x.*y.mu, x.*y.mu2var)
.*(x::Array{Float64}, y::NatParam)  = NatParam(x.*y.muPrec, x.*y.negPrec)

.*(y::MeanParam, x::Array{Float64}) = MeanParam(x.*y.mu, x.*y.mu2var)
.*(y::NatParam, x::Array{Float64})  = NatParam(x.*y.muPrec, x.*y.negPrec)


/(y::MeanParam, x::Number) = MeanParam(y.mu/x, y.mu2var/x)
/(y::NatParam, x::Number)  = NatParam(y.muPrec/x, y.negPrec/x)

==(x::MeanParam,y::MeanParam) = all((x.mu==y.mu) & (x.mu2var==y.mu2var))
==(x::NatParam, y::NatParam)  = all((x.muPrec==y.muPrec) & (x.negPrec==y.negPrec))

end
