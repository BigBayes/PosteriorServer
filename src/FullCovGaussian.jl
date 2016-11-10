module FullCovGaussian

### Implements diagonal covariance Gaussian distributions

# import type hierarchy
using ExpFamily
using AbstractGaussian

import Base.+, Base.-, Base.*, Base./, Base.==, Base.size, Base.zeros, Base.ones, Base.(.*),
       Base.mean, Base.var, Base.std, Base.axpy!


# Natural parameter
# parameterisation (precision*mu, -precision)
immutable NatParam <: AbstractGaussian.NatParam
  muPrec::Vector{Float64}
  negPrec::Matrix{Float64}

  function NatParam(muprec::Vector{Float64},negprec::Matrix{Float64})
    if any(negprec .> 0.0)
      #warn("Negative precision detected.")
    elseif any(isinf(negprec))
      #warn("Infinite precision detected.")
    elseif any(isnan(negprec)) | any(isnan(muprec))
      warn("NaN precision detected.")
    end
    if maximum(abs(negprec-negprec'))>1e-6
      warn("negprec not symmetric! -- regularizing")
      negprec = (negprec+negprec')/2
    end
    new(muprec,negprec)
  end
  function NatParam(;mean::Vector{Float64} = [0.0],cov::Matrix{Float64} = [1.0])
    precision = inv(cov)
    if maximum(abs(cov-cov'))>1e-6
      warn("cov not symmetric! -- regularizing")
      cov = (cov+cov')/2
    end
    new(precision*mean,-precision)
  end
  function NatParam(muPrec::Number,negPrec::Number)
    new([convert(Float64,muPrec)],[convert(Float64,negPrec)])
  end
end

# Mean parameter
# parameterisation ( E[X], 0.5*E[XX^T])
immutable MeanParam <: AbstractGaussian.MeanParam
  mu::Vector{Float64}
  mu2var::Matrix{Float64}

  function MeanParam(mu::Vector{Float64},mu2var::Matrix{Float64})
    if any(2.0*mu2var .< mu.*mu)
      #warn("Negative variance detected.")
    elseif any(isinf(mu)) | any(isinf(mu2var))
      #warn("Infinite variance detected.")
    elseif any(isnan(mu)) | any(isnan(mu2var))
      error("NaN variance detected.")
    end
    if maximum(abs(mu2var-mu2var'))>1e-6
      warn("mu2var not symmetric! -- regularizing")
      mu2var = (mu2var+mu2var')/2
    end
    new(mu,mu2var)
  end
  function MeanParam(;mean::Vector{Float64} = [0.0],cov::Matrix{Float64} = [1.0])
    if maximum(abs(cov-cov'))>1e-6
      warn("cov not symmetric! -- regularizing")
      cov = (cov+cov')/2
    end
    new(mean,.5*(mean*mean'+cov))
  end
  function MeanParam(mu::Number,mu2var::Number)
    new([convert(Float64,mu)],[convert(Float64,mu2var)])
  end
end


# conversion functions
function ExpFamily.meanparam(x::NatParam)
  mm = mean(x)
  vv = cov(x)
  MeanParam(mm, .5*(mm*mm'+vv))
end

function ExpFamily.natparam(x::MeanParam)
  vv = cov(x)
  ppmm = vv \ x.mu
  pp = inv(vv)
  NatParam(ppmm, -pp)
end

# sufficient statistics
function ExpFamily.suffstats(::Type{MeanParam},x::Vector{Float64})
  MeanParam(x,.5*(x*x'))
end
function ExpFamily.suffstats(::Type{NatParam},x::Vector{Float64})
  MeanParam(x,.5*(x*x'))
end


Base.size(x::NatParam) = size(x.muPrec)
Base.size(x::MeanParam) = size(x.mu)

Base.length(x::NatParam) = length(x.muPrec)
Base.length(x::MeanParam) = length(x.mu)

# NOTE: The zeros and ones refers to the variances. The means and off-diagonal covariances are zero
Base.zeros(::Type{MeanParam},n) = MeanParam(zeros(Float64,n),zeros(Float64,n,n))
Base.zeros(::Type{NatParam},n)  = NatParam(zeros(Float64,n),zeros(Float64,n,n))
Base.ones(::Type{MeanParam},n) = MeanParam(zeros(Float64,n),diagm(fill(0.5,n)))
Base.ones(::Type{NatParam},n)  = NatParam(zeros(Float64,n),diagm(fill(-1.0,n)))


# projection function to enforce minimum and maximum variance and means
function ExpFamily.project(x::MeanParam; minmu = -Inf,maxmu = Inf,minvar = 0,maxvar = Inf)
  mm = max(minmu,min(maxmu,mean(x)))
  (V,U) = eig(cov(x))
  V = max(minvar,min(maxvar,real(V)))
  U = real(U)
  MeanParam(mean = mm, cov = U*Diagonal(V)*U')
end
function ExpFamily.project(x::NatParam; minmu = -Inf,maxmu = Inf,minvar = 0,maxvar = Inf)
  (pp,U) = eig(-x.negPrec)
  U = real(U)
  pp = real(pp)
  mm = U*Diagonal(1.0./pp)*U'*x.muPrec
  pp = max(1.0./maxvar,min(1.0./minvar,pp))
  mm = max(minmu,min(maxmu,mm))
  prec = U*Diagonal(pp)*U'
  NatParam(prec*mm,-prec)
end

isvalid(x::NatParam) = isposdef(-x.negPrec)
isvalid(x::MeanParam) = isposdef(2.0*x.mu2var - x.mu*x.mu')


Base.mean(x::NatParam) = (-x.negPrec)\x.muPrec
Base.mean(x::MeanParam) = x.mu

Base.var(x::NatParam) = diag(cov(x))
Base.var(x::MeanParam) = diag(cov(x))
Base.std(x::NatParam) = sqrt(var(x))
Base.std(x::MeanParam) = sqrt(var(x))

Base.cov(x::NatParam) = -inv(x.negPrec)
Base.cov(x::MeanParam) = 2.0*x.mu2var - x.mu*x.mu'

# random number generation
ExpFamily.rand(x::NatParam) = mean(x) + chol(-x.negPrec)\randn(length(x))
ExpFamily.rand(x::MeanParam) = mean(x) + chol(cov(x))*randn(length(x))


# log density
function ExpFamily.loglikelihood(g::NatParam,x::Vector{Float64})
  const neghalflog2pi = -.5*log(2.0*pi)
  Pm = g.muPrec
  P = -g.negPrec
  C = chol(P)
  delta = C'\Pm
  sum(neghalflog2pi + log(diag(C))) -.5*(x'*P*x -2.0*x'*Pm + delta'*delta)
end

# log density
function ExpFamily.loglikelihood(g::MeanParam,x::Vector{Float64})
  const log2pi = log(2.0*pi)
  vv = cov(g)
  cc = chol(vv)
  mu = g.mu
  delta = cc'\(x-mu)
  .5*sum(-log2pi - 2. * log(diag(cc)) - delta'*delta)
end

# log density gradient
function AbstractGaussian.gradloglik(g::NatParam,x::Vector{Float64})
  g.muPrec + g.negPrec*x
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
