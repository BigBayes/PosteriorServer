using FullCovGaussian

### This is called from logreg_sms.jl / logreg_snep.jl
# -> it generates a dataset corresponding to the generative model
#    for the logistic regression presented in the paper

srand(1)
# GLOBAL SETTINGS
# > dimension is defined in script logreg_snep/sms
diagonal_covariance = false

# GENERATE DATA [following SMS setup]
# output: (X,Y) [(NxD), (N,1)], observations and labels
# .
# . covariance (size dxd)
P   = diagonal_covariance ? diagm(-1+2*rand(dim)):(-1+2*rand(dim,dim))
cov = P * P'
# . mean (size dx1), uniformly distributed on [0,1]
mu = rand(dim)
# . observations (size NxD)
X = (repmat(mu,1,nObs)+P*randn(dim,nObs))'
# . true parameters (size dx1)
priorCov = 10*eye(dim)
w        = chol(priorCov)'*randn(dim)
# . generate labels (size Nx1)
logit(z) = 1./(1.+exp(-z))
y01      = rand(nObs).<logit(X*w)
y        = y01*2.-1.

# natural parameters of the prior
priorNP  = FullCovGaussian.NatParam(mean=rand(dim),cov=priorCov)
