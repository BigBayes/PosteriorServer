# User experiment setting variables
method     = "logistic_regression"	# sgd distbayes sgld
dataset    = "" # mnist omniglot cifar-100
run_suffix = "sms"

# Quick way of setting up essential parameters
_NWK        = 1        # Number of workers
_NOBS       = 500      # Number of observations (synthetic)
_DIM        = 2        # Number of dimensions
_FULLCOV    = true     # Use full (true) or diagonal covariance (false)
_STEPSIZE   = 1E-2
_LEARNRATE  = 1E-1
_NIS        = 10       # Number of samples per steps
_NITERS     = 50       # Number of iterations
_ALG        = "SMS"   # Method to use (SNEP/SMS)
_SAMPLER    = "Adam"   # Sampler to use

# DEFAULTS ----------------------
using PosteriorServer
specs = PosteriorServer.Specs(
  nworkers = 8,
  beta     = 1.0,
  priorvar = 100.,

  fullcov    = false,
  sampler    = :Adam,
  initparams = :Gaussian, #Xavier
  initvar    = 1.0,

  learnrateinitialopt = 1e-2,
  stepsizeinitialopt  = 1e-2,
  nitersinitialopt    = 0,

  learnrateinitialmcmc = 1e-2,
  stepsizeinitialmcmc  = 1e-2,
  nitersinitialmcmc    = 0,
  injectnoisemcmc      = 0.1,
  averagegradmcmc      = true,

  learnrate     = 1e-2,
  stepsize      = 1e-2,
  niters        = 600*15,
  injectnoise   = 1.0,
  averagegrad   = false,

  nitersstartaveraging = 100,
  niterspersync        = 10,
  nitersperdamp        = 10,
  niterssample         = 1,
  synchronizedamping   = false,

  nsyncsperexaminemaster  = 10,
  nitersperexamineinitial = 10,
  nitersperexamineworker  = 0,
  loadinitstate           = "",
  saveinitstate           = "",

  meanlimits = (-10.0,10.0),
  varlimits  = (1e-2,1e2),
  minmass    = 1e-10,
  default_plot = false,
  use_aws = false,
  regu_coef = 0.,
  regu_coef_initial = 0.
)
# MODIFICATIONS -------------------
specs[:model]       = "logreg"
specs[:nworkers]    = 2
specs[:initparams]  = :Xavier
specs[:initlikvar]  = 0.1
specs[:priorvar]    = 100.
specs[:sampler]     = :Adam

specs[:batchsize]   = 100
specs[:averagepredprobs] = false

specs[:nitersinitialopt]    = 0
specs[:learnrateinitialopt] = 1e-2
specs[:stepsizeinitialopt]  = 1e-3

specs[:nitersinitial]       = 0
specs[:learnrateinitial]    = 1e-2
specs[:stepsizeinitial]     = 1e-3
specs[:injectnoise]         = 0.

specs[:nitersinitialmcmc]    = 0
specs[:learnrateinitialmcmc] = 1e-2
specs[:stepsizeinitialmcmc]  = 1e-3
specs[:injectnoisemcmc]      = 0.1
specs[:averagegradmcmc]      = false

specs[:niters]           = _NITERS
specs[:niterspersync]    = 1
specs[:learnrate]        = _LEARNRATE
specs[:stepsize]         = _STEPSIZE
specs[:injectnoise]      = 1.0
specs[:niterssample]     = _NIS
specs[:nitersperexamine] = 1

specs[:logReg] = true
specs[:algep]  = :sms

specs[:fullcov] = _FULLCOV

specs[:varlimits]             = (1e-2,1e8)
specs[:nitersstartaveraging]  = 60
specs[:synchronizedamping]    = false
specs[:adaptation]            = :none

nObs = _NOBS
dim  = _DIM

tmpf,tmpd  = "full","diag"
proj,noproj= "proj","noproj"
run_suffix = "smssgld_N$(nObs)_d$(dim)_$(specs[:fullcov]?tmpf:tmpd)_$(specs[:learnrate])_nis$(specs[:niterssample])"

varyparam = :nworkers
varyvalues = _NWK

println("...........................................")
println(run_suffix)
println("...........................................")

include("logreg_dataGeneration.jl")
logReg_datX = X
logReg_datY = y

println("...........................................")
println("Synthetic data generated ($nObs x $dim)")
println("...........................................")
