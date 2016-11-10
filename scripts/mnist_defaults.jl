using PosteriorServer

specs = PosteriorServer.Specs(
  nworkers = 8,
  beta = 1.0,
  priorvar = 100.,

  fullcov = false,
  sampler = :Adam,
  initparams = :Gaussian, #Xavier
  initvar = 1.0,

  learnrateinitialopt = 1e-2,
  stepsizeinitialopt = 1e-2,
  nitersinitialopt = 0,

  learnrateinitialmcmc = 1e-2,
  stepsizeinitialmcmc = 1e-2,
  nitersinitialmcmc = 200,
  injectnoisemcmc = 0.1,
  averagegradmcmc = true,

  learnrate = 1e-2,
  stepsize = 1e-2,
  niters = 600*15,
  injectnoise = 1.0,
  averagegrad = false,

  nitersstartaveraging = 100,
  niterspersync = 10,
  nitersperdamp = 10,
  niterssample = 1,
  synchronizedamping = false,

  nsyncsperexaminemaster = 10,
  nitersperexamineinitial = 10,
  nitersperexamineworker = 0,
  loadinitstate = "",
  saveinitstate = "",

  meanlimits = (-10.0,10.0),
  varlimits = (1e-2,1e2),
  minmass = 1e-10,
  default_plot = false,
  use_aws = false,
  regu_coef = 0.,
  regu_coef_initial = 0.
)
specs[:commit] = ""
