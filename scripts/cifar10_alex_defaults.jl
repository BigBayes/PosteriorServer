using PosteriorServer

specs = PosteriorServer.Specs(
  nworkers = 8,
  beta = 1.0,
  priorvar = 100.,

  fullcov = false,
  sampler = :Adam,
  initparams = :Xavier, #Gaussian
  initvar = 0.1,
  initlikvar = 1.0,

  learnrateinitialopt = 0.0,
  stepsizeinitialopt = 1e-5,
  nitersinitialopt = 500*2,

  learnrateinitialmcmc = 1e-5,
  stepsizeinitialmcmc = 1e-5,
  nitersinitialmcmc = 500*2,
  injectnoisemcmc = 0.1,
  averagegradmcmc = true,

  learnrate = 1e-5,
  stepsize = 1e-5,
  niters = 500*20,
  injectnoise = Inf,
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
  default_plot = true,
  use_aws = false
)
specs[:commit] =  readall(`git log --pretty=format:'%h' -n 1`)
