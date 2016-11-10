module LogRegDataModel

using Compat
using ExpFamily
using AbstractGaussian
using DataModel
using SGMCMC
using MLUtilities
using Utilities

typealias Float Float64

export LogRegSGMCMCDataModel, startup

# ==============================================================================
# Startup
# ==============================================================================
function startup(X,Y,
                examinerid,
                trainexaminerid,
                initworkerid,
                workerids,
                workerexaminerids;
                allworkeralldata=false)
    #
    ntrain   = length(Y)
    nworkers = length(workerids)
    #
    println("=== Set up master examiner ===")
    examiner = Remote{AbstractDataModel}(
                        @spawnat examinerid LogRegSGMCMCDataModel(X,Y) )
    #
    #
    println("=== Set up train examiner")
    trainexaminer = Remote{AbstractDataModel}(
                        @spawnat trainexaminerid LogRegSGMCMCDataModel(X,Y) )
    #
    println("=== Set up worker examiners ===")
    workerexaminers = Nullable{Array{Remote{AbstractDataModel},1}}()
    #
    assignments     = shuffle(rem(convert(Array,1:ntrain),nworkers)+1)
    workers         = Array(Remote{AbstractDataModel},nworkers)
    #
    println("=== Set up initial worker === ")
    initworker = Remote{AbstractDataModel}(
                        @spawnat initworkerid LogRegSGMCMCDataModel(
                                allworkeralldata ? X : X[assignments.==1,:],
                                allworkeralldata ? Y : Y[assignments.==1] ) )
    #
    println("=== Set up workers === ")
    for worker = 1:nworkers
        #
        workerid = workerids[worker]
        #
        wX = allworkeralldata ? X : X[assignments.==worker,:]
        wY = allworkeralldata ? Y : Y[assignments.==worker]
        #
        workers[worker] = Remote{AbstractDataModel}(
                                    @spawnat workerid LogRegSGMCMCDataModel(wX,wY) )
    end
    return @dict(examiner,trainexaminer,initworker,workers,workerexaminers)
 end
# ==============================================================================
# DATA MODELS
# ==============================================================================
# useful side functions
logit(z)        =  1./(1.+exp(-z))
logLogit(z)     = -log(1.+exp(-z))
gradLogLogit(z) =  1./(1.+exp( z))

abstract LogRegAbstractDataModel <: AbstractDataModel

DataModel.fetchnparams(dm::LogRegAbstractDataModel) = dm.D

type LogRegSGMCMCDataModel <: LogRegAbstractDataModel
    N::Int64
    D::Int64                            # dimension dim (covariates)
    X::Array{Float64,2}                 # NxD
    Y::Array{Float64,1}                 # Nx1 labels
    #
    function LogRegSGMCMCDataModel(X,Y)
        N,D = size(X)
        new(N,D,X,Y)
    end
end
#
function getloglik( dm::LogRegSGMCMCDataModel,
                    w::Vector{Float} )
    #
    sum([logLogit(dm.Y[i].*(dm.X[i,:]*w)) for i in 1:dm.N])
end
#
function DataModel.evaluate( dm::LogRegSGMCMCDataModel,
                             w::Vector{Float} )
    #
    predY  = logit(dm.X * w)
    # RMSE
    rmse   = sqrt( 1./dm.N * sum([(predY[i]-float(dm.Y[i]>0))^2 for i in 1:dm.N]) )
    loglik = getloglik(dm,w)
    # standardise names for dict
#    @show rmse
#    @show w
    accuracy      = rmse
    loglikelihood = loglik
    @dict(accuracy,loglikelihood)
end
#
# ==============
# SGLD SAMPLER
# ==============
function DataModel.sample!( dm::LogRegSGMCMCDataModel,
                            state::SamplerState,
                            prior::AbstractGaussian.NatParam,
                            beta::Float64)
        #
        function learngrad(w::Vector{Float})
            # local factors are: logit(y_i*<w,x_i>)
            # gradient: y_i*gradlogit(..)*x_i :
            getGLL(i) = dm.Y[i] .* gradLogLogit( dm.Y[i] .* (dm.X[i,:]*w) ) .* dm.X[i,:]'
            # need to sum that on all data points
            gll       = sum([getGLL(i) for i in 1:dm.N])
            # scale, add prior
            return  vec(gll) ./ beta + gradloglik(prior,w)
        end
        SGMCMC.sample!(state,learngrad)
end
# =============
# HMC SAMPLER
# =============
function DataModel.sample!( dm::LogRegSGMCMCDataModel,
                            state::HMCState,
                            prior::AbstractGaussian.NatParam,
                            beta::Float64)
        #
        function learngrad(w::Vector{Float})
            # local factors are: logit(y_i*<w,x_i>)
            # gradient: y_i*gradlogit(..)*x_i :
            getGLL(i) = dm.Y[i] .* gradLogLogit( dm.Y[i] .* (dm.X[i,:]*w) ) .* dm.X[i,:]'
            # need to sum that on all data points
            gll = sum([getGLL(i) for i in 1:dm.N])
            # scale, add prior
            return  vec(gll) ./ beta + gradloglik(prior,w)
        end
        function llik(w)
            getloglik(dm,w)+loglikelihood(prior,w)
        end
        SGMCMC.sample!(state,llik,learngrad)
end
#



# END OF MODULE
end
