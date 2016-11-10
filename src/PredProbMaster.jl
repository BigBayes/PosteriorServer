export PredProbMaster, shutdown, time

### implements server to average class probabilities from multiple workers

using Compat

# constructor
function PredProbMaster(name::AbstractString,
                snapper::Remote{Snapper},
                testclasses::Array{Float64,1},master::Remote{Master},nsyncsstartaveraging::Int64)
  status = :startup
  workerpredprobs = Array(Array{Float64,2},0)
  avgpredprobs = zeros(1)
  dimensions = size(avgpredprobs)
  predprobmaster = PredProbMaster(name,status,workerpredprobs,0,testclasses,master,dimensions,avgpredprobs,nsyncsstartaveraging,:none,snapper)
  snap(snapper,:startup,name)
  predprobmaster
end


function shutdown(master::Remote{PredProbMaster},args...; keyargs...)
  callfetch(shutdown,master,args...; keyargs...)
end


function shutdown(master::PredProbMaster; which = :none, err = :none)
  master.status = err == :none ? :shutdown : :interrupt
  master.exception = err
  snap(master.snapper,master.status,master.name,master.exception)
  master
end

function interrupt(master::Remote{PredProbMaster})
  call(interrupt,master)
end
function interrupt(master::PredProbMaster)
  shutdown(master,:user,InterruptException())
end


function waitToStart(master::PredProbMaster)
    master.status = :running
    snap(master.snapper,:run,master.name)
end


function fetchstatus(master::PredProbMaster,snapdata...)
  snap(master.snapper,:fetchstatus,snapdata...)
  return  master.status
end

function synchronizefetch(master::Remote{PredProbMaster},pred_probs::Nullable{Array{Float64}},snapdata...)
    callfetch(synchronize,master,pred_probs,snapdata...)
end

# evaluate averaged class probabilities
function evaluate(dms::PredProbMaster
  )
  predprobs = dms.avgpredprobs
  prediction =  (findmax(predprobs,1)[2]-1.) % size(predprobs)[1]
  accuracy = sum(dms.testclasses .== prediction[:])./(length(dms.testclasses)+0.0)
  loglikelihood = sum(log(max(predprobs[[convert(Int,c) for c in dms.testclasses]+1+(vec(1:size(predprobs)[2])-1).*size(predprobs)[1]],1e-20)))
  @dict(accuracy,loglikelihood)
end


function synchronize(master::PredProbMaster,pred_probs::Nullable{Array{Float64}},snapdata...)
    if !isnull(pred_probs)
        push!(master.workerpredprobs,get(pred_probs))
        master.dimensions = size(get(pred_probs))
    end
    iter=  fetchmeaniter(master.master)
    # averaging
    if iter <= master.nitersstartaveraging
        if !isempty(master.workerpredprobs)
            master.avgpredprobs = master.workerpredprobs[end]
            master.workerpredprobs = Array(Array{Float64,2},0)
        end
    elseif !isempty(master.workerpredprobs)
        snap(master.snapper,:startExamPredProb,iter,snapdata...)
        if size(master.avgpredprobs) != master.dimensions
            master.avgpredprobs = pop!(master.workerpredprobs)
            master.numpredprobs += 1
        end
        while length(master.workerpredprobs)>0
            predprobs = pop!(master.workerpredprobs)
            epspredprob = 1.0/max(1.0,master.numpredprobs+1)
            master.avgpredprobs = master.numpredprobs*epspredprob.*master.avgpredprobs + epspredprob.*predprobs
            master.numpredprobs += 1
        end
        results = evaluate(master)
        snap(master.snapper,:endExamPredProb,results)
    end



end

function Base.show(io::IO,master::PredProbMaster)
  print(io,"PredProbMaster($(master.name),$(master.status))")
end
