export WorkerExaminer, examine, examinefetch

### evaluates class probabilities

function WorkerExaminer(dm::AbstractDataModel; snapper = Nullable{Remote{Snapper}}())
  WorkerExaminer(:running,dm,snapper)
end

function WorkerExaminer(dm::AbstractDataModel; snapper = Nullable{Remote{Snapper}}())
  WorkerExaminer(:running,dm,snapper)
end

function RemoteWorkerExaminer(dm::Remote{AbstractDataModel}; snapper = Nullable{Remote{Snapper}}())
  id = where(dm)
  Remote{WorkerExaminer}(@spawnat id WorkerExaminer(fetch(dm),snapper = snapper))
end

function shutdown(examiner::Remote{WorkerExaminer})
  callfetch(shutdown,examiner)
end
function shutdown(examiner::WorkerExaminer)
  snap(examiner.snapper,:shutdown,"examiner")
  examiner.status = :shutdown
end

function examinefetch(examiner::Remote{WorkerExaminer},snapdata...;datadict...)
  callfetch(examine,examiner,snapdata...; datadict...)
end

# snapdata is just data passed to snapper.
# if keyargs is k1=v1,k2=v2... return Dict(k1=>result1,k2=>result2,...)


function examine(examiner::WorkerExaminer,snapdata...; datadict...)
  if examiner.status != :running
    return Dict()
  end
  if !isnull(examiner.snapper)
    snap(get(examiner.snapper),:startExamWorker,snapdata...)
  end
  result = Dict([(k, DataModel.evaluatePredProb(examiner.dm,v)) for (k,v) in datadict])
  if !isnull(examiner.snapper)
    snap(get(examiner.snapper),:endExamWorker,result)
  end
  #@show Dict([(k,sum(v[:predictiveprobs],1)) for (k,v) in result])
  result
end
