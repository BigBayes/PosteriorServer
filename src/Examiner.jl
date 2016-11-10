export Examiner, examine, examinefetch

### Examiner handles the evaluation of performance metrics such as training and test error
# The Examiner type contains:
#                       a status:       :running or :shutdown
#                       a data model:   implements the model to evaluate on
#                       a snapper:      to save results
#                       train:          Bool to distinguish between train and test performance
###

# constructor
function Examiner(dm::AbstractDataModel; snapper = Nullable{Remote{Snapper}}(), train::Bool = false)
  Examiner(:running,dm,snapper, train)
end

# constructor for Examiner on different process
function RemoteExaminer(dm::Remote{AbstractDataModel}; snapper = Nullable{Remote{Snapper}}(), train::Bool = false)
  id = where(dm)
  Remote{Examiner}(@spawnat id Examiner(fetch(dm),snapper = snapper, train = train))
end

# shutdown
function shutdown(examiner::Remote{Examiner})
  callfetch(shutdown,examiner)
end
function shutdown(examiner::Examiner)
    if examiner.train
        snap(examiner.snapper,:shutdown,"trainexaminer")
    else
        snap(examiner.snapper,:shutdown,"examiner")
    end
  examiner.status = :shutdown
end

# examine for Remote Examiners
function examinefetch(examiner::Remote{Examiner},snapdata...;datadict...)
  callfetch(examine,examiner,snapdata...; datadict...)
end

# snapdata is just data passed to snapper.
# if keyargs is k1=v1,k2=v2... return Dict(k1=>result1,k2=>result2,...)
function examine(examiner::Examiner,snapdata...; datadict...)
  if examiner.status != :running
    return Dict()
  end

  # if snapper available save appropriate snapshot
  if !isnull(examiner.snapper)
    if examiner.train
        snap(get(examiner.snapper),:startTrainExam,snapdata...)
    else
        snap(get(examiner.snapper),:startExam,snapdata...)
    end
  end

  # evaluate
  result = Dict([(k, DataModel.evaluate(examiner.dm,v)) for (k,v) in datadict])

  # save results
  if !isnull(examiner.snapper)
      if examiner.train
          snap(get(examiner.snapper),:endTrainExam,result)
      else
          snap(get(examiner.snapper),:endExam,result)
      end
  end
  result
end

function Base.show(io::IO,examiner::AbstractExaminer)
  print(io,"Examiner()")
end
