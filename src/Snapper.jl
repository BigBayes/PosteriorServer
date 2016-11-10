export Snapshot, Snapper, actions, snap

# Snapper is a util to save data about the distributed algorithm for
# later analysis

# specifies output actions that will be displayed
const actions = [:startup,:shutdown,
                 :run, :stop, :interrupt,
                 #:startSync, :endSync,
                 :startExam, :endExam,
                 :startExamPredProb, :endExamPredProb,
                 :startTrainExam, :endTrainExam,
                 :error
                 ]


function Snapper(where::Int; keyargs...)
  Remote{Snapper}(@spawnat where Snapper(;keyargs...))
end

function Snapper(;inittime::Float64 = time(),
                  stream = STDOUT,
                  filename = "",
                  tostream = actions)
  if !isempty(filename)
    stream = open(filename,"w")
  end
  Snapper(inittime,stream,[],tostream)
end

function snapshots(snapper::Snapper)
  snapper.snapshots
end

function fetchsnapshots(snapper::Remote{Snapper})
  callfetch(snapshots,snapper)
end


# save a snapshot related to some action with some related data.
function snap(ref::Nullable{Remote{Snapper}},action::Symbol,data...)
  if !isnull(ref)
    snap(get(ref),action,data...)
  else
    error("no snapper")
  end
end

function snap(ref::Remote{Snapper},action::Symbol,data...)
  call(snap,ref,time(),action,data...)
end

function snap(snapper::Snapper,t::Float64,action::Symbol,data...)
  snapshot = Snapshot(t-snapper.inittime,action,data...)
  push!(snapper.snapshots,snapshot)
  # display in output if action in snapper.tostream
  if action in snapper.tostream
    show(snapper.io,snapshot)
  end
end

function Base.show(io::IO,snapshots::Array{Snapshot,1})
  for i=1:length(snapshots)
    show(io,snapshots[i])
  end
end

const timefmt = FormatSpec(">8.1f")
const actionfmt = FormatSpec("<12s")

function Base.show(io::IO,snapshot::Snapshot)
  etime = fmt(timefmt,snapshot.time)
  action = fmt(actionfmt,snapshot.action)

  data = snapshot.data
  if action==:startSync || action==:endSync
      println(io,"$iter @ $etime: worker $(data[1]) $action at iter $(data[2])")
  elseif action==:examinePosterior
      println(io,"$iter @ $etime: $action, accuracy $(data[1])")
  else
      println(io,"$etime $action: $data")
  end
end
