
### WorkerExamFacility


function WorkerExamFacility(examiner::Remote{WorkerExaminer};
                             nitersperexamine = typemax(Int),
                             timeperexamine = Inf)
  bouncer = Bouncer(nitersperentry=nitersperexamine,timeperentry = timeperexamine)
  results = RemoteRef()
  WorkerExamFacility(bouncer,examiner,false,results)
end

function exam_avail(fac::WorkerExamFacility)
  !fac.waiting && isready(fac.bouncer)
end

function request_exam(fac::WorkerExamFacility,args...;keyargs...)
  if request(fac.bouncer) && !fac.waiting
    fac.waiting = true
    @schedule put!(fac.results,examinefetch(fac.examiner,args...;keyargs...))
    reset(fac.bouncer)
  end
end

function results_ready(fac::WorkerExamFacility)
  isready(fac.results)
end

function request_results(fac::WorkerExamFacility)
  if isready(fac.results)
    wait_results(fac)
  else
    (fac.lastpushed,:none)
  end
end

function fetch_results(fac::WorkerExamFacility)
  results = take!(fac.results)
  fac.waiting = false
  results
end
