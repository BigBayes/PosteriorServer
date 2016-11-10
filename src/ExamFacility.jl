import Base.isready

### ExamFacility handles requests to evaluate performance. A request will lead
### a performance evaluation if the Examiner is not busy.

function ExamFacility(examiner::Remote{Examiner};
                             nitersperexamine = typemax(Int),
                             timeperexamine = Inf)
  bouncer = Bouncer(nitersperentry=nitersperexamine,timeperentry = timeperexamine)
  results = RemoteRef()
  ExamFacility(bouncer,examiner,false,results)
end

function exam_avail(fac::ExamFacility)
  !fac.waiting && isready(fac.bouncer)
end

# request exam
function request_exam(fac::ExamFacility,args...;keyargs...)
    # if the facility is not waiting and the bouncer allows an exam
  if request(fac.bouncer) && !fac.waiting
    fac.waiting = true
    @schedule put!(fac.results,examinefetch(fac.examiner,args...;keyargs...))
    reset(fac.bouncer)
  end
end

# check if exams are ready
function results_ready(fac::ExamFacility)
  isready(fac.results)
end

function request_results(fac::ExamFacility)
  if isready(fac.results)
    wait_results(fac)
  else
    (fac.lastpushed,:none)
  end
end

function fetch_results(fac::ExamFacility)
  results = take!(fac.results)
  fac.waiting = false
  results
end
