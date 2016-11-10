using SGMCMC
using DiagGaussian
using PosteriorServer

using JLD
using Utilities


#PosteriorServer version
function read_predprob_output(filename::AbstractString)
	d = JLD.load(filename)

	snapshots = d["results"][:snapshots]
	count_snapshots = length(snapshots)

	times = Array(Float64, 0)
	iters = Array(Float64, 0)
	worker_iters = Array(Float64, 0)
	accuracy = Array(Float64, 0)

	for i in 1:count_snapshots
		if snapshots[i].action == :startExamPredProb
			push!(times, snapshots[i].time)
			push!(iters, snapshots[i].data[1])
			push!(worker_iters, snapshots[i].data[2])
		elseif snapshots[i].action == :endExamPredProb
			 push!(accuracy, snapshots[i].data[1][:accuracy])
		end
	end
	times = times[1:length(accuracy)]
	iters = iters[1:length(accuracy)]
	worker_iters = worker_iters[1:length(accuracy)]
	return @dict(times, iters, worker_iters, accuracy)
end

#PosteriorServer version
function read_sgd_output(filename::AbstractString)
	d = JLD.load(filename)

	snapshots = d["results"][:snapshots]
	count_snapshots = length(snapshots)

	times = Array(Float64, 0)
	iters = Array(Float64, 0)
	worker_iters = Array(Float64, 0)
	accuracy = Array(Float64, 0)

	for i in 1:count_snapshots
		if snapshots[i].action == :startExam
			push!(times, snapshots[i].time)
			push!(iters, snapshots[i].data[2])
			push!(worker_iters, snapshots[i].data[3])
		elseif snapshots[i].action == :endExam
			#test for distributed phase
			if :state in keys(snapshots[i].data[1])
				push!(accuracy, snapshots[i].data[1][:state][:accuracy])
			end
		end
	end
	times = times[1:length(accuracy)]
	iters = iters[1:length(accuracy)]
	worker_iters = worker_iters[1:length(accuracy)]
	return @dict(times, iters, worker_iters, accuracy)
end

#PosteriorServer version
function read_distbayes_output(filename::AbstractString)
	d = JLD.load(filename)

	snapshots = d["results"][:snapshots]
	count_snapshots = length(snapshots)

	times = Array(Float64, 0)
	iters = Array(Float64, 0)
	worker_iters = Array(Float64, 0)

	acc_state = Array(Float64, 0)
	acc_post = Array(Float64, 0)
	acc_natpost = Array(Float64, 0)
	acc_avgparams = Array(Float64, 0)
	acc_avgnatpost = Array(Float64, 0)

	llik_state = Array(Float64, 0)
	llik_post = Array(Float64, 0)
	llik_natpost = Array(Float64, 0)
	llik_avgparams = Array(Float64, 0)
	llik_avgnatpost = Array(Float64, 0)

	times_train = Array(Float64, 0)
	iters_train = Array(Float64, 0)

	acc_train = Array(Float64, 0)
	llik_train = Array(Float64, 0)

	for i in 1:count_snapshots
		if snapshots[i].action == :startExam
			push!(times, snapshots[i].time)
			push!(iters, snapshots[i].data[2])
			push!(worker_iters, snapshots[i].data[3])
		elseif snapshots[i].action == :endExam
			#test for distributed phase
			if :natposterior in keys(snapshots[i].data[1])
				push!(acc_natpost, snapshots[i].data[1][:natposterior][:accuracy])
				push!(acc_avgnatpost, snapshots[i].data[1][:avgnatposterior][:accuracy])
				push!(acc_avgparams, snapshots[i].data[1][:avgparams][:accuracy])
				push!(llik_natpost, snapshots[i].data[1][:natposterior][:loglikelihood])
				push!(llik_avgnatpost, snapshots[i].data[1][:avgnatposterior][:loglikelihood])
				push!(llik_avgparams, snapshots[i].data[1][:avgparams][:loglikelihood])
			elseif :posterior in keys(snapshots[i].data[1])
				push!(acc_post, snapshots[i].data[1][:posterior][:accuracy])
				push!(acc_state, snapshots[i].data[1][:state][:accuracy])
				push!(llik_post, snapshots[i].data[1][:posterior][:loglikelihood])
				push!(llik_state, snapshots[i].data[1][:state][:loglikelihood])
			end
		elseif snapshots[i].action == :startTrainExam
			push!(times_train, snapshots[i].time)
			push!(iters_train, snapshots[i].data[2])
		elseif snapshots[i].action == :endTrainExam
			push!(acc_train, snapshots[i].data[1][:natposterior][:accuracy])
			push!(llik_train, snapshots[i].data[1][:natposterior][:loglikelihood])
		end
	end
	times_dist = times[length(acc_post)+1:end]
	times_init = times[1:length(acc_post)]
	iters_dist = iters[length(acc_post)+1:end]
	iters_init = iters[1:length(acc_post)]

	worker_iters_dist = worker_iters[length(acc_post)+1:end]
	worker_iters_init = worker_iters[1:length(acc_post)]

	specs = d["specs"]

	return @dict(times_init, iters_init, worker_iters_init, times_dist ,iters_dist, worker_iters_dist, acc_post,acc_state,acc_natpost, acc_avgparams, acc_avgnatpost,llik_state,llik_post,llik_natpost,llik_avgparams,llik_avgnatpost,times_train, iters_train,acc_train,llik_train, specs)
end