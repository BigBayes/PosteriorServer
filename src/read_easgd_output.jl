using SGMCMC
using DiagGaussian
using ElasticServer

using JLD
using Utilities

#DownpourServer version, returns only distributed phase
function read_elastic_output(filename::AbstractString)
	d = JLD.load(filename)

	snapshots = d["results"][:snapshots]
	count_snapshots = length(snapshots)

	initial_learning_time = d["results"][:initial_learning_time]

	times = Array(Float64, 0)
	iters = Array(Float64, 0)
	worker_iters = Array(Float64, 0)
	acc = Array(Float64, 0)

	for i in 1:count_snapshots
		if snapshots[i].action == :startExam
			push!(times, snapshots[i].time)
			push!(iters, snapshots[i].data[2])
			push!(worker_iters, snapshots[i].data[3])
		elseif snapshots[i].action == :endExam
			push!(acc, snapshots[i].data[1][:params][:accuracy])
		end
	end

	# Find how many iterations to skip over
	decrease_in_iters = find(x -> x < 0, diff(iters))
	initial_iters = 0
	if length(decrease_in_iters) > 0
		initial_iters = decrease_in_iters[1]
	end

	times = times[(1 + initial_iters):end]
	iters = iters[(1 + initial_iters):end]
	worker_iters = worker_iters[(1 + initial_iters):end]
	acc = acc[(1 + initial_iters):end]
	specs = d["specs"]

	return @dict(times, iters, worker_iters, acc, specs, initial_learning_time)
end