module MLUtilities

### miscellaneous utility functions

export logsumexp, checkgrad, BatchIterator, interpolate, interpolatedAverage

type BatchIterator
  nitems::Int
  batchsize::Int
  niters::Int
  randomize::Bool

  function BatchIterator(nitems::Int,batchsize::Int,niters::Int;randomize::Bool=false)
    @assert batchsize > 0
    @assert nitems >= batchsize
    @assert niters >= 0
    new(nitems,batchsize,niters,randomize)
  end
end

function Base.start(B::BatchIterator)
  (0,0,B.randomize ? randperm(B.nitems) : (1:B.nitems))
end
function Base.next(B::BatchIterator,state)
  iter = state[1]
  id = state[2]
  order = state[3]

  # +1 is necessary since the Base.start uses a 0 index
  # The remainder means that the batches wrap around if the batch size doesn't evenly divide the data size
  batch = order[rem(Int[i for i = id:id+B.batchsize-1],B.nitems)+1]

  if id+B.batchsize >= B.nitems
    if B.randomize
      order = randperm(B.nitems)
    end
  end

  id = rem(id+B.batchsize,B.nitems)
  iter += 1

  # Return the indices of the next minibatch, and update the state (which actually points to the next minibatch)
  (batch,(iter,id,order))
end

# Iterating is done when the iteration in the state matches the preset iteration limit in the iterator object
function Base.done(B::BatchIterator,state)
  state[1] >= B.niters
end


function logsumexp(x...)
  m = max(x...)
  return m + log(+([exp(a-m) for a in x]...))
end

# util to check gradient implementation by comparing with finite difference approximation
function checkgrad(x,func,grad; eps=1e-6)
  x = copy(x)
  ndim = length(x)
  f = func(x)
  g = grad(x)
  g2 = copy(g)
  for i=1:ndim
    x[i] += eps
    f2 = func(x)
    g2[i] = (f2-f)/eps
    x[i] -= eps
  end
  println("CheckGrad on $func with stepsize $eps")
  println("Maximum difference: $(maximum(abs(g2-g)))")
  println("Mean difference:    $(mean(abs(g2-g)))")
  (g,g2)
end

function interpolate(grid,xvals,yvals)
	#@assert grid[1] >= xvals[1]
	#@assert grid[end] <= xvals[end]
	i = 1
	n = length(xvals)
	yint = zeros(size(grid))
	for j = 1:length(grid)
		x = grid[j]

    # NOTE: Need this to fix weird bug where end of grid is slightly greater than maximum of xvals
    if j == length(grid)
      x -= 1e-6
    end

		while xvals[i+1] < x
			i += 1
		end

		yint[j] = (yvals[i+1]-yvals[i])/(xvals[i+1]-xvals[i])*(x-xvals[i])+yvals[i]
	end
	yint
end

function interpolatedAverage(xvals,yvals;gridsize=100)
	# Find the maximum and minimal values over all the arrays in xvals, and make a linearly spaced set of points over these
	# (each corresponds to a run)
	xstart = maximum([x[1] for x in xvals])
	xend = minimum([x[end] for x in xvals])
	grid = linspace(xstart, xend, gridsize)

	# Interpolate each run to values on grid
	ysum = zeros(gridsize)
	for i = 1:length(xvals)
		ysum += interpolate(grid,xvals[i],yvals[i])
	end

	# Take average over runs
	(grid, ysum/length(xvals))
end

end
