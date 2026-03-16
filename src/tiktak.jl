####
#### Implementation of the TikTak method.
####

export TikTak

struct TikTak
    quasirandom_N::Int
    initial_N::Int
    θ_min::Float64
    θ_max::Float64
    θ_pow::Float64
end

"""
$(SIGNATURES)

The “TikTak” multistart method, as described in *Arnoud, Guvenen, and Kleineberg (2019)*.

This implements the *multistart* part, can be called with arbitrary local methods, see
[`multistart_minimization`](@ref).

# Arguments

- `quasirandom_N`: the number of quasirandom points for the first pass (using a Sobol
  sequence).

# Keyword arguments

- `keep_ratio`: the fraction of best quasirandom points which are kept

- `θ_min` and `θ_max` clamp the weight parameter, `θ_pow` determines the power it is raised
  to.

The defaults are from the paper cited above.
"""
function TikTak(quasirandom_N; keep_ratio = 0.1, θ_min = 0.1, θ_max = 0.995, θ_pow = 0.5)
    @argcheck 0 < keep_ratio ≤ 1
    TikTak(quasirandom_N, ceil(keep_ratio * quasirandom_N), θ_min, θ_max, θ_pow)
end

function _weight_parameter(t::TikTak, i)
    (; initial_N, θ_min, θ_max, θ_pow) = t
    clamp((i / initial_N)^θ_pow, θ_min, θ_max)
end

"""
$(SIGNATURES)

Determine if the value of the objective is a finite or `-Inf` real number.

Internal, a basic sanity check to catch errors early.
"""
_acceptable_value(value) = value isa Real && (isfinite(value) || isinf(value)) # we don't want NaNs

"""
$(SIGNATURES)

Evaluate `objective` at `location`, returning `location` and `value` in a `NamedTuple`.
Perform basic sanity checks.

Internal.
"""
function _objective_at_location(objective, location)
        value = objective(location)
        @argcheck _acceptable_value(value)
        (; location, value)
end

"""
$(SIGNATURES)

Evaluate and return points of an `N`-element Sobol sequence.

Execution is potentially parallelized using the strategy specified by `scheduler`.
"""
function sobol_starting_points(minimization_problem::MinimizationProblem, N::Integer, scheduler::Scheduler)
    (; objective, lower_bounds, upper_bounds) = minimization_problem
    s = SobolSeq(lower_bounds, upper_bounds)
    skip(s, N)                  # better uniformity
    points = collect(Iterators.take(s, N))
    _initial(x) = _objective_at_location(objective, x)
    tmap(_initial, Base.promote_op(_initial, eltype(points)), points; scheduler)
end

"""
$(SIGNATURES)

Helper function to keep the `N` points with the lowest `value`.
"""
function _keep_lowest!(xs, N)
    @argcheck 1 ≤ N ≤ length(xs)
    partialsort!(xs, 1:N, by = p -> p.value)
end

struct EnumeratedCatVector{T,X<:Tuple{AbstractVector{T},Vararg{AbstractVector{T}}}} <: AbstractVector{Tuple{Int,T}}
    v::X
end

Base.size(e::EnumeratedCatVector) = (sum(length, e.v, init=0),)
@generated function Base.getindex(e::EnumeratedCatVector, i::Int)
    result = Expr(:block, Expr(:meta, :inline), Expr(:meta, :propagate_inbounds),
        :(@boundscheck i ≤ 0 && throw(BoundsError(e, i))),
        :(i_ = i)
    )
    jmax = fieldcount(fieldtype(e, :v))
    for j in 1:fieldcount(fieldtype(e, :v))-1
        push!(result.args, :(len = length(e.v[$j])), :(if i ≤ len
            return i_, @inbounds e.v[$j][begin+i-1]
        else
            i -= len
        end))
    end
    push!(result.args, :(return i_, e.v[$jmax][i]))
    return result
end
Base.IndexStyle(::Type{X}) where {X<:EnumeratedCatVector} = IndexLinear()

"""
$(SIGNATURES)

Solve `minimization_problem` by using `local_method` within `multistart_method`.

Initial point search and subsequent minimizations are executed according to the scheduler
policy. The legacy argument `use_threads` enables default parallelization settings. More
control over multi-threading is possible by passing an explicit `scheduler` or, preferrably,
by using keyword arguments for constructing a scheduler from `OhMyThreads`.

`prepend_points` should contain a vector of initial starting points that are prepended to
the Sobol sequence. These are useful if a guess is available for the vicinity of the
optimum.

!!! warning
    It is not advisable to use the `chunksize` option when constructing the scheduler, as the
    construction of all possible initial point candidates and the actual minimization both use
    parallelization, but have different numbers of points. It is recommended to use the keyword
    interface instead of a `Scheduler` object, as different types of schedulers are optimal for
    the different stages: a `StaticScheduler` for initial point selection and a `GreedyScheduler`
    for the local minimization.
"""
function multistart_minimization(multistart_method::TikTak, local_method,
                                 minimization_problem, scheduler::Union{Scheduler,Nothing}=nothing;
                                 prepend_points = Vector{Vector{Float64}}(),
                                 use_threads::Union{Nothing,Bool}=nothing, # legacy argument
                                 ntasks::Union{Integer,Nothing}=nothing,
                                 chunksize::Union{Integer,Nothing}=nothing,
                                 minchunksize::Union{Integer,Nothing}=nothing,
                                 chunking::Union{Bool,Nothing}=nothing,
                                 nchunks::Union{Integer,Nothing}=nothing,
                                 split::Union{OhMyThreads.Split,Symbol,Nothing}=nothing)
    !isnothing(use_threads) && !isnothing(scheduler) &&
        error("Legacy argument use_threads cannot combined with new scheduler arguments")
    !isnothing(scheduler) && !all(isnothing, (ntasks, chunksize, minchunksize, chunking, split)) &&
        error("Either a Scheduler or keyword options for the scheduler have to be specified")
    if isnothing(use_threads)
        use_threads = !(scheduler isa SerialScheduler || isnothing(ntasks) || isone(ntasks))
    end
    if use_threads
        if isnothing(scheduler)
            initial_scheduler = StaticScheduler(; ntasks=isnothing(ntasks) ? Threads.nthreads() : ntasks,
                                                chunksize, minchunksize,
                                                chunking=isnothing(chunking) ? true : chunking,
                                                split=isnothing(split) ? OhMyThreads.Consecutive() : split)
            optimization_scheduler = GreedyScheduler(; ntasks=isnothing(ntasks) ? Threads.nthreads() : ntasks,
                                                     chunking=isnothing(chunking) ? false : chunking,
                                                     nchunks, chunksize, minchunksize,
                                                     split=isnothing(split) ? OhMyThreads.RoundRobin() : split)
        else
            initial_scheduler = optimization_scheduler = scheduler
        end
    else
        initial_scheduler = optimization_scheduler = SerialScheduler()
    end
    # We now have at most two choices for the schedulers, so union splitting will work - this is type stable
    (; quasirandom_N, initial_N, θ_min, θ_max, θ_pow) = multistart_method
    (; objective, lower_bounds, upper_bounds) = minimization_problem
    @argcheck(all(x -> all(lower_bounds .≤ x .≤ upper_bounds), prepend_points),
              "prepend_points outside problem bounds")
    quasirandom_points = sobol_starting_points(minimization_problem, quasirandom_N, initial_scheduler)
    initial_points = _keep_lowest!(quasirandom_points, initial_N)
    all_points = EnumeratedCatVector((map(Base.Fix1(_objective_at_location, objective), prepend_points), quasirandom_points))
    function _step((_, visited_minimum)::Tuple{Int,NamedTuple}, (i, initial_point)::Tuple{Int,NamedTuple})
        θ = _weight_parameter(multistart_method, i)
        x = @. (1 - θ) * initial_point.location + θ * visited_minimum.location
        local_minimum = local_minimization(local_method, minimization_problem, x)
        local_minimum ≡ nothing && return visited_minimum
        # we don't care about the index parameter, but for type stability of the function, we'll always give back one
        (0, local_minimum.value < visited_minimum.value ? (; local_minimum.location, local_minimum.value) : visited_minimum)
    end
    # do not drop the first point - the local optimizer won't run on init!
    treduce(_step, all_points; scheduler=optimization_scheduler, init=first(all_points))[2]
end
