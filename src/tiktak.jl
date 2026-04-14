####
#### Implementation of the TikTak method.
####

export TikTak

VERSION ≥ v"1.11.0-DEV.469" && eval(Expr(:public, :default_initial_point_scheduler, :default_local_scheduler))

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

"""
$(SIGNATURES)

Constructs the scheduler used to compute the function values at the initial points. If
multiple threads are used, this is a `StaticScheduler`, to which arguments may be passed.
For a single-threaded run, this is a `SerialScheduler`.

This function is not exported.
"""
default_initial_point_scheduler(; ntasks=Threads.nthreads(), chunksize=nothing, minchunksize=nothing,
                                split::Union{OhMyThreads.Split,Symbol}=OhMyThreads.Consecutive(), chunking::Bool=true,
                                thread_ids=Base.OneTo(ntasks)) =
    isone(Threads.nthreads()) || isone(ntasks) ? SerialScheduler() :
                                                 StaticScheduler(; ntasks, chunksize, minchunksize, split, chunking,
                                                                 thread_ids)

"""
$(SIGNATURES)

Constructs the scheduler used to invoke the local minimizer on the objective from various
initial points. If multiple threads are used, this is a `GreedyScheduler`, to which
arguments may be passed. For a single-threaded run, this is a `SerialScheduler`.

This function is not exported.

!!! warning
    It is highly recommended to only use the `RoundRobin` splitting strategy in order to
    resemble most closely the steps a serial execution would take.
"""
default_local_scheduler(; ntasks=Threads.nthreads(), nchunks=nothing, chunksize=nothing, minchunksize=nothing,
                        split::Union{OhMyThreads.Split,Symbol}=OhMyThreads.RoundRobin(), chunking::Bool=false,
                        thread_ids=nothing) =
    isone(Threads.nthreads()) || isone(ntasks) ? SerialScheduler() :
                                                 GreedyScheduler(; ntasks, nchunks, chunksize, minchunksize, split, chunking,
                                                                 thread_ids)

# We need an indexable enumerate() and also want to iterate over the concatenation of two vectors without having to concatenate
# them. Define this simple struct to achieve both things.
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

struct NoLock <: Base.AbstractLock end
Base.lock(::NoLock) = nothing
Base.trylock(::NoLock) = true
Base.unlock(::NoLock) = nothing
Base.islocked(::NoLock) = false

mutable struct VisitedMinimum{V,T}
    location::V
    value::T
end

"""
$(SIGNATURES)

Solve `minimization_problem` by using `local_method` within `multistart_method`.

Initial point search and subsequent minimizations are executed according to the scheduler
policy. The argument `use_threads` enables default parallelization settings; for more
fine-grained control, explicitly specify the schedulers for the two stages.

`prepend_points` should contain a vector of initial starting points that are prepended to
the Sobol sequence. These are useful if a guess is available for the vicinity of the
optimum.

!!! warning
    If the local optimization method is deterministic, so is the multistart optimization,
    provided the `local_scheduler` is serial (multi-threading initial point generation is
    fine). For a multi-threaded `local_scheduler`, no guarantees can be made either to
    visit the same starting points in each run.
    While chunking can be enabled, it is strongly recommended to use the `RoundRobin`
    method is to be preferred over the `Consecutive` splitting strategy for
    `local_scheduler`, so that it is more likely to be mixing initial points from
    probably better starting points.
"""
function multistart_minimization(multistart_method::TikTak, local_method,
                                 minimization_problem;
                                 prepend_points = Vector{Vector{Float64}}(),
                                 use_threads::Bool=false,
                                 initial_point_scheduler::Scheduler=use_threads ? default_initial_point_scheduler() :
                                                                    SerialScheduler(),
                                 local_scheduler::Scheduler=use_threads ? default_local_scheduler() : SerialScheduler())
    # We now have at most two choices for the schedulers, so union splitting will work - this is type stable
    (; quasirandom_N, initial_N, θ_min, θ_max, θ_pow) = multistart_method
    (; objective, lower_bounds, upper_bounds) = minimization_problem
    @argcheck(all(x -> begin
        length(lower_bounds) == length(x) || throw(DimensionMismatch("prepend_points with invalid length"))
        all(issorted, zip(lower_bounds, x, upper_bounds))
    end, prepend_points), "prepend_points outside problem bounds")
    quasirandom_points = sobol_starting_points(minimization_problem, quasirandom_N, initial_point_scheduler)
    initial_points = _keep_lowest!(quasirandom_points, initial_N)
    all_points = EnumeratedCatVector((map(Base.Fix1(_objective_at_location, objective), prepend_points), initial_points))

    init = first(all_points)[2]
    if local_scheduler isa SerialScheduler
        tlv = Ref(similar(init.location))
        l = NoLock()
    else
        tlv = OhMyThreads.TaskLocalValue{Base.RefValue{typeof(init.location)}}(let x=init.location; () -> Ref(similar(x)) end)
        l = Threads.SpinLock()
    end
    visited_minimum = VisitedMinimum(copy(init.location), init.value)
    _step = let tlv=tlv, l=l, visited_minimum=visited_minimum
        function _step((i, initial_point)::Tuple{Int,NamedTuple})
            θ = _weight_parameter(multistart_method, i)
            xref = local_scheduler isa SerialScheduler ? tlv : tlv[]
            x = xref[]
            @. x = (1 - θ) * initial_point.location + θ * visited_minimum.location
            local_minimum = local_minimization(local_method, minimization_problem, x)
            local_minimum ≡ nothing && return visited_minimum
            if local_minimum.value < visited_minimum.value
                lock(l)
                if local_minimum.value < visited_minimum.value
                    # If local_minimization works in-place, local_minimum ≡ x - so the next iteration in this task would
                    # overwrite the minimum position. Hence, we swap our temporary with the previous minimum, which has now
                    # become obsolete.
                    xref[] = visited_minimum.location
                    visited_minimum.location = local_minimum.location
                    visited_minimum.value = local_minimum.value
                end
                unlock(l)
            end
            return
        end
    end
    tforeach(_step, all_points; scheduler=local_scheduler)
    return (; location=visited_minimum.location, value=visited_minimum.value)
end
