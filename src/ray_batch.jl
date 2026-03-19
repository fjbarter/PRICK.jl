# Batched ray state and tracing results
# ============================================================

@enum RayTermination::UInt8 begin
    term_sink = 1
    term_bbox_sink = 2
    term_max_bounces = 3
    term_max_length = 4
    term_no_hit = 5
end

struct RayTraceBatchResult
    escaped::BitVector
    total_length::Vector{Float64}
    nsteps::Vector{Int}
    termination::Vector{RayTermination}
    p0_termination_distance::Vector{Float64}
end

mutable struct ActiveTraversalCache{I <: Integer}
    contacts::Vector{ImplicitBVH.IndexPair{I}}
    thread_ncontacts_raw::Vector{I}
    num_contacts::Int
end

mutable struct RayBatchBuffer{T <: AbstractFloat}
    positions::Matrix{T}
    directions::Matrix{T}
    active::BitVector
    escaped::BitVector
    total_length::Vector{T}
    nsteps::Vector{Int}
    termination::Vector{RayTermination}
    p0_termination_distance::Vector{T}

    sphere_t::Vector{T}
    sphere_idx::Vector{Int}
    polyh_t::Vector{T}
    polyh_idx::Vector{Int}
    polyh_nx::Vector{T}
    polyh_ny::Vector{T}
    polyh_nz::Vector{T}

    wall_t::Vector{T}
    wall_idx::Vector{Int}
    wall_nx::Vector{T}
    wall_ny::Vector{T}
    wall_nz::Vector{T}
    wall_sink::BitVector
end

function RayBatchBuffer(::Type{T}, nrays::Integer) where {T <: AbstractFloat}
    nrays > 0 || throw(ArgumentError("nrays must be > 0"))

    positions = Matrix{T}(undef, 3, nrays)
    directions = Matrix{T}(undef, 3, nrays)
    active = trues(nrays)
    escaped = falses(nrays)
    total_length = zeros(T, nrays)
    nsteps = zeros(Int, nrays)
    termination = fill(term_no_hit, nrays)
    p0_termination_distance = zeros(T, nrays)

    sphere_t = fill(T(Inf), nrays)
    sphere_idx = zeros(Int, nrays)
    polyh_t = fill(T(Inf), nrays)
    polyh_idx = zeros(Int, nrays)
    polyh_nx = zeros(T, nrays)
    polyh_ny = zeros(T, nrays)
    polyh_nz = zeros(T, nrays)

    wall_t = fill(T(Inf), nrays)
    wall_idx = zeros(Int, nrays)
    wall_nx = zeros(T, nrays)
    wall_ny = zeros(T, nrays)
    wall_nz = zeros(T, nrays)
    wall_sink = falses(nrays)

    RayBatchBuffer(
        positions, directions, active, escaped, total_length, nsteps, termination, p0_termination_distance,
        sphere_t, sphere_idx, polyh_t, polyh_idx, polyh_nx, polyh_ny, polyh_nz,
        wall_t, wall_idx, wall_nx, wall_ny, wall_nz, wall_sink,
    )
end

function reset_ray_batch!(rb::RayBatchBuffer{T}, p0::NTuple{3,<:Real}, D0::AbstractMatrix{<:Real}) where {T}
    size(D0, 1) == 3 || throw(ArgumentError("directions must be a (3, N) matrix"))
    size(D0, 2) == size(rb.positions, 2) || throw(ArgumentError("direction count does not match buffer size"))

    n = size(D0, 2)
    px = T(p0[1]); py = T(p0[2]); pz = T(p0[3])

    @inbounds for i in 1:n
        rb.positions[1, i] = px
        rb.positions[2, i] = py
        rb.positions[3, i] = pz

        dx = T(D0[1, i])
        dy = T(D0[2, i])
        dz = T(D0[3, i])
        invn = inv(sqrt(dx * dx + dy * dy + dz * dz))
        if !isfinite(invn)
            throw(ArgumentError("direction column $i has near-zero norm"))
        end
        rb.directions[1, i] = dx * invn
        rb.directions[2, i] = dy * invn
        rb.directions[3, i] = dz * invn
    end

    fill!(rb.active, true)
    fill!(rb.escaped, false)
    fill!(rb.total_length, zero(T))
    fill!(rb.nsteps, 0)
    fill!(rb.termination, term_no_hit)
    fill!(rb.p0_termination_distance, zero(T))

    fill!(rb.sphere_t, T(Inf))
    fill!(rb.sphere_idx, 0)
    fill!(rb.polyh_t, T(Inf))
    fill!(rb.polyh_idx, 0)
    fill!(rb.polyh_nx, zero(T))
    fill!(rb.polyh_ny, zero(T))
    fill!(rb.polyh_nz, zero(T))
    fill!(rb.wall_t, T(Inf))
    fill!(rb.wall_idx, 0)
    fill!(rb.wall_nx, zero(T))
    fill!(rb.wall_ny, zero(T))
    fill!(rb.wall_nz, zero(T))
    fill!(rb.wall_sink, false)

    return rb
end

function reset_bounce_hits!(rb::RayBatchBuffer{T}) where {T}
    fill!(rb.sphere_t, T(Inf))
    fill!(rb.sphere_idx, 0)
    fill!(rb.polyh_t, T(Inf))
    fill!(rb.polyh_idx, 0)
    fill!(rb.polyh_nx, zero(T))
    fill!(rb.polyh_ny, zero(T))
    fill!(rb.polyh_nz, zero(T))
    fill!(rb.wall_t, T(Inf))
    fill!(rb.wall_idx, 0)
    fill!(rb.wall_nx, zero(T))
    fill!(rb.wall_ny, zero(T))
    fill!(rb.wall_nz, zero(T))
    fill!(rb.wall_sink, false)
    return rb
end

@inline function finish_ray!(
    rb::RayBatchBuffer,
    i::Int,
    reason::RayTermination,
    escaped::Bool,
)
    rb.active[i] = false
    rb.termination[i] = reason
    rb.escaped[i] = escaped
    return nothing
end

@inline function set_termination_distance!(
    rb::RayBatchBuffer{T},
    i::Int,
    p0::NTuple{3,Float64},
    p_term::NTuple{3,Float64},
) where {T}
    dx = p_term[1] - p0[1]
    dy = p_term[2] - p0[2]
    dz = p_term[3] - p0[3]
    rb.p0_termination_distance[i] = T(sqrt(dx * dx + dy * dy + dz * dz))
    return nothing
end
