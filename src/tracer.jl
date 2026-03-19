# ============================================================
# Tracer
# ============================================================

@inline reflect_specular(d::NTuple{3,Float64}, n::NTuple{3,Float64}) = sub3(d, mul3(n, 2.0 * dot3(d, n)))

const _bbox_warned = Ref(false)
const _polyh_start_inside_warned = Ref(false)

# Legacy visualisation container kept for backwards compatibility with visualise.jl.
struct RayTraceResult
    escaped::Bool
    total_length::Float64
    nsteps::Int
    path::Vector{SVector{3,Float64}}
end

function start_outside_spheres_aabb!(
    rb::RayBatchBuffer{Float64},
    spheres::SphereBVH;
    tol::Float64=1e-12,
    eps_hit::Float64=1e-12,
    eps_shift::Float64=1e-10,
    count_start_solid::Bool=false,
)
    P = rb.positions
    D = rb.directions
    T = rb.total_length

    @inbounds for i in axes(P, 2)
        p = (P[1, i], P[2, i], P[3, i])
        d = (D[1, i], D[2, i], D[3, i])
        tmax = 0.0
        n_inside = 0

        for j in eachindex(spheres.radii)
            c = spheres.centres[j]
            r = spheres.radii[j]

            dx = p[1] - c[1]; adx = abs(dx)
            adx > r && continue
            dy = p[2] - c[2]; ady = abs(dy)
            ady > r && continue
            dz = p[3] - c[3]; adz = abs(dz)
            adz > r && continue

            r_eff = r - tol
            r_eff <= 0 && continue
            if (dx * dx + dy * dy + dz * dz) < (r_eff * r_eff)
                n_inside += 1
                t = ray_sphere_exit_distance(p, d, c, r; eps=eps_hit)
                if t !== nothing && t > tmax
                    tmax = t
                end
            end
        end

        if n_inside > 0
            p_next = madd3(p, tmax + eps_shift, d)
            P[1, i] = p_next[1]
            P[2, i] = p_next[2]
            P[3, i] = p_next[3]
            if count_start_solid
                T[i] += tmax
            end
        end
    end

    return nothing
end

function trace_rays(
    p0::NTuple{3,<:Real},
    directions::AbstractMatrix{<:Real},
    surfaces::AbstractVector{<:TriangleSurface},
    spheres::SphereBVH;
    eps::Float64=1e-10,
    tri_eps::Float64=1e-12,
    sph_eps::Float64=1e-12,
    max_bounces::Int=1_000_000,
    max_length::Float64=Inf,
    allow_start_inside::Bool=true,
    count_start_solid::Bool=false,
    options=ImplicitBVH.BVHOptions(),
)
    surface_bvh = build_surface_bvh(surfaces)
    return trace_rays(
        p0, directions, surface_bvh, spheres;
        eps=eps,
        tri_eps=tri_eps,
        sph_eps=sph_eps,
        max_bounces=max_bounces,
        max_length=max_length,
        allow_start_inside=allow_start_inside,
        count_start_solid=count_start_solid,
        options=options,
    )
end

function trace_rays(
    p0::NTuple{3,<:Real},
    directions::AbstractMatrix{<:Real},
    surface_bvh::SurfaceBVH,
    spheres::SphereBVH;
    eps::Float64=1e-10,
    tri_eps::Float64=1e-12,
    sph_eps::Float64=1e-12,
    max_bounces::Int=1_000_000,
    max_length::Float64=Inf,
    allow_start_inside::Bool=true,
    count_start_solid::Bool=false,
    options=ImplicitBVH.BVHOptions(),
)
    max_bounces >= 0 || throw(ArgumentError("max_bounces must be >= 0"))
    size(directions, 1) == 3 || throw(ArgumentError("directions must be a (3, N) matrix"))
    n_rays = size(directions, 2)
    n_rays > 0 || throw(ArgumentError("directions must contain at least one ray"))

    p0f = (Float64(p0[1]), Float64(p0[2]), Float64(p0[3]))
    rb = RayBatchBuffer(Float64, n_rays)
    reset_ray_batch!(rb, p0f, directions)

    if max_bounces == 0
        @inbounds for i in eachindex(rb.active)
            set_termination_distance!(rb, i, p0f, p0f)
            finish_ray!(rb, i, PRICK.term_max_bounces, false)
        end
        return RayTraceBatchResult(
            copy(rb.escaped),
            copy(rb.total_length),
            copy(rb.nsteps),
            copy(rb.termination),
            copy(rb.p0_termination_distance),
        )
    end

    allow_start_inside && start_outside_spheres_aabb!(
        rb, spheres; tol=1e-12, eps_hit=sph_eps, eps_shift=eps, count_start_solid=count_start_solid
    )

    sphere_traversal_cache = nothing
    wall_traversal_cache = nothing

    active_count = n_rays
    while active_count > 0
        reset_bounce_hits!(rb)

        sphere_traversal = traverse_rays_active_lvt!(
            spheres.bvh, rb.positions, rb.directions, rb.active;
            cache=sphere_traversal_cache,
            options=options,
        )
        sphere_traversal_cache = sphere_traversal
        update_nearest_sphere_hits!(rb, spheres, sphere_traversal; eps=sph_eps)

        wall_traversal = traverse_rays_active_lvt!(
            surface_bvh.bvh, rb.positions, rb.directions, rb.active;
            cache=wall_traversal_cache,
            options=options,
        )
        wall_traversal_cache = wall_traversal
        update_nearest_surface_hits!(rb, surface_bvh, wall_traversal; eps=tri_eps)

        @inbounds for i in 1:n_rays
            rb.active[i] || continue
            p = (rb.positions[1, i], rb.positions[2, i], rb.positions[3, i])
            d = (rb.directions[1, i], rb.directions[2, i], rb.directions[3, i])

            ts = rb.sphere_t[i]
            tm = rb.wall_t[i]
            tbox = ray_aabb_intersect(p, d, surface_bvh.bbox_mins, surface_bvh.bbox_maxs; eps=eps)
            tboxf = tbox === nothing ? Inf : tbox

            if isinf(ts) && isinf(tm) && isinf(tboxf)
                set_termination_distance!(rb, i, p0f, p)
                finish_ray!(rb, i, term_no_hit, false)
                active_count -= 1
                continue
            end

            t = min(ts, tm, tboxf)
            p_hit = madd3(p, t, d)
            rb.total_length[i] += t

            if rb.total_length[i] > max_length
                set_termination_distance!(rb, i, p0f, p_hit)
                finish_ray!(rb, i, PRICK.term_max_length, false)
                active_count -= 1
                continue
            end

            if tboxf <= tm && tboxf <= ts
                if !_bbox_warned[]
                    @warn "Ray hit bounding box sink; mesh may not be watertight."
                    _bbox_warned[] = true
                end
                set_termination_distance!(rb, i, p0f, p_hit)
                finish_ray!(rb, i, term_bbox_sink, true)
                active_count -= 1
                continue
            elseif tm < ts
                nhat = (rb.wall_nx[i], rb.wall_ny[i], rb.wall_nz[i])
                if dot3(d, nhat) > 0.0
                    nhat = mul3(nhat, -1.0)
                end

                if rb.wall_sink[i]
                    set_termination_distance!(rb, i, p0f, p_hit)
                    finish_ray!(rb, i, term_sink, true)
                    active_count -= 1
                    continue
                end

                d_next = normalize3(reflect_specular(d, nhat))
                p_next = madd3(p_hit, eps, nhat)
                rb.directions[1, i] = d_next[1]
                rb.directions[2, i] = d_next[2]
                rb.directions[3, i] = d_next[3]
                rb.positions[1, i] = p_next[1]
                rb.positions[2, i] = p_next[2]
                rb.positions[3, i] = p_next[3]

                rb.nsteps[i] += 1
                if rb.nsteps[i] >= max_bounces
                    set_termination_distance!(rb, i, p0f, p_hit)
                    finish_ray!(rb, i, PRICK.term_max_bounces, false)
                    active_count -= 1
                end
            else
                idx = rb.sphere_idx[i]
                idx > 0 || throw(ArgumentError("internal error: missing sphere index for finite sphere hit"))

                c = spheres.centres[idx]
                r = spheres.radii[idx]
                nhat = normalize3(sub3(p_hit, c))
                d_next = normalize3(reflect_specular(d, nhat))
                eps_rel = max(eps, 1e-9 * r)
                p_next = add3(c, mul3(nhat, r + eps_rel))
                rb.directions[1, i] = d_next[1]
                rb.directions[2, i] = d_next[2]
                rb.directions[3, i] = d_next[3]
                rb.positions[1, i] = p_next[1]
                rb.positions[2, i] = p_next[2]
                rb.positions[3, i] = p_next[3]

                rb.nsteps[i] += 1
                if rb.nsteps[i] >= max_bounces
                    set_termination_distance!(rb, i, p0f, p_hit)
                    finish_ray!(rb, i, PRICK.term_max_bounces, false)
                    active_count -= 1
                end
            end
        end
    end

    return RayTraceBatchResult(
        copy(rb.escaped),
        copy(rb.total_length),
        copy(rb.nsteps),
        copy(rb.termination),
        copy(rb.p0_termination_distance),
    )
end

function trace_rays(
    p0::NTuple{3,<:Real},
    directions::AbstractMatrix{<:Real},
    surfaces::AbstractVector{<:TriangleSurface},
    polyh::PolyhedralBVH;
    eps::Float64=1e-10,
    tri_eps::Float64=1e-12,
    polyh_eps::Float64=1e-12,
    max_bounces::Int=1_000_000,
    max_length::Float64=Inf,
    allow_start_inside::Bool=true,
    count_start_solid::Bool=false,
    options=ImplicitBVH.BVHOptions(),
)
    surface_bvh = build_surface_bvh(surfaces)
    return trace_rays(
        p0, directions, surface_bvh, polyh;
        eps=eps,
        tri_eps=tri_eps,
        polyh_eps=polyh_eps,
        max_bounces=max_bounces,
        max_length=max_length,
        allow_start_inside=allow_start_inside,
        count_start_solid=count_start_solid,
        options=options,
    )
end

function trace_rays(
    p0::NTuple{3,<:Real},
    directions::AbstractMatrix{<:Real},
    surface_bvh::SurfaceBVH,
    polyh::PolyhedralBVH;
    eps::Float64=1e-10,
    tri_eps::Float64=1e-12,
    polyh_eps::Float64=1e-12,
    max_bounces::Int=1_000_000,
    max_length::Float64=Inf,
    allow_start_inside::Bool=true,
    count_start_solid::Bool=false,
    options=ImplicitBVH.BVHOptions(),
)
    max_bounces >= 0 || throw(ArgumentError("max_bounces must be >= 0"))
    size(directions, 1) == 3 || throw(ArgumentError("directions must be a (3, N) matrix"))
    n_rays = size(directions, 2)
    n_rays > 0 || throw(ArgumentError("directions must contain at least one ray"))

    p0f = (Float64(p0[1]), Float64(p0[2]), Float64(p0[3]))
    rb = RayBatchBuffer(Float64, n_rays)
    reset_ray_batch!(rb, p0f, directions)

    if max_bounces == 0
        @inbounds for i in eachindex(rb.active)
            set_termination_distance!(rb, i, p0f, p0f)
            finish_ray!(rb, i, PRICK.term_max_bounces, false)
        end
        return RayTraceBatchResult(
            copy(rb.escaped),
            copy(rb.total_length),
            copy(rb.nsteps),
            copy(rb.termination),
            copy(rb.p0_termination_distance),
        )
    end

    if allow_start_inside && !_polyh_start_inside_warned[]
        @warn "allow_start_inside for polyhedral tracing is currently not implemented; start rays in void space."
        _polyh_start_inside_warned[] = true
    end
    count_start_solid && @warn "count_start_solid has no effect for polyhedral tracing at present."

    polyh_traversal_cache = nothing
    wall_traversal_cache = nothing

    active_count = n_rays
    while active_count > 0
        reset_bounce_hits!(rb)

        polyh_traversal = traverse_rays_active_lvt!(
            polyh.bvh, rb.positions, rb.directions, rb.active;
            cache=polyh_traversal_cache,
            options=options,
        )
        polyh_traversal_cache = polyh_traversal
        update_nearest_polyh_hits!(rb, polyh, polyh_traversal; eps=polyh_eps)

        wall_traversal = traverse_rays_active_lvt!(
            surface_bvh.bvh, rb.positions, rb.directions, rb.active;
            cache=wall_traversal_cache,
            options=options,
        )
        wall_traversal_cache = wall_traversal
        update_nearest_surface_hits!(rb, surface_bvh, wall_traversal; eps=tri_eps)

        @inbounds for i in 1:n_rays
            rb.active[i] || continue
            p = (rb.positions[1, i], rb.positions[2, i], rb.positions[3, i])
            d = (rb.directions[1, i], rb.directions[2, i], rb.directions[3, i])

            tp = rb.polyh_t[i]
            tm = rb.wall_t[i]
            tbox = ray_aabb_intersect(p, d, surface_bvh.bbox_mins, surface_bvh.bbox_maxs; eps=eps)
            tboxf = tbox === nothing ? Inf : tbox

            if isinf(tp) && isinf(tm) && isinf(tboxf)
                set_termination_distance!(rb, i, p0f, p)
                finish_ray!(rb, i, term_no_hit, false)
                active_count -= 1
                continue
            end

            t = min(tp, tm, tboxf)
            p_hit = madd3(p, t, d)
            rb.total_length[i] += t

            if rb.total_length[i] > max_length
                set_termination_distance!(rb, i, p0f, p_hit)
                finish_ray!(rb, i, PRICK.term_max_length, false)
                active_count -= 1
                continue
            end

            if tboxf <= tm && tboxf <= tp
                if !_bbox_warned[]
                    @warn "Ray hit bounding box sink; mesh may not be watertight."
                    _bbox_warned[] = true
                end
                set_termination_distance!(rb, i, p0f, p_hit)
                finish_ray!(rb, i, term_bbox_sink, true)
                active_count -= 1
                continue
            elseif tm < tp
                nhat = (rb.wall_nx[i], rb.wall_ny[i], rb.wall_nz[i])
                if dot3(d, nhat) > 0.0
                    nhat = mul3(nhat, -1.0)
                end

                if rb.wall_sink[i]
                    set_termination_distance!(rb, i, p0f, p_hit)
                    finish_ray!(rb, i, term_sink, true)
                    active_count -= 1
                    continue
                end

                d_next = normalize3(reflect_specular(d, nhat))
                p_next = madd3(p_hit, eps, nhat)
                rb.directions[1, i] = d_next[1]
                rb.directions[2, i] = d_next[2]
                rb.directions[3, i] = d_next[3]
                rb.positions[1, i] = p_next[1]
                rb.positions[2, i] = p_next[2]
                rb.positions[3, i] = p_next[3]

                rb.nsteps[i] += 1
                if rb.nsteps[i] >= max_bounces
                    set_termination_distance!(rb, i, p0f, p_hit)
                    finish_ray!(rb, i, PRICK.term_max_bounces, false)
                    active_count -= 1
                end
            else
                idx = rb.polyh_idx[i]
                idx > 0 || throw(ArgumentError("internal error: missing polyhedral index for finite polyhedral hit"))

                nhat = (rb.polyh_nx[i], rb.polyh_ny[i], rb.polyh_nz[i])
                if dot3(d, nhat) > 0.0
                    nhat = mul3(nhat, -1.0)
                end
                d_next = normalize3(reflect_specular(d, nhat))
                eps_rel = max(eps, 1e-9 * polyh.radii[idx])
                p_next = madd3(p_hit, eps_rel, nhat)
                rb.directions[1, i] = d_next[1]
                rb.directions[2, i] = d_next[2]
                rb.directions[3, i] = d_next[3]
                rb.positions[1, i] = p_next[1]
                rb.positions[2, i] = p_next[2]
                rb.positions[3, i] = p_next[3]

                rb.nsteps[i] += 1
                if rb.nsteps[i] >= max_bounces
                    set_termination_distance!(rb, i, p0f, p_hit)
                    finish_ray!(rb, i, PRICK.term_max_bounces, false)
                    active_count -= 1
                end
            end
        end
    end

    return RayTraceBatchResult(
        copy(rb.escaped),
        copy(rb.total_length),
        copy(rb.nsteps),
        copy(rb.termination),
        copy(rb.p0_termination_distance),
    )
end

function find_void_rrhc(
    X::Matrix{Float64}, r::Vector{Float64};
    restarts::Int=50, steps::Int=2_000, rel_tol::Float64=1e-2,
    lambda::Float64=0.05,
)
    cx, cy, cz = mean(X[1, :]), mean(X[2, :]), mean(X[3, :])
    sx, sy, sz = stddev(X[1, :]), stddev(X[2, :]), stddev(X[3, :])

    bed_center = (cx, cy, cz)
    bounds_min = (cx - 0.5 * sx, cy - 0.5 * sy, cz - 0.5 * sz)
    bounds_max = (cx + 0.5 * sx, cy + 0.5 * sy, cz + 0.5 * sz)
    N = length(r)
    mean_r = mean(r)
    success_tol = rel_tol * mean_r

    println("Searching for centralized void. Target gap: $(success_tol) m")

    for attempt in 1:restarts
        p = if attempt == 1
            bed_center
        else
            (
                bounds_min[1] + rand() * (bounds_max[1] - bounds_min[1]),
                bounds_min[2] + rand() * (bounds_max[2] - bounds_min[2]),
                bounds_min[3] + rand() * (bounds_max[3] - bounds_min[3]),
            )
        end

        step_size = mean_r * 0.1

        for _ in 1:steps
            min_surf_dist = Inf
            nearest_idx = 0

            @inbounds for j in 1:N
                dx = p[1] - X[1, j]
                dy = p[2] - X[2, j]
                dz = p[3] - X[3, j]
                d2 = dx^2 + dy^2 + dz^2
                ds = sqrt(d2) - r[j]
                if ds < min_surf_dist
                    min_surf_dist = ds
                    nearest_idx = j
                end
            end

            if min_surf_dist > success_tol
                println("Found valid start point via gradient ascent: $p (dist=$min_surf_dist m)")
                return p
            end

            c = (X[1, nearest_idx], X[2, nearest_idx], X[3, nearest_idx])
            vec = sub3(p, c)
            norm_vec = sqrt(dot3(vec, vec))
            dir_repulse = if norm_vec < 1e-12
                d_rand = (rand() - 0.5, rand() - 0.5, rand() - 0.5)
                normalize3(d_rand)
            else
                mul3(vec, 1.0 / norm_vec)
            end

            vec_center = sub3(bed_center, p)
            dist_center = sqrt(dot3(vec_center, vec_center))
            dir_attract = dist_center > 1e-9 ? mul3(vec_center, 1.0 / dist_center) : (0.0, 0.0, 0.0)

            combined_dir = add3(dir_repulse, mul3(dir_attract, lambda))
            final_dir = normalize3(combined_dir)

            factor = 1.0
            if min_surf_dist < -0.5 * mean_r
                factor = 2.0
            elseif min_surf_dist > -0.1 * mean_r
                factor = 0.5
            end

            move = mul3(final_dir, step_size * factor)
            p = add3(p, move)
            p = (
                clamp(p[1], bounds_min[1], bounds_max[1]),
                clamp(p[2], bounds_min[2], bounds_max[2]),
                clamp(p[3], bounds_min[3], bounds_max[3]),
            )
            step_size *= 0.995
        end
    end

    error("Gradient optimization failed to find void space > $(success_tol) m.")
end
