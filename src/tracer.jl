# ============================================================
# Tracer
# ============================================================
@inline function reflect_specular(d::SVector{3,Float64}, n::SVector{3,Float64})
    return d - 2.0 * dot3(d, n) * n
end

const _bbox_warned = Ref(false)

struct RayTraceResult
    escaped::Bool
    total_length::Float64
    nsteps::Int
    path::Vector{SVector{3,Float64}}   # polyline vertices (start, hits..., exit)
end

function start_outside_spheres_aabb(p::SVector{3,Float64},
                                   d::SVector{3,Float64},
                                   spheres::SphereBVH;
                                   tol::Float64 = 1e-12,
                                   eps_hit::Float64 = 1e-12)

    px, py, pz = p
    tmax = 0.0
    n_inside = 0

    @inbounds for i in eachindex(spheres.radii)
        c = spheres.centres[i]
        r = spheres.radii[i]

        dx = px - c[1];  adx = abs(dx)
        adx > r && continue
        dy = py - c[2];  ady = abs(dy)
        ady > r && continue
        dz = pz - c[3];  adz = abs(dz)
        adz > r && continue

        # Only now do the actual sphere check
        r_eff = r - tol
        r_eff <= 0 && continue
        if (dx*dx + dy*dy + dz*dz) < (r_eff*r_eff)
            n_inside += 1
            t = ray_sphere_exit_distance(p, d, c, r; eps=eps_hit)
            if t !== nothing && t > tmax
                tmax = t
            end
        end
    end

    return n_inside, tmax
end

function trace_ray_geometric(p0::SVector{3,Float64},
                             d0::SVector{3,Float64},
                             surfaces::AbstractVector{<:TriangleSurface},
                             spheres::SphereBVH;
                             eps::Float64=1e-10,
                             tri_eps::Float64=1e-12,
                             sph_eps::Float64=1e-12,
                             max_steps::Int=1_000_000,
                             max_length::Float64=Inf,
                             record_path::Bool=true,
                             allow_start_inside::Bool=true,
                             count_start_solid::Bool=false)

    surface_bvh = build_surface_bvh(surfaces)
    ray_escaped = false

    invn = 1.0 / sqrt(dot3(d0, d0))
    d = d0 * invn
    p = p0

    path = record_path ? SVector{3,Float64}[p0] : SVector{3,Float64}[]
    total = 0.0

    # --- NEW: if starting inside spheres, move to outside all of them (no reflection) ---
    if allow_start_inside
        n_inside, tmax = start_outside_spheres_aabb(p, d, spheres; tol=1e-12, eps_hit=sph_eps)
        if n_inside > 0
            p = p + (tmax + eps) * d
            if count_start_solid
                total += tmax
            end
            record_path && push!(path, p)
        end
    end

    for step in 1:max_steps
        sh = nearest_sphere_hit(p, d, spheres; eps=sph_eps)
        mh = nearest_surface_hit(p, d, surface_bvh; eps=tri_eps)
        tb = ray_aabb_intersect(p, d, surface_bvh.bbox_mins, surface_bvh.bbox_maxs; eps=eps)

        ts = sh === nothing ? Inf : sh[1]
        tm = mh === nothing ? Inf : mh[1]
        tbox = tb === nothing ? Inf : tb

        if isinf(ts) && isinf(tm) && isinf(tbox)
            # If the vessel is truly closed and you're inside it, this shouldn't happen.
            return RayTraceResult(false, total, step, path)
        end

        t = min(ts, tm, tbox)
        p_hit = p + t * d
        total += t

        if record_path
            push!(path, p_hit)
        end

        if total > max_length
            return RayTraceResult(false, total, step, path)
        end

        if tbox <= tm && tbox <= ts
            # Hit bounding box sink
            ray_escaped = true
            if !_bbox_warned[]
                @warn "Ray hit bounding box sink; mesh may not be watertight."
                _bbox_warned[] = true
            end
            return RayTraceResult(true, total, step, path)
        elseif tm < ts
            # Hit triangle surface
            _, _, nhat, kind = mh
            if dot3(d, nhat) > 0.0
                nhat = -nhat
            end

            if kind isa Sink
                return RayTraceResult(true, total, step, path)
            end

            d = reflect_specular(d, nhat)
            d *= 1.0 / sqrt(dot3(d, d))
            p = p_hit + eps * nhat
        else
            # Hit sphere => reflect and continue (robust)
            _, idx = sh
            c = spheres.centres[idx]
            r = spheres.radii[idx]

            # Robust unit normal from centre->hit
            nhat = p_hit - c
            invlen = 1.0 / sqrt(dot3(nhat, nhat))
            nhat = nhat * invlen

            # Specular reflection about unit normal
            d = reflect_specular(d, nhat)

            # Renormalise direction to avoid drift
            d *= 1.0 / sqrt(dot3(d, d))

            # Reposition point to be just outside the sphere
            eps_rel = max(eps, 1e-9 * r)
            p = c + (r + eps_rel) * nhat
        end

    end

    return RayTraceResult(false, total, max_steps, path)
end


function trace_ray_geometric(p0::SVector{3,Float64},
    d0::SVector{3,Float64},
    surfaces::AbstractVector{<:TriangleSurface},
    particles::PolyhedralBVH;
    eps::Float64=1e-10,
    tri_eps::Float64=1e-12,
    part_eps::Float64=1e-12,
    max_steps::Int=1_000_000,
    max_length::Float64=Inf,
    record_path::Bool=true,
    allow_start_inside::Bool=true,
    count_start_solid::Bool=false)

    surface_bvh = build_surface_bvh(surfaces)

    invn = 1.0 / sqrt(dot3(d0, d0))
    d = d0 * invn
    p = p0

    path = record_path ? SVector{3,Float64}[p0] : SVector{3,Float64}[]
    total = 0.0

    # Note: start_outside_spheres logic is tricky for polyhedra. 
    # For now we can assume ray is inside.
    _ = allow_start_inside
    _ = count_start_solid

    for step in 1:max_steps
        ph = nearest_polyh_hit(p, d, particles; eps=part_eps) # (t, idx, n)
        mh = nearest_surface_hit(p, d, surface_bvh; eps=tri_eps)
        tb = ray_aabb_intersect(p, d, surface_bvh.bbox_mins, surface_bvh.bbox_maxs; eps=eps)

        ts = ph === nothing ? Inf : ph[1]
        tm = mh === nothing ? Inf : mh[1]
        tbox = tb === nothing ? Inf : tb

        if isinf(ts) && isinf(tm) && isinf(tbox)
            return RayTraceResult(false, total, step, path)
        end

        t = min(ts, tm, tbox)
        p_hit = p + t * d
        total += t

        if record_path
            push!(path, p_hit)
        end

        if total > max_length
            return RayTraceResult(false, total, step, path)
        end

        if tbox <= tm && tbox <= ts
            # Hit bounding box sink
            return RayTraceResult(true, total, step, path)
        elseif tm < ts
            # Hit triangle surface (wall/mirror)
            _, _, nhat, kind = mh
            if dot3(d, nhat) > 0.0
                nhat = -nhat
            end

            if kind isa Sink
                return RayTraceResult(true, total, step, path)
            end

            d = reflect_specular(d, nhat)
            # Normalize manually
            d *= 1.0 / sqrt(dot3(d, d))
            p = p_hit + eps * nhat
        else
            # Hit Polyhedron => reflect
            _, idx, nhat = ph

            # Ensure normal faces against ray
            if dot3(d, nhat) > 0.0
                nhat = -nhat
            end

            # Specular reflection
            d = reflect_specular(d, nhat)
            # Normalize manually
            d *= 1.0 / sqrt(dot3(d, d))

            # Reposition point just outside
            p = p_hit + eps * nhat
        end
    end

    return RayTraceResult(false, total, max_steps, path)
end

function find_void_rrhc(
    X::Matrix{Float64}, r::Vector{Float64};
    restarts=50, steps=2_000, rel_tol=1e-2,
    lambda=0.05
)
    # This is more robust than extrema for defining the "dense core"
    cx, cy, cz = mean(X[1, :]), mean(X[2, :]), mean(X[3, :])
    sx, sy, sz = stddev(X[1, :]), stddev(X[2, :]), stddev(X[3, :])

    bed_center = SVector(cx, cy, cz)
    # Restrict search to core 1-sigma region
    bounds_min = SVector(cx - 0.5 * sx, cy - 0.5 * sy, cz - 0.5 * sz)
    bounds_max = SVector(cx + 0.5 * sx, cy + 0.5 * sy, cz + 0.5 * sz)
    N = length(r)
    mean_r = mean(r)
    success_tol = rel_tol * mean_r

    println("Searching for centralized void. Target gap: $(success_tol) m")

    for attempt in 1:restarts
        # Start exactly at center or very close to it
        if attempt == 1
            p = bed_center
        else
            px = bounds_min[1] + rand() * (bounds_max[1] - bounds_min[1])
            py = bounds_min[2] + rand() * (bounds_max[2] - bounds_min[2])
            pz = bounds_min[3] + rand() * (bounds_max[3] - bounds_min[3])
            p = SVector(px, py, pz)
        end

        step_size = mean_r * 0.1

        for s in 1:steps
            # 1. Find nearest particle surface (Repulsive term)
            min_surf_dist = Inf
            nearest_idx = 0

            for j in 1:N
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

            # Check if outside of particles by stated tolerance
            if min_surf_dist > success_tol
                println("Found valid start point via gradient ascent: $p (dist=$min_surf_dist m)")
                return p
            end

            # 2. Gradient of Repulsion (Away from surface)
            c = SVector(X[1, nearest_idx], X[2, nearest_idx], X[3, nearest_idx])
            vec = p - c
            norm_vec = sqrt(dot3(vec, vec))

            dir_repulse = if norm_vec < 1e-12
                d_rand = SVector(rand() - 0.5, rand() - 0.5, rand() - 0.5)
                d_rand / sqrt(dot3(d_rand, d_rand))
            else
                vec * (1.0 / norm_vec)
            end

            # 3. Gradient of Attraction (Towards Center)
            vec_center = bed_center - p
            dist_center = sqrt(dot3(vec_center, vec_center))
            if dist_center > 1e-9
                dir_attract = vec_center * (1.0 / dist_center)
            else
                dir_attract = SVector(0.0, 0.0, 0.0)
            end

            # Combine forces: push out of particle vs pull to center.
            combined_dir = dir_repulse + lambda * dir_attract

            # Re-normalise direction
            len_comb = sqrt(dot3(combined_dir, combined_dir))
            final_dir = combined_dir * (1.0 / len_comb)

            # Adaptive step logic
            factor = 1.0
            if min_surf_dist < -0.5 * mean_r
                factor = 2.0
            elseif min_surf_dist > -0.1 * mean_r
                factor = 0.5
            end

            move = final_dir * (step_size * factor)
            p = p + move

            # Clamp not strictly needed if attraction works, but good for safety
            p = SVector(
                clamp(p[1], bounds_min[1], bounds_max[1]),
                clamp(p[2], bounds_min[2], bounds_max[2]),
                clamp(p[3], bounds_min[3], bounds_max[3])
            )

            step_size *= 0.995
        end
    end

    error("Gradient optimization failed to find void space > $(success_tol) m.")
end



