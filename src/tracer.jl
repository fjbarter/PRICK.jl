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
