# Ray/geometry intersections
# ============================================================

@inline function ray_triangle_intersect(
    p::NTuple{3,Float64},
    d::NTuple{3,Float64},
    v0::NTuple{3,Float64},
    v1::NTuple{3,Float64},
    v2::NTuple{3,Float64};
    eps::Float64=1e-12,
)
    e1 = sub3(v1, v0)
    e2 = sub3(v2, v0)
    h = cross3(d, e2)
    a = dot3(e1, h)
    if abs(a) < eps
        return nothing
    end
    f = 1.0 / a
    s = sub3(p, v0)
    u = f * dot3(s, h)
    if (u < 0.0) || (u > 1.0)
        return nothing
    end
    q = cross3(s, e1)
    v = f * dot3(d, q)
    if (v < 0.0) || (u + v > 1.0)
        return nothing
    end
    t = f * dot3(e2, q)
    return t > eps ? t : nothing
end

@inline function triangle_unit_normal(
    v0::NTuple{3,Float64},
    v1::NTuple{3,Float64},
    v2::NTuple{3,Float64},
)
    n = cross3(sub3(v1, v0), sub3(v2, v0))
    invn = inv(sqrt(dot3(n, n)))
    return mul3(n, invn)
end

@inline function ray_sphere_exit_distance(
    p::NTuple{3,Float64},
    d::NTuple{3,Float64},
    c::NTuple{3,Float64},
    r::Float64;
    eps::Float64=1e-12,
)
    m = sub3(p, c)
    b = dot3(m, d)
    c0 = dot3(m, m) - r * r
    disc = b * b - c0
    if disc < 0.0
        return nothing
    end
    s = sqrt(disc)
    t_exit = -b + s
    return t_exit > eps ? t_exit : nothing
end

function ray_aabb_intersect(
    p::NTuple{3,Float64},
    d::NTuple{3,Float64},
    mins::NTuple{3,Float64},
    maxs::NTuple{3,Float64};
    eps::Float64=1e-12,
)
    tmin = -Inf
    tmax = Inf
    @inbounds for k in 1:3
        dk = d[k]
        if abs(dk) < eps
            if p[k] < mins[k] || p[k] > maxs[k]
                return nothing
            end
        else
            invd = 1.0 / dk
            t1 = (mins[k] - p[k]) * invd
            t2 = (maxs[k] - p[k]) * invd
            if t1 > t2
                t1, t2 = t2, t1
            end
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax
                return nothing
            end
        end
    end

    t = tmin > eps ? tmin : tmax
    return t > eps ? t : nothing
end

@inline function ray_sphere_intersect(
    p::NTuple{3,Float64},
    d::NTuple{3,Float64},
    c::NTuple{3,Float64},
    r::Float64;
    eps::Float64=1e-12,
)
    m = sub3(p, c)
    b = dot3(m, d)
    c0 = dot3(m, m) - r * r
    disc = b * b - c0
    if disc < 0.0
        return nothing
    end
    s = sqrt(disc)
    t1 = -b - s
    if t1 > eps
        return t1
    end
    t2 = -b + s
    return t2 > eps ? t2 : nothing
end

@inline function rotate3(R::SMatrix{3,3,Float64,9}, v::NTuple{3,Float64})
    return (
        R[1, 1] * v[1] + R[1, 2] * v[2] + R[1, 3] * v[3],
        R[2, 1] * v[1] + R[2, 2] * v[2] + R[2, 3] * v[3],
        R[3, 1] * v[1] + R[3, 2] * v[2] + R[3, 3] * v[3],
    )
end

@inline function rotate3_transpose(R::SMatrix{3,3,Float64,9}, v::NTuple{3,Float64})
    return (
        R[1, 1] * v[1] + R[2, 1] * v[2] + R[3, 1] * v[3],
        R[1, 2] * v[1] + R[2, 2] * v[2] + R[3, 2] * v[3],
        R[1, 3] * v[1] + R[2, 3] * v[2] + R[3, 3] * v[3],
    )
end

@inline function ray_polyh_intersect(
    p::NTuple{3,Float64},
    d::NTuple{3,Float64},
    pbvh::PolyhedralBVH,
    idx::Int;
    eps::Float64=1e-12,
)
    c = pbvh.centres[idx]
    c_tuple = (c[1], c[2], c[3])
    R = pbvh.orientations[idx]
    s = pbvh.scales[idx]

    invs = 1.0 / s
    p_local = mul3(rotate3_transpose(R, sub3(p, c_tuple)), invs)
    d_local = mul3(rotate3_transpose(R, d), invs)

    t_best = Inf
    n_local_best = (0.0, 0.0, 0.0)
    tris = pbvh.local_tris
    verts = pbvh.local_verts
    normals = pbvh.local_normals

    @inbounds for j in eachindex(tris)
        i1, i2, i3 = tris[j]
        v0 = verts[i1]
        v1 = verts[i2]
        v2 = verts[i3]
        t = ray_triangle_intersect(p_local, d_local, v0, v1, v2; eps=eps)
        t === nothing && continue
        if t < t_best
            t_best = t
            n_local_best = normals[j]
        end
    end

    isfinite(t_best) || return nothing

    n_world = normalize3(rotate3(R, n_local_best))
    return (t_best, n_world)
end

function update_nearest_sphere_hits!(
    rb::RayBatchBuffer{Float64},
    sbvh::SphereBVH,
    traversal::ActiveTraversalCache;
    eps::Float64=1e-12,
)
    P = rb.positions
    D = rb.directions
    active = rb.active
    sphere_t = rb.sphere_t
    sphere_idx = rb.sphere_idx

    @inbounds for icontact in 1:traversal.num_contacts
        leaf_idx, ray_idx_raw = traversal.contacts[icontact]
        ray_idx = Int(ray_idx_raw)
        active[ray_idx] || continue

        p = (P[1, ray_idx], P[2, ray_idx], P[3, ray_idx])
        d = (D[1, ray_idx], D[2, ray_idx], D[3, ray_idx])
        c = sbvh.centres[Int(leaf_idx)]
        r = sbvh.radii[Int(leaf_idx)]
        t = ray_sphere_intersect(p, d, c, r; eps=eps)
        t === nothing && continue

        tcur = sphere_t[ray_idx]
        idxcur = sphere_idx[ray_idx]
        if (t < tcur) || ((t == tcur) && (idxcur == 0 || Int(leaf_idx) < idxcur))
            sphere_t[ray_idx] = t
            sphere_idx[ray_idx] = Int(leaf_idx)
        end
    end
    return nothing
end

function update_nearest_polyh_hits!(
    rb::RayBatchBuffer{Float64},
    pbvh::PolyhedralBVH,
    traversal::ActiveTraversalCache;
    eps::Float64=1e-12,
)
    P = rb.positions
    D = rb.directions
    active = rb.active
    polyh_t = rb.polyh_t
    polyh_idx = rb.polyh_idx
    polyh_nx = rb.polyh_nx
    polyh_ny = rb.polyh_ny
    polyh_nz = rb.polyh_nz

    @inbounds for icontact in 1:traversal.num_contacts
        leaf_idx_raw, ray_idx_raw = traversal.contacts[icontact]
        ray_idx = Int(ray_idx_raw)
        active[ray_idx] || continue
        leaf_idx = Int(leaf_idx_raw)

        p = (P[1, ray_idx], P[2, ray_idx], P[3, ray_idx])
        d = (D[1, ray_idx], D[2, ray_idx], D[3, ray_idx])
        hit = ray_polyh_intersect(p, d, pbvh, leaf_idx; eps=eps)
        hit === nothing && continue
        t, n = hit

        tcur = polyh_t[ray_idx]
        idxcur = polyh_idx[ray_idx]
        if (t < tcur) || ((t == tcur) && (idxcur == 0 || leaf_idx < idxcur))
            polyh_t[ray_idx] = t
            polyh_idx[ray_idx] = leaf_idx
            polyh_nx[ray_idx] = n[1]
            polyh_ny[ray_idx] = n[2]
            polyh_nz[ray_idx] = n[3]
        end
    end
    return nothing
end

function update_nearest_surface_hits!(
    rb::RayBatchBuffer{Float64},
    surface::SurfaceBVH,
    traversal::ActiveTraversalCache;
    eps::Float64=1e-12,
)
    P = rb.positions
    D = rb.directions
    active = rb.active
    wall_t = rb.wall_t
    wall_idx = rb.wall_idx
    wall_nx = rb.wall_nx
    wall_ny = rb.wall_ny
    wall_nz = rb.wall_nz
    wall_sink = rb.wall_sink

    @inbounds for icontact in 1:traversal.num_contacts
        leaf_idx_raw, ray_idx_raw = traversal.contacts[icontact]
        ray_idx = Int(ray_idx_raw)
        active[ray_idx] || continue
        leaf_idx = Int(leaf_idx_raw)

        i1, i2, i3 = surface.tris[leaf_idx]
        v0 = surface.verts[i1]
        v1 = surface.verts[i2]
        v2 = surface.verts[i3]
        p = (P[1, ray_idx], P[2, ray_idx], P[3, ray_idx])
        d = (D[1, ray_idx], D[2, ray_idx], D[3, ray_idx])
        t = ray_triangle_intersect(p, d, v0, v1, v2; eps=eps)
        t === nothing && continue

        tcur = wall_t[ray_idx]
        idxcur = wall_idx[ray_idx]
        if (t < tcur) || ((t == tcur) && (idxcur == 0 || leaf_idx < idxcur))
            n = triangle_unit_normal(v0, v1, v2)
            wall_t[ray_idx] = t
            wall_idx[ray_idx] = leaf_idx
            wall_nx[ray_idx] = n[1]
            wall_ny[ray_idx] = n[2]
            wall_nz[ray_idx] = n[3]
            wall_sink[ray_idx] = surface.kinds[leaf_idx] isa Sink
        end
    end

    return nothing
end
