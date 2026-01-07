# ============================================================
# Ray/geometry intersections
# ============================================================
function ray_triangle_intersect(p::SVector{3,Float64},
                                d::SVector{3,Float64},
                                v0::SVector{3,Float64},
                                v1::SVector{3,Float64},
                                v2::SVector{3,Float64};
                                eps=1e-12)
    e1 = v1 - v0
    e2 = v2 - v0
    h  = cross3(d, e2)
    a  = dot3(e1, h)
    if abs(a) < eps
        return nothing
    end
    f = 1.0 / a
    s = p - v0
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

function nearest_surface_hit(p::SVector{3,Float64},
                             d::SVector{3,Float64},
                             surface::SurfaceBVH;
                             eps=1e-12)
    P = to_mat3x1(p)
    D = to_mat3x1(d)
    trav = traverse_rays(surface.bvh, P, D)

    tmin = Inf
    imin = 0

    @inbounds for (leaf_idx, ray_idx) in trav.contacts
        ray_idx == 1 || continue
        i1, i2, i3 = surface.tris[leaf_idx]
        v0, v1, v2 = surface.verts[i1], surface.verts[i2], surface.verts[i3]
        t = ray_triangle_intersect(p, d, v0, v1, v2; eps=eps)
        if t !== nothing && t < tmin
            tmin = t
            imin = leaf_idx
        end
    end

    if imin == 0
        return nothing
    end

    i1, i2, i3 = surface.tris[imin]
    v0, v1, v2 = surface.verts[i1], surface.verts[i2], surface.verts[i3]
    n = cross3(v1 - v0, v2 - v0)
    invn = 1.0 / sqrt(dot3(n, n))
    n = n * invn
    kind = surface.kinds[imin]
    return (tmin, imin, n, kind)
end

@inline function ray_sphere_exit_distance(p::SVector{3,Float64},
                                          d::SVector{3,Float64},
                                          c::SVector{3,Float64},
                                          r::Float64;
                                          eps=1e-12)
    # Assumes d is unit. Returns the forward distance to exit the sphere.
    m = p - c
    b = dot3(m, d)
    c0 = dot3(m, m) - r*r
    disc = b*b - c0
    if disc < 0.0
        return nothing
    end
    s = sqrt(disc)
    t_exit = -b + s
    return t_exit > eps ? t_exit : nothing
end

function ray_aabb_intersect(p::SVector{3,Float64},
                            d::SVector{3,Float64},
                            mins::SVector{3,Float64},
                            maxs::SVector{3,Float64};
                            eps=1e-12)
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

@inline function ray_sphere_intersect(p::SVector{3,Float64},
                                      d::SVector{3,Float64},
                                      c::SVector{3,Float64},
                                      r::Float64;
                                      eps=1e-12)
    # Solve ||p + t d - c||^2 = r^2
    m = p - c
    b = dot3(m, d)
    c0 = dot3(m, m) - r*r
    disc = b*b - c0
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

function nearest_sphere_hit(p::SVector{3,Float64},
                            d::SVector{3,Float64},
                            sbvh::SphereBVH;
                            eps=1e-12)
    P = to_mat3x1(p)
    D = to_mat3x1(d)
    trav = traverse_rays(sbvh.bvh, P, D)

    tmin = Inf
    imin = 0

    @inbounds for (leaf_idx, ray_idx) in trav.contacts
        ray_idx == 1 || continue
        c = sbvh.centres[leaf_idx]
        r = sbvh.radii[leaf_idx]
        t = ray_sphere_intersect(p, d, c, r; eps=eps)
        if t !== nothing && t < tmin
            tmin = t
            imin = leaf_idx
        end
    end

    return imin == 0 ? nothing : (tmin, imin)
end
