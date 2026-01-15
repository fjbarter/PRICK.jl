 # Optional visualisation utitilies using GLMakie
 
function _glmakie()
    if isdefined(Main, :GLMakie)
        return getfield(Main, :GLMakie)
    end
    error("GLMakie is not loaded. Run `using GLMakie` before calling visualisation functions.")
end
function path_to_matrix(path::Vector{SVector{3,Float64}})
    K = length(path)
    M = Matrix{Float64}(undef, 3, K)
    @inbounds for k in 1:K
        M[1,k] = path[k][1]
        M[2,k] = path[k][2]
        M[3,k] = path[k][3]
    end
    return M
end

# Convert (3,N) + (3,M) to a GeometryBasics.Mesh
function to_geometrybasics_mesh(tm::TriangleMesh)
    V = tm.vertices
    C = tm.connectivity

    points = [GeometryBasics.Point3f(Float32(V[1,i]), Float32(V[2,i]), Float32(V[3,i]))
              for i in 1:size(V,2)]
    faces  = [GeometryBasics.TriangleFace(Int(C[1,j]), Int(C[2,j]), Int(C[3,j]))
              for j in 1:size(C,2)]

    return GeometryBasics.Mesh(points, faces)
end

# Make a sphere surface mesh centred at `c` with radius `r`
function sphere_mesh(c::NTuple{3,Float32}, r::Float32; nθ::Int=24, nϕ::Int=16)
    cx, cy, cz = c
    θs = range(0f0, 2f0*pi, length=nθ)
    ϕs = range(0f0, pi,      length=nϕ)

    pts = Vector{GeometryBasics.Point3f}(undef, nθ*nϕ)
    @inbounds for j in 1:nϕ
        ϕ = ϕs[j]
        sϕ = sin(ϕ); cϕ = cos(ϕ)
        for i in 1:nθ
            θ = θs[i]
            x = cx + r * (cos(θ) * sϕ)
            y = cy + r * (sin(θ) * sϕ)
            z = cz + r * (cϕ)
            pts[(j-1)*nθ + i] = GeometryBasics.Point3f(x, y, z)
        end
    end

    # Triangulate grid on (θ,ϕ)
    faces = GeometryBasics.TriangleFace{Int}[]
    @inbounds for j in 1:(nϕ-1)
        for i in 1:(nθ-1)
            a = (j-1)*nθ + i
            b = a + 1
            c_ = j*nθ + i
            d = c_ + 1
            push!(faces, GeometryBasics.TriangleFace(a, c_, b))
            push!(faces, GeometryBasics.TriangleFace(b, c_, d))
        end
        # wrap seam at θ = 2π
        a = (j-1)*nθ + nθ
        b = (j-1)*nθ + 1
        c_ = j*nθ + nθ
        d = j*nθ + 1
        push!(faces, GeometryBasics.TriangleFace(a, c_, b))
        push!(faces, GeometryBasics.TriangleFace(b, c_, d))
    end

    return GeometryBasics.Mesh(pts, faces)
end

function visualise_trace(res::RayTraceResult;
                         X::AbstractMatrix,
                         radii::AbstractVector,
                         vessel_tm::Union{Nothing,TriangleMesh}=nothing,
                         sphere_res::Int=20,
                         show_vessel::Bool=true,
                         show_spheres::Bool=true)
    gm = _glmakie()
    fig = gm.Figure()
    ax = gm.Axis3(fig[1,1], aspect=:data)

    # --- vessel ---
    if show_vessel && vessel_tm !== nothing
        m = to_geometrybasics_mesh(vessel_tm)
        gm.mesh!(ax, m;
            color = (:gray, 0.15),
            transparency = true,
            shading = gm.NoShading,
        )
    end

    # --- spheres ---
    if show_spheres
        idxs = spheres_near_path_indices(X, radii, res.path; margin=2e-3, path_stride=20)
        # If you want to see how many you kept:
        println("Plotting $(length(idxs)) / $(size(X,2)) spheres near the ray")

        for i in idxs
            c = (Float32(X[1,i]), Float32(X[2,i]), Float32(X[3,i]))
            r = Float32(radii[i])
            sm = sphere_mesh(c, r; nθ=sphere_res, nϕ=max(8, sphere_res ÷ 2))
            gm.mesh!(ax, sm, color = (:dodgerblue, 0.1))
        end
    end

    # --- path ---
    if !isempty(res.path)
        pts = [GeometryBasics.Point3f(Float32(p[1]), Float32(p[2]), Float32(p[3]))
            for p in res.path]

        gm.lines!(ax, pts;
            linewidth = 4,
            overdraw = true,        # draw on top of everything
        )
        gm.scatter!(ax, [pts[1]];  markersize=12, color=:red,    overdraw=true)
        gm.scatter!(ax, [pts[end]]; markersize=12, color=:purple, overdraw=true)
    end

    return fig
end


# AABB of a set of points (Vector{SVector{3,Float64}})
function path_aabb(path::Vector{SVector{3,Float64}})
    p0 = path[1]
    minx = p0[1]; maxx = p0[1]
    miny = p0[2]; maxy = p0[2]
    minz = p0[3]; maxz = p0[3]
    @inbounds for i in 2:length(path)
        p = path[i]
        x=p[1]; y=p[2]; z=p[3]
        if x < minx; minx = x; end
        if x > maxx; maxx = x; end
        if y < miny; miny = y; end
        if y > maxy; maxy = y; end
        if z < minz; minz = z; end
        if z > maxz; maxz = z; end
    end
    return (SVector{3,Float64}(minx,miny,minz), SVector{3,Float64}(maxx,maxy,maxz))
end

# Return indices of spheres whose centres come within (radius + margin) of any sampled path point.
function spheres_near_path_indices(X::AbstractMatrix{<:Real},
                                  radii::AbstractVector{<:Real},
                                  path::Vector{SVector{3,Float64}};
                                  margin::Float64 = 2e-3,
                                  path_stride::Int = 20)

    N = size(X,2)
    @assert size(X,1) == 3
    @assert length(radii) == N
    isempty(path) && return Int[]

    # Downsample path points
    idxs = 1:path_stride:length(path)
    P = path[idxs]

    # AABB cull (expanded by max radius + margin)
    rmax = float(maximum(radii))
    mins, maxs = path_aabb(P)
    mins = mins .- (rmax + margin)
    maxs = maxs .+ (rmax + margin)

    keep = Int[]
    @inbounds for i in 1:N
        cx = float(X[1,i]); cy = float(X[2,i]); cz = float(X[3,i])
        # quick AABB reject
        if cx < mins[1] || cx > maxs[1] || cy < mins[2] || cy > maxs[2] || cz < mins[3] || cz > maxs[3]
            continue
        end

        ri = float(radii[i])
        thresh2 = (ri + margin)^2

        # distance to nearest sampled point (tube around polyline)
        d2min = Inf
        for p in P
            dx = cx - p[1]; dy = cy - p[2]; dz = cz - p[3]
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < d2min
                d2min = d2
                d2min < thresh2 && break
            end
        end

        if d2min < thresh2
            push!(keep, i)
        end
    end

    return keep
end
