# ============================================================
# BVH structures and builders
# ============================================================
struct MeshBVH
    bvh::BVH
    verts::Vector{SVector{3,Float64}}
    tris::Vector{NTuple{3,Int32}}
end

struct SurfaceBVH
    bvh::BVH
    verts::Vector{SVector{3,Float64}}
    tris::Vector{NTuple{3,Int32}}
    kinds::Vector{SurfaceKind}
    bbox_mins::SVector{3,Float64}
    bbox_maxs::SVector{3,Float64}
end

struct SphereBVH
    bvh::BVH
    centres::Vector{SVector{3,Float64}}
    radii::Vector{Float64}
end

struct PolyhedralBVH
    bvh::BVH
    centres::Vector{SVector{3,Float64}}
    radii::Vector{Float64}
    orientations::Vector{SMatrix{3,3,Float64,9}}
    polyh_mesh::TriangleMesh
end

function build_mesh_bvh(tm::TriangleMesh)::MeshBVH
    V = tm.vertices
    C = tm.connectivity
    verts = [SVector{3,Float64}(V[1,i], V[2,i], V[3,i]) for i in 1:size(V,2)]
    tris  = [(Int32(C[1,j]), Int32(C[2,j]), Int32(C[3,j])) for j in 1:size(C,2)]

    tri_centres = Vector{SVector{3,Float64}}(undef, length(tris))
    tri_radii   = Vector{Float64}(undef, length(tris))
    for (j,(i1,i2,i3)) in enumerate(tris)
        v0, v1, v2 = verts[i1], verts[i2], verts[i3]
        c = (v0 + v1 + v2) / 3.0
        rtri = max(
            sqrt(dot3(c - v0, c - v0)),
            max(
                sqrt(dot3(c - v1, c - v1)),
                sqrt(dot3(c - v2, c - v2)),
            ),
        )
        tri_centres[j] = c
        tri_radii[j]   = rtri
    end

    leaves = [BSphere{Float64}(tri_centres[j], tri_radii[j]) for j in eachindex(tris)]
    bvh = BVH(leaves, BBox{Float64})
    MeshBVH(bvh, verts, tris)
end

function build_surface_bvh(surfaces::AbstractVector{<:TriangleSurface})::SurfaceBVH
    isempty(surfaces) && throw(ArgumentError("build_surface_bvh: surfaces must not be empty"))

    meshes = [s.mesh for s in surfaces]
    union_mesh = union_trianglemesh(meshes)
    bbox_mins, bbox_maxs = compute_bounds(union_mesh)
    center = 0.5 * (bbox_mins + bbox_maxs)
    half = 0.5 * (bbox_maxs - bbox_mins)
    half = 1.1 * half
    bbox_mins = center - half
    bbox_maxs = center + half

    verts = Vector{SVector{3,Float64}}()
    tris = Vector{NTuple{3,Int32}}()
    kinds = Vector{SurfaceKind}()
    nverts_total = 0

    for surface in surfaces
        tm = surface.mesh
        V = tm.vertices
        C = tm.connectivity
        Nv = size(V, 2)
        Nt = size(C, 2)

        @inbounds for i in 1:Nv
            push!(verts, SVector{3,Float64}(V[1,i], V[2,i], V[3,i]))
        end

        off = Int32(nverts_total)
        @inbounds for j in 1:Nt
            push!(tris, (Int32(C[1,j]) + off, Int32(C[2,j]) + off, Int32(C[3,j]) + off))
            push!(kinds, surface.kind)
        end

        nverts_total += Nv
    end

    tri_centres = Vector{SVector{3,Float64}}(undef, length(tris))
    tri_radii   = Vector{Float64}(undef, length(tris))
    @inbounds for (j,(i1,i2,i3)) in enumerate(tris)
        v0, v1, v2 = verts[i1], verts[i2], verts[i3]
        c = (v0 + v1 + v2) / 3.0
        rtri = max(
            sqrt(dot3(c - v0, c - v0)),
            max(
                sqrt(dot3(c - v1, c - v1)),
                sqrt(dot3(c - v2, c - v2)),
            ),
        )
        tri_centres[j] = c
        tri_radii[j]   = rtri
    end

    leaves = [BSphere{Float64}(tri_centres[j], tri_radii[j]) for j in eachindex(tris)]
    bvh = BVH(leaves, BBox{Float64})
    return SurfaceBVH(bvh, verts, tris, kinds, bbox_mins, bbox_maxs)
end

build_surface_bvh(surface::TriangleSurface)::SurfaceBVH = build_surface_bvh([surface])

function build_sphere_bvh(X::AbstractMatrix{<:Real}, r::AbstractVector{<:Real})::SphereBVH
    @assert size(X,1) == 3 "X must be 3xN"
    n = size(X,2)
    @assert length(r) == n "r must have length N"

    centres = Vector{SVector{3,Float64}}(undef, n)
    radii   = Vector{Float64}(undef, n)
    leaves  = Vector{BSphere{Float64}}(undef, n)

    @inbounds for i in 1:n
        c = SVector{3,Float64}(float(X[1,i]), float(X[2,i]), float(X[3,i]))
        ri = float(r[i])
        centres[i] = c
        radii[i]   = ri
        leaves[i]  = BSphere{Float64}(c, ri)  # tight bounding sphere
    end

    bvh = BVH(leaves, BBox{Float64})
    return SphereBVH(bvh, centres, radii)
end


function build_polyh_bvh(
    polyh_mesh::TriangleMesh,
    X::AbstractMatrix{<:Real},
    r::AbstractVector{<:Real},
    orients::AbstractMatrix{<:Real} # Assuming 3xN Euler angles
)::PolyhedralBVH

    @assert size(X, 1) == 3 "X must be 3xN"
    n = size(X, 2)
    @assert length(r) == n "r must have length N"
    @assert size(orients, 1) == 3 "orients must be 3xN (Euler angles)"
    @assert size(orients, 2) == n "orients must have N columns"

    centres = Vector{SVector{3,Float64}}(undef, n)
    radii = Vector{Float64}(undef, n)
    orientations = Vector{SMatrix{3,3,Float64,9}}(undef, n)
    leaves = Vector{BBox{Float64}}(undef, n)

    # Pre-allocate bounds for loop efficiency
    V_local = polyh_mesh.vertices
    Nv = size(V_local, 2)

    @inbounds for i in 1:n
        c = SVector{3,Float64}(float(X[1, i]), float(X[2, i]), float(X[3, i]))
        ri = float(r[i])

        centres[i] = c
        radii[i] = ri

        # Compute and store global rotation matrix
        r_mat = get_rotation_matrix(view(orients, :, i), units=u"°")
        orientations[i] = r_mat
        sf = get_scale_factor(polyh_mesh, Float32(ri))

        # Initialize bounds with the first transformed vertex
        v1 = SVector(V_local[1, 1], V_local[2, 1], V_local[3, 1])
        v1_trans = (r_mat * v1) * sf + c

        min_b = v1_trans
        max_b = v1_trans

        for k in 2:Nv
            v = SVector(V_local[1, k], V_local[2, k], V_local[3, k])
            v_trans = (r_mat * v) * sf + c

            min_b = min.(min_b, v_trans)
            max_b = max.(max_b, v_trans)
        end

        leaves[i] = BBox{Float64}(min_b, max_b)
    end

    bvh = BVH(leaves, BBox{Float64})
    return PolyhedralBVH(bvh, centres, radii, orientations, polyh_mesh)
end
