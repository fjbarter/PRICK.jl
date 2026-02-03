# ============================================================
# Mesh definitions and construction helpers
# ============================================================
struct TriangleMesh
    vertices::Matrix{Float64}     # (3, Nv) stored in metres
    connectivity::Matrix{Int32}   # (3, Nt)
end

struct ParticleTriangleMesh
    vertices::Matrix{Float64}     # (3, Nv) stored in metres
    connectivity::Matrix{Int32}   # (3, Nt)
    volume::Float64                # volume of the mesh in m^3
end

ParticleTriangleMesh(v::Matrix{Float64}, c::Matrix{Int32}) =
    ParticleTriangleMesh(v, c, NaN)

abstract type SurfaceKind end
struct Mirror <: SurfaceKind end
struct Sink <: SurfaceKind end

struct TriangleSurface{K<:SurfaceKind}
    mesh::TriangleMesh
    kind::K
end

# Interpret a coordinate value `x` as having units `u_in` if it is unitless,
# then convert to metres and return a Float64.
@inline function to_metres(x, u_in)
    if x isa Unitful.Quantity
        return float(ustrip(uconvert(m, x)))
    else
        # unitless number: interpret as in `u_in`
        return float(ustrip(uconvert(m, x * u_in)))
    end
end

function convert_mesh(mesh::GeometryBasics.Mesh)
    # Taken from https://github.com/JuliaIO/MeshIO.jl/issues/67
    points = [Tuple(p) for p in Set(mesh.position)]
    indices = Dict(p => i for (i, p) in enumerate(points))
    connectivities = map(mesh) do el
        Meshes.connect(Tuple(indices[Tuple(p)] for p in el))
    end
    Meshes.SimpleMesh(points, connectivities)
end

function TriangleMesh(mesh::Meshes.SimpleMesh; units=u"m")
    c = mesh.vertices
    V = Matrix{Float64}(undef, 3, size(c, 1))

    @inbounds for (iv, xyz) in enumerate(c)
        V[1, iv] = to_metres(xyz.coords.x, units)
        V[2, iv] = to_metres(xyz.coords.y, units)
        V[3, iv] = to_metres(xyz.coords.z, units)
    end

    f = mesh.topology.connec
    C = Matrix{Int32}(undef, 3, size(f, 1))
    @inbounds for (ic, tri) in enumerate(f)
        ix, iy, iz = tri.indices
        C[1, ic] = Int32(ix)
        C[2, ic] = Int32(iy)
        C[3, ic] = Int32(iz)
    end

    return TriangleMesh(V, C)
end

function ParticleTriangleMesh(mesh::Meshes.SimpleMesh; units=u"m")
    c = mesh.vertices
    V = Matrix{Float64}(undef, 3, size(c, 1))

    @inbounds for (iv, xyz) in enumerate(c)
        V[1, iv] = to_metres(xyz.coords.x, units)
        V[2, iv] = to_metres(xyz.coords.y, units)
        V[3, iv] = to_metres(xyz.coords.z, units)
    end

    f = mesh.topology.connec
    C = Matrix{Int32}(undef, 3, size(f, 1))
    @inbounds for (ic, tri) in enumerate(f)
        ix, iy, iz = tri.indices
        C[1, ic] = Int32(ix)
        C[2, ic] = Int32(iy)
        C[3, ic] = Int32(iz)
    end

    # Compute volume using divergence theorem
    vol = 0.0
    # Compute signed volume of tetrahedrons formed by each triangle and the origin
    @inbounds for i in 1:size(C, 2)
        idx1 = C[1, i]
        idx2 = C[2, i]
        idx3 = C[3, i]

        # Vertices
        p1x, p1y, p1z = V[1, idx1], V[2, idx1], V[3, idx1]
        p2x, p2y, p2z = V[1, idx2], V[2, idx2], V[3, idx2]
        p3x, p3y, p3z = V[1, idx3], V[2, idx3], V[3, idx3]

        # Cross product of p2 and p3 (p2 x p3)
        cx = p2y * p3z - p2z * p3y
        cy = p2z * p3x - p2x * p3z
        cz = p2x * p3y - p2y * p3x

        # Dot product with p1 (p1 . (p2 x p3))
        vol += p1x * cx + p1y * cy + p1z * cz
    end

    vol = abs(vol) / 6.0
    return ParticleTriangleMesh(V, C, vol)
end


TriangleMesh(mesh::GeometryBasics.Mesh; units=u"m") =
    TriangleMesh(convert_mesh(mesh); units=units)

TriangleMesh(filepath::AbstractString; units=u"m") =
    TriangleMesh(convert_mesh(FileIO.load(filepath)); units=units)

ParticleTriangleMesh(mesh::GeometryBasics.Mesh; units=u"m") =
    ParticleTriangleMesh(convert_mesh(mesh); units=units)

ParticleTriangleMesh(filepath::AbstractString; units=u"m") =
    ParticleTriangleMesh(convert_mesh(FileIO.load(filepath)); units=units)

@inline function materialize_trianglemesh(mesh; units=u"m")
    mesh isa TriangleMesh && return mesh
    return TriangleMesh(mesh; units=units)
end

function mirror(mesh; units=u"m")
    TriangleSurface(materialize_trianglemesh(mesh; units=units), Mirror())
end

function sink(mesh; units=u"m")
    TriangleSurface(materialize_trianglemesh(mesh; units=units), Sink())
end

# Compute axis-aligned bounding box of a TriangleMesh
function compute_bounds(tm::TriangleMesh)
    V = tm.vertices
    T = float(eltype(V))
    mins = SVector{3,T}(minimum(V[1, :]), minimum(V[2, :]), minimum(V[3, :]))
    maxs = SVector{3,T}(maximum(V[1, :]), maximum(V[2, :]), maximum(V[3, :]))
    return mins, maxs
end

# Union multiple TriangleMesh objects into a single TriangleMesh
function union_trianglemesh(meshes::AbstractVector{<:TriangleMesh})::TriangleMesh
    isempty(meshes) && throw(ArgumentError("union_trianglemesh: input meshes must not be empty"))

    T = float(eltype(meshes[1].vertices))
    verts_all = Vector{T}()
    conns_all = Int32[]
    nverts_total = 0

    for tm in meshes
        V = tm.vertices
        C = tm.connectivity
        Nv = size(V, 2)
        Nt = size(C, 2)

        append!(verts_all, vec(T.(V)))

        off = Int32(nverts_total)
        @inbounds for t in 1:Nt
            push!(conns_all, Int32(C[1, t]) + off)
            push!(conns_all, Int32(C[2, t]) + off)
            push!(conns_all, Int32(C[3, t]) + off)
        end

        nverts_total += Nv
    end

    Vmat = reshape(verts_all, 3, :)
    Cmat = reshape(conns_all, 3, :)
    return TriangleMesh(Vmat, Cmat)
end


function get_rotation_matrix(angles::AbstractVector; units=u"°")::SMatrix{3,3,Float64,9}
    length(angles) == 3 || throw(ArgumentError("get_rotation_matrix: input vector must contain 3 angles (θx, θy, θz)"))

    # Map inputs: Assume [ang_x, ang_y, ang_z] (Standard CSV/DataFrame order)
    # We want R = Rz(ang_z) * Ry(ang_y) * Rx(ang_x)

    θx_val = angles[1]
    θy_val = angles[2]
    θz_val = angles[3]

    # Convert to radians
    if units == u"°"
        θx_val = deg2rad(θx_val)
        θy_val = deg2rad(θy_val)
        θz_val = deg2rad(θz_val)
    else
        θx_val = Float64(θx_val)
        θy_val = Float64(θy_val)
        θz_val = Float64(θz_val)
    end

    θ1 = θz_val
    θ2 = θy_val
    θ3 = θx_val

    sθ1, cθ1 = sincos(θ1)
    sθ2, cθ2 = sincos(θ2)
    sθ3, cθ3 = sincos(θ3)

    # R = Rz(θ1) * Ry(θ2) * Rx(θ3)
    # Col 1
    r11 = cθ1 * cθ2
    r21 = sθ1 * cθ2
    r31 = -sθ2
    # Col 2
    r12 = cθ1 * sθ2 * sθ3 - sθ1 * cθ3
    r22 = sθ1 * sθ2 * sθ3 + cθ1 * cθ3
    r32 = cθ2 * sθ3

    # Col 3
    r13 = cθ1 * sθ2 * cθ3 + sθ1 * sθ3
    r23 = sθ1 * sθ2 * cθ3 - cθ1 * sθ3
    r33 = cθ2 * cθ3

    return SMatrix{3,3,Float64,9}(
        r11, r21, r31,
        r12, r22, r32,
        r13, r23, r33
    )
end

function translation_matrix(X::AbstractMatrix{<:Real})
    n_particles = size(X, 1)
    size(X, 2) == 3 || throw(ArgumentError("translation_matrix: input matrix must be Nx3"))

    translations = Vector{SVector{3,Float64}}(undef, n_particles)

    @inbounds for i in 1:n_particles
        translations[i] = SVector{3,Float64}(float(X[i, 1]), float(X[i, 2]), float(X[i, 3]))
    end

    return translations
end

function get_scale_factor(mesh::ParticleTriangleMesh, radius::Float64)::Float64
    mesh_vol = mesh.volume
    @assert !isnan(mesh_vol) "get_scale_factor: mesh volume is NaN; cannot compute scale factor"
    unit_sphere_vol = (4.0f0 / 3.0f0) * π * radius^3
    scale_factor = (unit_sphere_vol / mesh_vol)^(1.0f0 / 3.0f0)
    return scale_factor
end
