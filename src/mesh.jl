# ============================================================
# Mesh definitions and construction helpers
# ============================================================
struct TriangleMesh
    vertices::Matrix{Float64}     # (3, Nv) stored in metres
    connectivity::Matrix{Int32}   # (3, Nt)
end

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

function TriangleMesh(mesh::Meshes.SimpleMesh; units = u"m")
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

TriangleMesh(mesh::GeometryBasics.Mesh; units = u"m") =
    TriangleMesh(convert_mesh(mesh); units=units)

TriangleMesh(filepath::AbstractString; units = u"m") =
    TriangleMesh(convert_mesh(FileIO.load(filepath)); units=units)

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
