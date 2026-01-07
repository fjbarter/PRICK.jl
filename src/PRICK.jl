# PRICK.jl

module PRICK

using StaticArrays
import ImplicitBVH: BVH, BBox, BSphere, traverse_rays

using Meshes
using FileIO               # FileIO.load
using GeometryBasics       # mesh.position, mesh.faces
using Packing3D            # For legacy VTK IO from LIGGGHTS
using Unitful
using Unitful: ustrip, uconvert, m, @u_str

include("utils.jl")
include("mesh.jl")
include("bvh.jl")
include("intersections.jl")
include("tracer.jl")
include("visualise.jl")

# Unitful
export @u_str

# Packing3D
export read_vtk_file, retrieve_coordinates

# StaticArrays
export SVector, SMatrix

# PRICK
export TriangleMesh, TriangleSurface, Mirror, Sink
export mirror, sink
export build_sphere_bvh
export trace_ray_geometric, RayTraceResult
export visualise_trace
export path_to_matrix

end # module PRICK
