# PRICK.jl

module PRICK

using StaticArrays
using ImplicitBVH
import ImplicitBVH: BVH, BBox, BSphere
import AcceleratedKernels as AK

using Meshes
using FileIO               # FileIO.load
using GeometryBasics       # mesh.position, mesh.faces
using Packing3D            # For legacy VTK IO from LIGGGHTS
using Unitful
using Unitful: ustrip, uconvert, m, @u_str

include("utils.jl")
include("mesh.jl")
include("bvh.jl")
include("ray_batch.jl")
include("raytrace_active_lvt.jl")
include("intersections.jl")
include("tracer.jl")
include("visualise.jl")

# Unitful
export @u_str

# Packing3D
export read_vtk_file, retrieve_coordinates

# PRICK
export TriangleMesh, TriangleSurface, ParticleTriangleMesh, Mirror, Sink
export mirror, sink
export build_sphere_bvh, build_polyh_bvh, build_surface_bvh
export trace_rays, RayTraceBatchResult, RayTermination
export visualise_trace
export path_to_matrix
export find_void_rrhc

end # module PRICK
