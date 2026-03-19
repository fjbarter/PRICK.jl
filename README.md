# PRICK.jl

Particle Ray-tracing In Container Kernel (PRICK.jl) is a lightweight geometric ray tracer for triangles and spheres, designed for modelling tortuosity in packed beds

## Quick start

First, you will need to download PRICK. In a Julia REPL:
```julia
using Pkg
Pkg.develop(url="https://github.com/fjbarter/PRICK.jl")
```

Then you can load sphere and triangle-wall data, define one start point, and trace a batch of directions:
```julia
using PRICK

# Load mesh and tag each surface as a mirror or a sink
vessel_tm = TriangleMesh("RAMCylinder.stl"; units=u"m")
surfaces = [mirror(vessel_tm)]  # or sink(vessel_tm)
surface_bvh = build_surface_bvh(surfaces)  # build once and reuse

# Load spheres (example using Packing3D VTK reader)
data = read_vtk_file("particles_0.vtk")
x, y, z, r = retrieve_coordinates(data)
X = permutedims(Float64.(hcat(x, y, z)))
radii = Float64.(r)

# Build the spheres together and construct BVH
spheres = build_sphere_bvh(X, radii)

# Trace N rays from a single start point p0 with a (3, N) direction matrix
p0 = (0.0079, 0.0, 0.022)
N = 10_000
D = randn(3, N)  # directions are normalized internally

res = trace_rays(
    p0,
    D,
    surface_bvh,
    spheres;
    max_bounces=20_000,
    max_length=1e6,
)
```

## Notes

- `mirror` surfaces reflect specularly.
- `sink` surfaces terminate rays.
- The union mesh bounding box is used as an unmeshed sink and is expanded by 10%.
- Batched tracing returns per-ray `escaped`, `total_length`, `nsteps`, and `termination`.
