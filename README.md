# PRICK.jl

Particle Ray-tracing In Container Kernel (PRICK.jl) is a lightweight geometric ray tracer for triangles and spheres, designed for modelling tortuosity in packed beds

## Quick start

First, you will need to download PRICK. In a Julia REPL:
```julia
using Pkg
Pkg.develop(url="https://github.com/fjbarter/PRICK.jl")
```

Then you can simply load in your sphere and triangle-wall data, define a start point and direction as SVectors, as below:
```julia
using PRICK

# Load mesh and tag each surface as a mirror or a sink
vessel_tm = TriangleMesh("RAMCylinder.stl"; units=u"m")
surfaces = [mirror(vessel_tm)]  # or sink(vessel_tm)

# Load spheres (example using Packing3D VTK reader)
data = read_vtk_file("particles_0.vtk")
x, y, z, r = retrieve_coordinates(data)
X = permutedims(Float64.(hcat(x, y, z)))
radii = Float64.(r)

# Build the spheres together and construct BVH
spheres = build_sphere_bvh(X, radii)

# Trace a ray
p0 = SVector(0.0079, 0.0, 0.022)
d = SVector(1.0, 0.0, 0.0)
res = trace_ray_geometric(p0, d, surfaces, spheres)
```

## Visualisation (optional)

`visualise_trace` requires GLMakie to be loaded by the user:

```julia
using GLMakie
fig = visualise_trace(
    res;
    X=X, radii=radii,
    vessel_tm=vessel_tm
)
display(fig)
```

## Notes

- `mirror` surfaces reflect specularly.
- `sink` surfaces terminate rays.
- The union mesh bounding box is used as an unmeshed sink and is expanded by 10%.
