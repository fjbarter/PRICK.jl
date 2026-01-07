using PRICK
using GLMakie

# ============================================================
# example() demonstrating:
# - provide X, radii
# - provide vessel
# - trace a ray and return results
# ============================================================
function example()
    # Vessel from stl file
    vessel_trianglemesh = TriangleMesh("RAMCylinder.stl"; units=u"m")

    # Construct surface (each is either a sink or mirror)
    surfaces = [
        sink(vessel_trianglemesh),
    ]

    # Spheres from VTK file
    println("Reading file...")
    file = "particles_0.vtk"
    data = read_vtk_file(file)
    x, y, z, r = retrieve_coordinates(data)

    X = permutedims(Float64.(hcat(x, y, z)))   # 3×N
    radii = Float64.(r)
    spheres = build_sphere_bvh(X, radii)

    # Build p0 from mean sphere position
    p0x = PRICK.mean(x)
    p0y = PRICK.mean(y)
    p0z = PRICK.mean(z)
    p0 = SVector{3,Float64}(p0x, p0y, p0z)

    # p0 = SVector{3,Float64}(0.0079, 0.0, 0.022)

    N = 1
    path_lengths = zeros(Float64, N)
    n_steps = zeros(Float64, N)
    println("Tracing $N rays...")
    for i in 1:N
        # Fixed start point
        
        φ = 2π * rand()          # azimuth
        u = 2rand() - 1          # u = cos(θ) uniformly in [-1,1]
        s = sqrt(1 - u*u)
        d = SVector(s*cos(φ), s*sin(φ), u)

        res = trace_ray_geometric(p0, d, surfaces, spheres;
                                  record_path=false,
                                  max_steps=20000,
                                  max_length=1e6)
        
        path_lengths[i] = res.escaped ? res.total_length : NaN
        n_steps[i] = res.escaped ? res.nsteps : NaN
    end

    path_lengths_dense = filter(!isnan, path_lengths)

    mean_length = PRICK.mean(path_lengths_dense)
    println("Mean path length over $N rays (escaped: $(length(path_lengths_dense))): $(mean_length) m")

    n_steps_dense = filter(!isnan, n_steps)
    mean_number_of_steps = PRICK.mean(n_steps_dense)
    println("Mean number of steps over escaped rays: $(mean_number_of_steps)")

    # Single ray trace example

    # Start point and direction
    p0 = SVector{3,Float64}(0.0079, 0.0, 0.022)
    φ = 2π * rand()          # azimuth
        u = 2rand() - 1          # u = cos(θ) uniformly in [-1,1]
        s = sqrt(1 - u*u)
        d = SVector(s*cos(φ), s*sin(φ), u)

    res = trace_ray_geometric(p0, d, surfaces, spheres;
                              record_path=true,
                              max_steps=20000,
                              max_length=1e6)

    println("escaped       = ", res.escaped)
    println("total_length  = ", res.total_length)
    println("nsteps        = ", res.nsteps)
    println("path vertices = ", length(res.path))

    # If you want to plot later:
    P = path_to_matrix(res.path)  # 3xK
    return (res=res, X=X, radii=r, vessel_tm=vessel_trianglemesh, path_matrix=P)
end

out = example()
fig = visualise_trace(out.res; X=out.X, radii=out.radii, vessel_tm=out.vessel_tm)
display(fig)
