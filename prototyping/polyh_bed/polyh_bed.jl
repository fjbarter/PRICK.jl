using PRICK
using CSV
using DataFrames
using GLMakie

function example2()
    # Load geometry
    container_tm = TriangleMesh("Extents.stl", units=u"m")
    surfaces = [sink(container_tm)]

    # Load particle mesh
    particle_tm = ParticleTriangleMesh("Particle.stl"; units=u"m")

    # Load particle data
    # Expected columns: 
    #   x, y, z, r (radius)
    #   rot_x, rot_y, rot_z (Euler angles)
    df = CSV.read("ParticlePositions.csv", DataFrame)
    n_particles = nrow(df)

    # Convert df Matrix & Vector
    X = Matrix{Float64}(undef, 3, n_particles)
    X[1, :] = df.x
    X[2, :] = df.y
    X[3, :] = df.z

    r = Vector{Float64}(df.r)
    orients = Matrix{Float64}(undef, 3, n_particles)
    orients[1, :] = df.rot_x
    orients[2, :] = df.rot_y
    orients[3, :] = df.rot_z

    # Build polyhedral BVH
    println("Building Polyhedral BVH...")
    polyh_bvh = build_polyh_bvh(particle_tm, X, r, orients)

    # Use a random restart hill-climb to find a void starting point near center
    p0 = find_void_rrhc(X, r)

    # Random direction
    φ = 2π * rand()
    u = 2rand() - 1
    s = sqrt(1 - u * u)
    d = SVector(s * cos(φ), s * sin(φ), u)

    # Run Trace
    res = trace_ray_geometric(
        p0,
        d,
        surfaces,
        polyh_bvh
    )

    # Results
    println("--------------------------------")
    println("Trace Result:")
    println("  Escaped:      $(res.escaped)")
    println("  Total Length: $(res.total_length) m")
    println("  Steps:        $(res.nsteps)")
    println("--------------------------------")

    # Optional: Return data for visualization or analysis
    P = isempty(res.path) ? Float64[] : PRICK.path_to_matrix(res.path)

    return (
        res=res,
        path_matrix=P,
        container=container_tm,
        particles=polyh_bvh
    )
end

out = example2()

fig = visualise_trace(
    out.res,
    out.particles;
    vessel_tm=out.container,
    show_vessel=true,
    show_particles=false,
)

GLMakie.activate!()
display(fig)

println("Press Enter to exit...")
readline()