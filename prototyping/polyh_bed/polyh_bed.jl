using PRICK
using CSV
using DataFrames
using BenchmarkTools

function random_unit_directions(n::Integer)
    n > 0 || throw(ArgumentError("n must be > 0"))
    D = Matrix{Float64}(undef, 3, n)
    @inbounds for i in 1:n
        phi = 2pi * rand()
        u = 2rand() - 1
        s = sqrt(1 - u * u)
        D[1, i] = s * cos(phi)
        D[2, i] = s * sin(phi)
        D[3, i] = u
    end
    return D
end

function example(;
    n_rays::Int=1_000,
    max_bounces::Int=20_000,
    max_length::Float64=1e6,
)
    println("Building container surface...")
    container_tm = TriangleMesh("Extents.stl"; units=u"m")
    surfaces = [sink(container_tm)]
    surface_bvh = build_surface_bvh(surfaces)

    println("Building particle template...")
    particle_tm = ParticleTriangleMesh("Particle.stl"; units=u"m")

    println("Reading particle state...")
    df = CSV.read("ParticlePositions.csv", DataFrame)
    n_particles = nrow(df)

    X = Matrix{Float64}(undef, 3, n_particles)
    X[1, :] = df.x
    X[2, :] = df.y
    X[3, :] = df.z

    r = Vector{Float64}(df.r)
    orients = Matrix{Float64}(undef, 3, n_particles)
    orients[1, :] = df.rot_x
    orients[2, :] = df.rot_y
    orients[3, :] = df.rot_z

    println("Building polyhedral BVH...")
    polyh_bvh = build_polyh_bvh(particle_tm, X, r, orients)

    p0 = find_void_rrhc(X, r)
    D = random_unit_directions(n_rays)

    println("Tracing $n_rays rays...")
    res = trace_rays(
        p0,
        D,
        surface_bvh,
        polyh_bvh;
        max_bounces=max_bounces,
        max_length=max_length,
    )

    escaped_lengths = [res.total_length[i] for i in eachindex(res.total_length) if res.escaped[i]]
    escaped_steps = [res.nsteps[i] for i in eachindex(res.nsteps) if res.escaped[i]]
    escaped_term_dists = [res.p0_termination_distance[i] for i in eachindex(res.p0_termination_distance) if res.escaped[i]]

    mean_length = isempty(escaped_lengths) ? NaN : PRICK.mean(escaped_lengths)
    mean_steps = isempty(escaped_steps) ? NaN : PRICK.mean(escaped_steps)
    mean_tortuosity = isempty(escaped_term_dists) ? NaN : sum(escaped_lengths) / sum(escaped_term_dists)

    println("Escaped rays: $(length(escaped_lengths)) / $n_rays")
    println("Mean path length over escaped rays: $(mean_length) m")
    println("Mean number of reflections over escaped rays: $(mean_steps)")
    println("Mean tortuosity: $mean_tortuosity")

    return res
end

@benchmark example()
