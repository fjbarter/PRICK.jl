# ============================================================
# Example batched ray tracing in a sphere bed
# ============================================================
using PRICK
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

vessel_trianglemesh = TriangleMesh("RAMCylinder.stl"; units=u"m")
surfaces = [sink(vessel_trianglemesh)]
surface_bvh = build_surface_bvh(surfaces)

println("Reading file...")
data = read_vtk_file("particles_0.vtk")
x, y, z, r = retrieve_coordinates(data)

X = permutedims(Float64.(hcat(x, y, z)))
radii = Float64.(r)
spheres = build_sphere_bvh(X, radii)

p0 = (PRICK.mean(x), PRICK.mean(y), PRICK.mean(z))
n_rays = 1000
D = random_unit_directions(n_rays)

println("Tracing $n_rays rays...")
t_trace0 = time()

@benchmark trace_rays(
    $p0,
    $D,
    $surface_bvh,
    $spheres;
    max_bounces=20_000,
    max_length=1e6,
)
# t_trace = time() - t_trace0
# println("Traced $n_rays rays in $t_trace seconds")

# escaped_lengths = [res.total_length[i] for i in eachindex(res.total_length) if res.escaped[i]]
# escaped_steps = [res.nsteps[i] for i in eachindex(res.nsteps) if res.escaped[i]]
# escaped_term_dists = [res.p0_termination_distance[i] for i in eachindex(res.p0_termination_distance) if res.escaped[i]]

# mean_length = isempty(escaped_lengths) ? NaN : PRICK.mean(escaped_lengths)
# mean_steps = isempty(escaped_steps) ? NaN : PRICK.mean(escaped_steps)
# mean_tortuosity = sum(escaped_lengths)/sum(escaped_term_dists)

# println("Escaped rays: $(length(escaped_lengths)) / $n_rays")
# println("Mean path length over escaped rays: $(mean_length) m")
# println("Mean number of reflections over escaped rays: $(mean_steps)")
# println("Mean tortuosity: $mean_tortuosity")

