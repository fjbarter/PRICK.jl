
using Profile
using PRICK

using PProf

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

function main(;
    n_rays::Int=1_000,
    max_bounces::Int=20_000,
    max_length::Float64=1e6,
    sample_rate::Float64=0.01,
    flat_report_path::String="alloc_profile_flat.txt",
    tree_report_path::String="alloc_profile_tree.txt",
)
    println("Building geometry...")
    vessel_trianglemesh = TriangleMesh("RAMCylinder.stl"; units=u"m")
    surfaces = [sink(vessel_trianglemesh)]
    surface_bvh = build_surface_bvh(surfaces)

    println("Reading particles...")
    data = read_vtk_file("particles_0.vtk")
    x, y, z, r = retrieve_coordinates(data)
    X = permutedims(Float64.(hcat(x, y, z)))
    radii = Float64.(r)
    spheres = build_sphere_bvh(X, radii)

    p0 = (PRICK.mean(x), PRICK.mean(y), PRICK.mean(z))
    D = random_unit_directions(n_rays)

    println("Warmup run...")
    trace_rays(p0, D, surface_bvh, spheres; max_bounces=max_bounces, max_length=max_length)

    println("Profiling allocations (sample_rate=$sample_rate)...")
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=sample_rate begin
        trace_rays(p0, D, surface_bvh, spheres; max_bounces=max_bounces, max_length=max_length)
    end

    prof = Profile.Allocs.fetch()
    println("Collected allocation samples: ", length(prof.allocs))

    println("Writing text allocation reports...")
    open(flat_report_path, "w") do io
        Profile.Allocs.print(io, prof; format=:flat, C=false, maxdepth=12, mincount=1)
    end
    open(tree_report_path, "w") do io
        Profile.Allocs.print(io, prof; format=:tree, C=false, maxdepth=20, mincount=1)
    end
    println("Wrote: ", abspath(flat_report_path))
    println("Wrote: ", abspath(tree_report_path))

    println("Opening pprof web UI...")
    PProf.Allocs.pprof(prof; from_c=false)
    println("If browser does not open automatically, use the URL printed by PProf.")
end

main()
