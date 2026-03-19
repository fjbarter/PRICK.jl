# Active-ray LVT traversal wrapper (CPU only, in-place cache)
# ============================================================

function ensure_active_traversal_cache(
    cache::Any,
    bvh::BVH,
    options,
)
    I = ImplicitBVH.get_index_type(bvh)
    npadding = div(64, sizeof(I), RoundUp)
    thread_ncontacts_length = npadding * options.num_threads

    if cache isa ActiveTraversalCache{I}
        if length(cache.thread_ncontacts_raw) < thread_ncontacts_length
            resize!(cache.thread_ncontacts_raw, thread_ncontacts_length)
        end
        return cache, npadding
    end

    contacts = similar(bvh.nodes, ImplicitBVH.IndexPair{I}, 0)
    thread_ncontacts_raw = similar(bvh.nodes, I, thread_ncontacts_length)
    return ActiveTraversalCache{I}(contacts, thread_ncontacts_raw, 0), npadding
end

function traverse_rays_active_lvt!(
    bvh::BVH,
    points::AbstractMatrix,
    directions::AbstractMatrix,
    active::AbstractVector{Bool};
    start_level::Int=ImplicitBVH.default_start_level(bvh, ImplicitBVH.LVTTraversal()),
    narrow=(bv, p, d) -> true,
    cache=nothing,
    options=ImplicitBVH.BVHOptions(),
)
    size(points, 1) == 3 || throw(ArgumentError("points must have shape (3, N)"))
    size(directions, 1) == 3 || throw(ArgumentError("directions must have shape (3, N)"))
    size(points, 2) == size(directions, 2) || throw(ArgumentError("point and direction counts must match"))
    length(active) == size(points, 2) || throw(ArgumentError("active mask length must match ray count"))
    (bvh.built_level <= start_level <= bvh.tree.levels <= 32) || throw(ArgumentError("invalid start level"))

    AK.get_backend(bvh.nodes) == AK.get_backend(points) == AK.get_backend(directions) ||
        throw(ArgumentError("bvh, points and directions must share backend"))
    AK.get_backend(bvh.nodes) isa typeof(AK.CPU_BACKEND) ||
        throw(ArgumentError("traverse_rays_active_lvt! currently supports CPU backend only"))

    cache_typed, npadding = ensure_active_traversal_cache(cache, bvh, options)

    num_rays = size(points, 2)
    if num_rays == 0
        cache_typed.num_contacts = 0
        return cache_typed
    end

    I = ImplicitBVH.get_index_type(bvh)
    thread_ncontacts = view(reshape(cache_typed.thread_ncontacts_raw, npadding, :), 1, :)
    thread_ncontacts .= I(0)

    traverse_rays_lvt_active!(
        bvh, points, directions, active, start_level,
        thread_ncontacts, nothing, narrow, options,
    )

    AK.accumulate!(+, thread_ncontacts, init=I(0), max_tasks=1, block_size=options.block_size)

    total_contacts = Int(thread_ncontacts[end])
    cache_typed.num_contacts = total_contacts
    if length(cache_typed.contacts) < total_contacts
        resize!(cache_typed.contacts, total_contacts)
    end

    if total_contacts > 0
        traverse_rays_lvt_active!(
            bvh, points, directions, active, start_level,
            thread_ncontacts, cache_typed.contacts, narrow, options,
        )
    end

    return cache_typed
end

function traverse_rays_lvt_active!(
    bvh::BVH,
    points::AbstractMatrix,
    directions::AbstractMatrix,
    active::AbstractVector{Bool},
    start_level::Int,
    thread_ncontacts::AbstractVector,
    contacts::Union{Nothing, AbstractVector},
    narrow,
    options,
)
    num_checks = size(points, 2)
    AK.itask_partition(num_checks, options.num_threads, options.min_traversals_per_thread) do itask, irange
        I = ImplicitBVH.get_index_type(bvh)
        T = eltype(eltype(bvh.leaves))
        stack = ImplicitBVH.SimpleMVector{32, I}(undef)
        iwrite = if isnothing(contacts)
            Ref(I(itask))
        else
            Ref(itask == 0x1 ? I(0x1) : I(thread_ncontacts[itask - 0x1] + 0x1))
        end

        for i in irange
            @inbounds active[i] || continue
            point = (
                T(points[1, i]),
                T(points[2, i]),
                T(points[3, i]),
            )
            direction = (
                T(directions[1, i]),
                T(directions[2, i]),
                T(directions[3, i]),
            )
            @inline ImplicitBVH.traverse_ray_lvt!(
                point, direction, I(i),
                bvh, stack,
                I(start_level), iwrite,
                thread_ncontacts, contacts,
                narrow,
            )
        end
    end

    return nothing
end
