# ------------------------------------------------------------
# Small vector helpers
# ------------------------------------------------------------
@inline dot3(a, b) = @inbounds a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

@inline function cross3(a::SVector{3,Float64}, b::SVector{3,Float64})
    @inbounds return SVector{3,Float64}(
        a[2]*b[3] - a[3]*b[2],
        a[3]*b[1] - a[1]*b[3],
        a[1]*b[2] - a[2]*b[1],
    )
end

@inline function to_mat3x1(v::SVector{3,Float64})
    M = Matrix{Float64}(undef, 3, 1)
    @inbounds begin
        M[1,1] = v[1]; M[2,1] = v[2]; M[3,1] = v[3]
    end
    return M
end

function mean(values::AbstractVector{T}) where T <: Real
    if isempty(values)
        return T <: AbstractFloat ? T(NaN) : throw(ArgumentError("mean requires a non-empty vector"))
    end
    s = zero(T)
    @inbounds for v in values
        s += v
    end
    return s / length(values)
end
