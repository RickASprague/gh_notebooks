# Common to every imagenet64 project: the ImageNet-64 shards and their loaders. Nothing model- or
# evaluation-specific lives here.
#
# Shards: imagenet64/build/{train,val}, written once by utils/preprocess.jl.
#
#   include(joinpath(<imagenet64>, "utils", "data.jl"))
#   N, ys, Xs, μs = load_train()     # 10 shards: Xs[k] is a (64,64,3,Nk) mmapped view, ys[k] labels 1..1000
#   yv, Xv = load_val()              # 50 000 images, one shard

include(joinpath(@__DIR__, "utils.jl"))
using .Utils

const IMAGENET64_DATA = joinpath(dirname(@__DIR__), "build")

# Training shards. MMapReader maps Xs.ser once and hands back reshape'd views
# into the mapped buffer: no read, no parse, no copy. Labels and means are tiny
# and go through plain IO.
function load_train()
    base = joinpath(IMAGENET64_DATA, "train")
    open(joinpath(base, "count.ser"), "r") do io
        N  = deserialize(io, Int64)                         # number of shards
        Xs = Vector{AbstractArray{Float32, 4}}(undef, N)    # (64,64,3,Nk) views
        μs = Vector{AbstractArray{Float32, 3}}(undef, N)
        ys = Vector{Vector{Int}}(undef, N)

        yf = open(joinpath(base, "ys.ser"), "r")
        Xf = MMapReader(joinpath(base, "Xs.ser"))
        μf = open(joinpath(base, "μs.ser"), "r")
        count = 0
        for i ∈ 1:N
            Xs[i] = deserialize(Xf, Array{Float32})
            μs[i] = deserialize(μf, Array{Float32})
            ys[i] = deserialize(yf, Array{Int64})
            count += length(ys[i])
        end
        close(yf); close(Xf); close(μf)
        return count, ys, Xs, μs
    end
end

# Validation: one shard of 50k, same format, no means.
function load_val()
    base = joinpath(IMAGENET64_DATA, "val")
    Xf = MMapReader(joinpath(base, "Xs.ser"))
    X  = deserialize(Xf, Array{Float32})
    close(Xf)
    y = open(io -> deserialize(io, Array{Int64}), joinpath(base, "ys.ser"))
    return y, X
end
