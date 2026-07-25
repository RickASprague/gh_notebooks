# One-time conversion: ImageNet-64 Python pickles → flat binary shards that
# MMapReader can map at training time.
#
#   julia --project src/preprocess.jl
#
# Needs the raw pickles under ImageNet.IMAGENET_64 (/data/imagenet/Imagenet64).
# Writes build/train/{count,Xs,ys,μs}.ser and build/val/{Xs,ys}.ser.
# Format per array: [Int32 ndims][Int32 dims...][raw payload] (lib/utils.jl).

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "lib", "imagenet.jl"))
include(joinpath(ROOT, "lib", "utils.jl"))
import .ImageNet
using .Utils

# Unpickle → (N,64,64,3) → permute to Julia's column-major (64,64,3,N).
function load_one(file; has_mean = true)
    labels, images, μ = ImageNet.load(file, 64; has_mean)
    return labels, permutedims(images, (2, 3, 4, 1)), μ
end

function fresh_dir(name)
    base = joinpath(ROOT, "build", name)
    ispath(base) && rm(base, force = true, recursive = true)
    mkpath(base)
    return base
end

# 1.28M images in 10 pickles (2 parts x 5 batches); one shard per pickle.
function convert_train(files)
    base = fresh_dir("train")
    open(io -> write(io, length(files)), joinpath(base, "count.ser"), "w")
    open(joinpath(base, "ys.ser"), "w") do ys
        open(joinpath(base, "Xs.ser"), "w") do Xs
            open(joinpath(base, "μs.ser"), "w") do μs
                for f ∈ files
                    y, X, μ = load_one(f)
                    serialize(Xs, X)
                    serialize(μs, μ)
                    serialize(ys, y)
                    @info "train shard" file = f n = size(X, 4)
                end
            end
        end
    end
end

# 50k images, one pickle, no per-shard mean.
function convert_val(file)
    base = fresh_dir("val")
    y, X, _ = load_one(file; has_mean = false)
    open(io -> serialize(io, X), joinpath(base, "Xs.ser"), "w")
    open(io -> serialize(io, y), joinpath(base, "ys.ser"), "w")
    @info "val shard" file = file n = size(X, 4)
end

src = ImageNet.IMAGENET_64
@time convert_train(vcat(readdir("$src/Imagenet64_train_part1", join = true),
                         readdir("$src/Imagenet64_train_part2", join = true)))
@time convert_val(readdir("$src/Imagenet64_val", join = true)[1])
