# DataLoader for the ImageNet-64 training shards: one loader, fast on every machine.
#
#   include(joinpath(<imagenet64>, "utils", "data.jl"))
#   include(joinpath(<imagenet64>, "utils", "dataloader.jl"))
#   N, ys, Xs, _ = load_train()
#   loader = DataLoader(Xs, ys; batch = 1024)
#   for (x, y) ∈ loader          # one pass = one epoch, reshuffled every pass
#       copyto!(x_gpu, x)        # x: (64,64,3,batch) Float32 host buffer, y: its labels (1..1000)
#       ...
#   end
#
# `x` is a reused buffer: valid until the next iteration step, so copy it (to the GPU) first.
#
# How it reads:
#   - each batch is ONE read!() of batch·64·64·3·4 bytes straight from Xs.ser at the batch's byte
#     offset: no mmap page faults;
#   - a reader task on another thread reads ahead into `nbuf` reusable buffers while the caller
#     trains on the current one (needs julia -t ≥ 2 to overlap; with -t 1 it still works);
#   - shuffling is by chunks: every shard is cut into chunks of `chunk` consecutive batches, the
#     chunk order (across all shards) is reshuffled each epoch, and the batches inside a chunk are
#     read in order. Reads stay large and sequential while the batch order is close to random.
# Threadripper: the shards sit in the RAM cache, so a read!() is a memcpy and costs nothing.
# NAS: 60 GB of shards > 30 GB of RAM on spinning disks; big sequential reads + read-ahead are what
# keep the GPU fed (mmap page faults managed ~20 MB/s there).
#
# Full batches only: each shard's ragged tail is dropped, as before.

using Random

mutable struct DataLoader
    path::String                  # train/Xs.ser
    offsets::Vector{Int}          # byte offset of each shard's payload in Xs.ser
    sizes::Vector{Int}            # images per shard
    ys::Vector{Vector{Int}}       # labels per shard
    batch::Int
    chunk::Int                    # consecutive batches read in order
    nbuf::Int                     # read-ahead depth (host buffers in flight)
    limit::Int                    # max batches per pass (smoke tests)
    rng::AbstractRNG
    active::Any                   # (free, ready) channels of an unfinished pass, or nothing
end

# Xs[k] = reshape(reinterpret(Float32, view(mmap_bytes, a:b)), dims)  →  its payload starts at byte a - 1
shard_offset(Xk) = first(parentindices(parent(parent(Xk)))[1]) - 1

function DataLoader(Xs, ys; batch, chunk = 16, nbuf = 4, limit = typemax(Int),
                    rng = Random.default_rng(),
                    path = joinpath(IMAGENET64_DATA, "train", "Xs.ser"))
    Threads.nthreads() ≥ 2 ||
        @warn "DataLoader: julia runs single-threaded, so reads cannot overlap training; use julia -t 4"
    return DataLoader(path, shard_offset.(Xs), [size(X, 4) for X ∈ Xs], ys,
                      batch, chunk, nbuf, limit, rng, nothing)
end

Base.length(dl::DataLoader) = min(dl.limit, sum(n ÷ dl.batch for n ∈ dl.sizes))

# One epoch's reading plan: shuffled chunks (shard k, first image i0, number of batches m).
function epoch_plan(dl::DataLoader)
    chunks = Tuple{Int, Int, Int}[]
    for (k, n) ∈ enumerate(dl.sizes)
        nb = n ÷ dl.batch                                   # full batches in shard k
        for c ∈ 1:dl.chunk:nb
            push!(chunks, (k, (c - 1) * dl.batch + 1, min(dl.chunk, nb - c + 1)))
        end
    end
    return shuffle(dl.rng, chunks)
end

# Reader task: fill free buffers in plan order and hand them to the consumer.
function read_batches(dl::DataLoader, free, ready, plan)
    imgbytes = 64 * 64 * 3 * sizeof(Float32)
    nb = 0
    try
        open(dl.path, "r") do io
            for (k, i0, m) ∈ plan
                seek(io, dl.offsets[k] + (i0 - 1) * imgbytes)   # a chunk is contiguous: one seek
                for b ∈ 0:m-1
                    nb ≥ dl.limit && return
                    buf = take!(free)
                    read!(io, buf)                              # one sequential read per batch
                    put!(ready, (buf, k, i0 + b * dl.batch))
                    nb += 1
                end
            end
        end
    catch e
        e isa InvalidStateException || rethrow()                # pass abandoned (break): channels closed
    end
end

# Abandon an unfinished pass (the caller broke out of the loop): closing the channels makes the
# reader's blocked take!/put! throw, so it exits; its buffers are garbage.
function stop!(dl::DataLoader)
    dl.active === nothing && return
    free, ready = dl.active
    close(free); close(ready)
    dl.active = nothing
end

function Base.iterate(dl::DataLoader)
    stop!(dl)
    free  = Channel{Array{Float32, 4}}(dl.nbuf)
    ready = Channel{Tuple{Array{Float32, 4}, Int, Int}}(dl.nbuf)
    for _ ∈ 1:dl.nbuf
        put!(free, Array{Float32}(undef, 64, 64, 3, dl.batch))
    end
    reader = Threads.@spawn read_batches(dl, free, ready, epoch_plan(dl))
    bind(ready, reader)                          # closes `ready` when the reader ends; a reader error rethrows below
    dl.active = (free, ready)
    return iterate(dl, (free, ready, nothing))
end

function Base.iterate(dl::DataLoader, (free, ready, held))
    held === nothing || put!(free, held)         # the caller is done with the previous batch
    item = try
        take!(ready)
    catch e
        e isa InvalidStateException || rethrow() # a reader error surfaces here
        nothing                                  # pass finished
    end
    if item === nothing
        dl.active = nothing
        close(free)
        return nothing
    end
    buf, k, i = item
    return (buf, view(dl.ys[k], i:i + dl.batch - 1)), (free, ready, buf)
end
