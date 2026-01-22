module CIFAR

using CairoMakie
using ColorTypes
using ImageCore

const BATCHES_10="/data/cifar/cifar-10-binary/cifar-10-batches-bin"
const BATCHES_100="/data/cifar/cifar-100-binary"

function class_names(file::String, l::Int)
    strip.(readlines(file)[1:l]) 
end

class_names_10()=class_names("$(CIFAR.BATCHES_10)/batches.meta.txt", 10)
class_names_100()=class_names("$(CIFAR.BATCHES_100)/fine_label_names.txt", 100)

function load_cifar_bin_10(fileName::String) 
    path = "$(CIFAR.BATCHES_10)/$fileName"
    data = read(path) 
    IMG_SIZE=32 
    PIXELS=IMG_SIZE * IMG_SIZE 
    CHANNELS=3 
    RECORD_BYTES = 1 + (PIXELS * CHANNELS) 
    n = length(data) ÷ RECORD_BYTES 
    images = Array{Float32}(undef, n, IMG_SIZE, IMG_SIZE, CHANNELS) 
    labels = Vector{UInt8}(undef, n) 
    for i in 1:n 
        base = (i-1) * RECORD_BYTES + 1 
        labels[i] = data[base] + 1 
        base += 1
        img_u8 = reshape(data[base:base + PIXELS * CHANNELS - 1], IMG_SIZE, IMG_SIZE, CHANNELS) 
        img_u8 = permutedims(img_u8, (2, 1, 3)) 
        images[i, :,:, :] = Float32.(img_u8) ./ 255f0 
    end 
    
    @show size(images) 
    @show typeof(images) 
    @show size(labels) 
    return images, labels 
end

function load_cifar_bin_100(fileName::String) 
    path = "$(CIFAR.BATCHES_100)/$fileName"
    data = read(path) 
    IMG_SIZE=32 
    PIXELS=IMG_SIZE * IMG_SIZE 
    CHANNELS=3 
    RECORD_BYTES = 2 + (PIXELS * CHANNELS) 
    n = length(data) ÷ RECORD_BYTES 
    images = Array{Float32}(undef, n, IMG_SIZE, IMG_SIZE, CHANNELS) 
    labels = Vector{Tuple{UInt8,UInt8}}(undef, n) 
    for i in 1:n 
        base = (i-1) * RECORD_BYTES + 1 
        coarse = data[base + 0] + 1
        fine = data[base + 1] + 1
        labels[i] = (coarse, fine)
        base += 2
        img_u8 = reshape(data[base:base + PIXELS * CHANNELS - 1], IMG_SIZE, IMG_SIZE, CHANNELS) 
        img_u8 = permutedims(img_u8, (2, 1, 3)) 
        images[i, :,:, :] = Float32.(img_u8) ./ 255f0 
    end 
    
    @show size(images) 
    @show typeof(images) 
    @show size(labels) 
    return images, labels 
end

function plot_grid_10(images, labels, names; n_per_class=10)
    fig = Figure(size = (1600, 1400))
    Label(
        fig[0, 1:11],
        "CIFAR-10 Training Samples",
        fontsize = 22,
        tellwidth = false
    )

    for cls ∈ 1:10
        idxs = findall(labels .== cls)[1:n_per_class]

        Label(
            fig[cls, 1],
            names[cls],
            rotation = π/2,
            tellwidth = false,
            valign = :center
        )
        
        for (j, idx) ∈ enumerate(idxs)
            ax = Axis(fig[cls, j+1])
            img = permutedims(images[idx, :, :, :], (3, 1, 2))
            img = colorview(RGB, img)
            image!(ax, rotr90(img))
            hidedecorations!(ax)
            hidespines!(ax)
        end
    end

    colsize!(fig.layout, 1, CairoMakie.Fixed(60))
    for i ∈ 1:10
        colsize!(fig.layout, i+1, CairoMakie.Fixed(100))
        rowsize!(fig.layout, i, CairoMakie.Fixed(100))
    end
    
    fig
end

end
