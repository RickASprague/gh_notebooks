module ImageNet

using PythonCall

const IMAGENET_32 = "/data/imagenet/Imagenet32"
const IMAGENET_64 = "/data/imagenet/Imagenet64"

function convert(obj, size)
    img = pyconvert(Array, obj)
    img = reshape(img, (size, size, 3))
    img = permutedims(img, (2, 1, 3)) 
    img = Float32.(img) ./ 255f0 
    img
end

function load(fileName::String, sz)
    pickle = pyimport("pickle")
    builtins = pyimport("builtins")
    f = builtins.open(fileName, "rb")
    
    try
        obj = pickle.load(f)
        labels = pyconvert(Vector{Int}, obj["labels"])
        μ = convert(obj["mean"], sz)
        n = length(obj["data"])
        images = Array{Float32}(undef, n, sz, sz, 3) 
        data = obj["data"]
        for i in 1:n
            images[i, :, :, :] = convert(data[i-1], sz)
        end
        return labels, images, μ
    finally
        f.close()
    end
    nothing
end

load_32(fileName::String)=load("$IMAGENET_32/$fileName", 32)

load_64(fileName::String)=load("$IMAGENET_64/$fileName", 64)

end
