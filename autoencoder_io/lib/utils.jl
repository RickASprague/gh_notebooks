module Utils

using Mmap
export serialize, deserialize, MMapReader
import Base: close
using ColorTypes
using ImageCore

function makie_image(img)
    img = permutedims(img, (3, 1, 2))
    img = colorview(RGB, img)
    rotr90(img)    
end

mutable struct MMapReader 
	io::IOStream
	buf::Vector{UInt8}
	pos::Int

	function MMapReader(path::String)
		io = open(path, "r")
		buf = Mmap.mmap(io, Vector{UInt8}, filesize(path))
		return new(io, buf, 0)
	end
end

function close(io::MMapReader)
	close(io.io)
end

function deserialize(r::MMapReader, ::Type{Array{T}}) where {T}
    @assert isbitstype(T)

	@inline function read_i32()
		i = r.pos + 1
		x = reinterpret(Int32, @view r.buf[i:(i+3)])[1]
		r.pos += 4
		return Int(x)
	end

    nd = read_i32()
    dims = Vector{Int}(undef, nd)
    @inbounds for k in 1:nd
        dims[k] = read_i32()
    end

    nelem = prod(dims)
    nbytes = nelem * sizeof(T)

    # payload slice in the mapped byte buffer
    start = r.pos + 1
    stop  = r.pos + nbytes
    payload = @view r.buf[start:stop]

    A = reshape(reinterpret(T, payload), Tuple(dims)...)

    # advance past payload
    r.pos += nbytes

    return A
end

# arrays
function serialize(io::IO, array::Array{T,N}) where {T,N}
    @assert isbitstype(T)

    write(io, Int32(N))
    @inbounds for i in 1:N
        write(io, Int32(size(array, i)))
    end

    write(io, array)   # raw payload
end

function deserialize(io::IO, ::Type{Array{T}}) where {T}
    @assert isbitstype(T)

    N = Int(read(io, Int32))

    dims = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        dims[i] = Int(read(io, Int32))
    end

    A = Array{T}(undef, Tuple(dims)...)
    read!(io, A)
    return A
end

# scalars
function serialize(io::IO, x::T) where {T}
    @assert isbitstype(T)
    write(io, x)
end

function deserialize(io::IO, ::Type{T}) where {T}
    @assert isbitstype(T)
    return read(io, T)
end

end
