# ImageNet-64 Preprocessing: Pickles → Memory-Mapped Binary

ImageNet-64 ships as Python pickles of NumPy arrays. Unpickling and converting all 1.28M images
from Julia (through PythonCall) takes about 3 minutes, and doing that on every pass is the IO cost
this project removes. This step pays it once: each pickle becomes a flat binary shard that
[`MMapReader`](lib/utils.jl) can map at training time without reading or parsing anything.

Run: `julia --project src/preprocess.jl` (needs the raw pickles under `/data/imagenet/Imagenet64`).

## Format

Every array is written as

$$\underbrace{n_d}_{\text{Int32}}\;\underbrace{d_1\,\cdots\,d_{n_d}}_{\text{Int32}}\;\underbrace{x_1\,x_2\,\cdots\,x_{\prod d_i}}_{\text{raw }T\text{, column-major}}$$

and shards are concatenated back to back in one file. Reading shard $k$ is then pointer
arithmetic: `MMapReader` keeps a byte offset into the mapped buffer, reads the header, and returns
`reshape(reinterpret(T, view(buf, a:b)), dims)` — a view into the mapping, not a copy. The pages
are faulted in by the OS only when training touches them.

The pickles store images as $(N, 64, 64, 3)$, row-major. Julia is column-major and Lux expects
WHCN, so each shard is permuted to $(64, 64, 3, N)$ before it is written. That way the slice a batch
needs, `X[:, :, :, i:j]`, is one contiguous run of $64\cdot 64\cdot 3\cdot 4 = 49{,}152$ bytes per image.

## Output

| file | contents |
|---|---|
| `build/train/count.ser` | number of shards (10) |
| `build/train/Xs.ser` | images, Float32 in $[0,1]$, $(64,64,3,N_k)$ per shard — 63 GB |
| `build/train/ys.ser` | Int64 class labels per shard |
| `build/train/μs.ser` | per-shard channel means $(64,64,3)$ |
| `build/val/Xs.ser`, `ys.ser` | 50,000 validation images, one shard, no mean |

The training set is 10 pickles (2 parts × 5 batches) of about 128k images each, 1,281,167 in total.
