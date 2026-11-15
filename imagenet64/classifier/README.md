# ImageNet-64 classifier

A supervised classifier on ImageNet-64 (1000 classes, $64 \times 64$ images). It shares the data and
the loaders with the other `imagenet64` projects (`../utils/data.jl`, shards in `../build`); the
network, loss, training loop and evaluation are its own. A deliberately simple baseline, to be enhanced.

## Model

A VGG-style CNN, 5 200 296 parameters:

```
64²×3 → [64, 64] → maxpool → 32²×64 → [128, 128] → maxpool → 16²×128 → [256, 256] → maxpool → 8²×256
      → [512, 512] → global mean → 512 → dropout(0.3) → Dense → 1000 logits
```

Each `[c, c]` is a stage of two $3 \times 3$, stride-1 convolutions, each followed by BatchNorm and
ReLU (the convolutions have no bias; BatchNorm's $\beta$ replaces it).

- **Stride-1 convs inside a stage** compute features at full resolution before anything is discarded;
  two stacked $3 \times 3$ convs give a $5 \times 5$ receptive field per stage.
- **Max-pooling between stages** keeps the strongest response in each $2 \times 2$ window: the
  translation invariance a classifier wants (and an autoencoder must avoid).
- **Width doubles as resolution halves**, so every stage costs about the same FLOPs:
  $H W C^2$ is constant when $H, W$ halve and $C$ doubles.
- **Global average pooling plus one linear layer** is a small head, so the convolutional stack has to
  make the classes linearly separable.

## Loss

Softmax cross-entropy from the logits $z \in \mathbb{R}^{1000 \times B}$ with one-hot targets $Y$:

$$
\ell = -\frac{1}{B} \sum_{i=1}^{B} \sum_{c=1}^{1000} Y_{ci} \log \operatorname{softmax}(z)_{ci},
\qquad
\operatorname{softmax}(z)_{ci} = \frac{e^{z_{ci}}}{\sum_{c'} e^{z_{c'i}}}
$$

Computed as a dense product $Y \odot \log\operatorname{softmax}(z)$ rather than by indexing, so the GPU
needs no gather or scatter. At initialization the prediction is near uniform, so
$\ell \approx \ln 1000 = 6.908$: the first sanity check.

## Data

The common ImageNet-64 shards (`../build/{train,val}`, 59 GB + 2.3 GB, written by
`../utils/preprocess.jl`, loaded by `../utils/data.jl`): 1 281 167 train images in 10
mmap'd shards and 50 000 val, labels $1 \ldots 1000$. Batches come from the common `DataLoader`
(`../utils/dataloader.jl`), and the training loop is just `for (x, y) ∈ loader`:

- each batch is **one** `read!` of $B \cdot 64 \cdot 64 \cdot 3 \cdot 4$ bytes straight from `Xs.ser`
  (no mmap page faults);
- a reader thread reads ahead into 4 reusable host buffers while the GPU trains (run `julia -t 4`);
- shuffling is by **chunks** of 16 consecutive batches: chunk order is reshuffled across all shards
  every epoch, batches inside a chunk are read in order, so reads stay large and sequential;
- ragged shard tails are dropped.

The same code is fast on both machines. Loading alone (CPU, warm cache, batch 1024) on the
Threadripper: **42 300 img/s** vs 13 500 img/s for the old mmap copy, far above what the GPU trains at.
On the NAS (60 GB of shards on spinning disks, 30 GB of RAM) it took training on the TITAN X from
~450 to ~950 img/s, GPU-bound.

## Run

```bash
julia -t 4 --project src/train.jl              # 10 epochs, batch 1024, Adam 1e-3
julia -t 4 --project src/train.jl 2 30         # smoke test: 2 epochs × 30 steps
BATCH=512 julia -t 4 --project src/train.jl    # batch 512 (fits the NAS's 12 GB TITAN X)
```

After every epoch: val top-1 / top-5 (BatchNorm and dropout in test mode), a checkpoint in
`build/checkpoint/model_<epoch>.jld2`; at the end `results/train_{epochs,loss}.csv`.

## Results

| Run | val top-1 | val top-5 |
|:---|:---|:---|
| 10 epochs, batch 512, Adam $10^{-3}$ (NAS, TITAN X, 3.9 h) | **45.72 %** | **70.71 %** |
| smoke test, 60 steps | 1.34 % | 4.89 % |
| chance | 0.10 % | 0.50 % |

The 10-epoch run, on the 50 000 validation images after each epoch:

| epoch | train loss | val top-1 | val top-5 |
|---:|---:|---:|---:|
| 1 | 4.568 | 24.07 % | 48.06 % |
| 2 | 3.226 | 33.18 % | 58.92 % |
| 3 | 2.788 | 37.55 % | 63.36 % |
| 4 | 2.541 | 37.89 % | 63.69 % |
| 5 | 2.376 | 42.12 % | 67.55 % |
| 6 | 2.248 | 43.31 % | 69.03 % |
| 7 | 2.143 | 43.90 % | 69.22 % |
| 8 | 2.058 | 44.92 % | 70.17 % |
| 9 | 1.984 | 44.67 % | 69.84 % |
| 10 | 1.914 | 45.72 % | 70.71 % |

- **Not converged.** Train loss falls every epoch and val top-1 still gains ~1 point per epoch at the
  end, so more epochs (or a learning-rate decay) should help.
- **Throughput:** ~938 img/s, 2 500 steps and ~1 365 s per epoch, GPU-bound on the TITAN X
  (sm_61, FP32, no tensor cores).
- **Files:** `results/nas-titanx-20261115/` (`train_epochs.csv`, `train_loss.csv`, `train.log`) and
  `build/checkpoint-nas-titanx-20261115/model_{1…10}.jld2`. The suffix keeps them apart from the
  Threadripper's own `results/` and `build/checkpoint/`.

## Files

| Path | What |
|:---|:---|
| `src/model.jl` | the network (`build_model`) |
| `src/train.jl` | loss, gradient step, evaluation, the training loop, entry point |
| `build/`, `results/` | generated (git-ignored) |
| `../utils/data.jl` | common shard readers: `load_train`, `load_val` (included) |
| `../utils/dataloader.jl` | common `DataLoader`: read-ahead, chunk-shuffled batches (included) |
| `../build/{train,val}` | common ImageNet-64 shards |
