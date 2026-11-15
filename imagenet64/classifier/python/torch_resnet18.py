"""PyTorch reference for the classifier's ResNet-18, to compare against the Julia/Lux runs.

Same network as ../src/model.jl `build_resnet18` (11 681 832 parameters), same recipe (Adam 1e-3,
cross-entropy, batch 512, 10 epochs), same data (the ../../build shards, read directly), and the same
TensorBoard tags in ../build/tensorboard, so the runs overlay in one view.

  # throughput of six configurations on synthetic batches (no data loading)
  uv run --no-project --index https://download.pytorch.org/whl/cu130 \
      --with torch --with numpy --with tensorboard python -I torch_resnet18.py bench

  # full training run on the real data
  uv run ... python -I torch_resnet18.py train --precision bf16 --layout cl --compile [--epochs 10]

Data: each .ser array is [Int32 ndims][Int32 dims...][payload], arrays concatenated per file. Julia
writes images as (64, 64, 3, N) column-major, which NumPy reads as (N, 3, 64, 64) C-order: NCHW with
the two image axes swapped relative to Julia's, the same for train and val. Labels are 1..1000 in
the files, 0..999 here.
"""
import argparse, os, queue, threading, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

HERE = os.path.dirname(os.path.abspath(__file__))
CLS = os.path.dirname(HERE)                                  # imagenet64/classifier
DATA = os.path.join(os.path.dirname(CLS), "build")          # imagenet64/build
TB_DIR = os.path.join(CLS, "build", "tensorboard")
RESULTS = os.path.join(CLS, "results")
NCLASS, SUBSET_EVERY, SUBSET_N, EVAL_BATCH = 1000, 500, 5000, 1000

# ── model ────────────────────────────────────────────────────────────────────────────────────────
def conv_bn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))

class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(conv_bn(cin, cout, stride),
                                  nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.short = (nn.Identity() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
    def forward(self, x):
        return F.relu(self.body(x) + self.short(x))

def stage(cin, cout, stride):
    return nn.Sequential(Block(cin, cout, stride), Block(cout, cout))

def resnet18_64(nclass=NCLASS):
    return nn.Sequential(conv_bn(3, 64), stage(64, 64, 1), stage(64, 128, 2), stage(128, 256, 2),
                         stage(256, 512, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(512, nclass))

# ── data ─────────────────────────────────────────────────────────────────────────────────────────
def read_arrays(path, dtype, n):
    """Memory-map n consecutive [ndims][dims][payload] arrays from a .ser file (dims reversed to C order)."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    out, pos = [], 0
    for _ in range(n):
        nd = int(np.frombuffer(raw[pos:pos+4], np.int32)[0]); pos += 4
        dims = np.frombuffer(raw[pos:pos+4*nd], np.int32).astype(int); pos += 4*nd
        count = int(np.prod(dims)); nbytes = count * np.dtype(dtype).itemsize
        out.append(np.ndarray(tuple(dims[::-1]), dtype=dtype, buffer=raw, offset=pos)); pos += nbytes
    return out

def load_train():
    n = int(np.fromfile(os.path.join(DATA, "train", "count.ser"), np.int64)[0])
    Xs = read_arrays(os.path.join(DATA, "train", "Xs.ser"), np.float32, n)
    ys = [np.array(y) - 1 for y in read_arrays(os.path.join(DATA, "train", "ys.ser"), np.int64, n)]
    return Xs, ys

def load_val():
    X, = read_arrays(os.path.join(DATA, "val", "Xs.ser"), np.float32, 1)
    y, = read_arrays(os.path.join(DATA, "val", "ys.ser"), np.int64, 1)
    return X, np.array(y) - 1

class Loader:
    """Full batches of B contiguous images; chunks of 16 consecutive batches shuffled across all
    shards every epoch (as ../../utils/dataloader.jl). A reader thread fills pinned host buffers
    ahead of the GPU; copies to the device are asynchronous."""
    def __init__(self, Xs, ys, B, chunk=16, nbuf=4, limit=None, seed=0):
        self.Xs, self.ys, self.B, self.limit = Xs, ys, B, limit
        starts = [(k, i) for k, X in enumerate(Xs) for i in range(0, X.shape[0] - B + 1, B)]
        self.chunks = [starts[j:j+chunk] for j in range(0, len(starts), chunk)]
        self.n = min(len(starts), limit or len(starts))
        self.rng = np.random.default_rng(seed)
        self.pinned = torch.empty((nbuf, B, 3, 64, 64), dtype=torch.float32).pin_memory()
        self.pinned_np = self.pinned.numpy()
        self.nbuf = nbuf
    def __len__(self): return self.n
    def __iter__(self):
        order = [b for c in self.rng.permutation(len(self.chunks)) for b in self.chunks[c]][:self.n]
        free, full = queue.Queue(), queue.Queue(maxsize=self.nbuf)
        for i in range(self.nbuf): free.put(i)
        def reader():
            for k, i in order:
                j = free.get()
                np.copyto(self.pinned_np[j], self.Xs[k][i:i+self.B])
                full.put((j, self.ys[k][i:i+self.B]))
            full.put(None)
        threading.Thread(target=reader, daemon=True).start()
        prev = None
        while (item := full.get()) is not None:
            j, y = item
            x = self.pinned[j].cuda(non_blocking=True)
            ev = torch.cuda.Event(); ev.record()
            yl = torch.from_numpy(y).cuda(non_blocking=True)
            if prev is not None:                      # previous buffer is reusable once its copy ran
                prev[1].synchronize(); free.put(prev[0])
            prev = (j, ev)
            yield x, yl
        if prev is not None:
            prev[1].synchronize(); free.put(prev[0])

# ── training ─────────────────────────────────────────────────────────────────────────────────────
def make(precision, layout, compiled, lr=1e-3):
    torch.manual_seed(0)
    mf = torch.channels_last if layout == "cl" else torch.contiguous_format
    model = resnet18_64().cuda().to(memory_format=mf)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bf16 = precision == "bf16"
    def step(x, y):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        return loss.detach()
    return model, (torch.compile(step) if compiled else step), mf, bf16

@torch.no_grad()
def evaluate(model, X, y, mf, bf16):
    model.eval()
    top1 = top5 = 0
    for i in range(0, X.shape[0], EVAL_BATCH):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i+EVAL_BATCH])).cuda().to(memory_format=mf)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            z = model(xb).float()
        t5 = z.topk(5, dim=1).indices.cpu().numpy()
        yb = y[i:i+EVAL_BATCH]
        top1 += int((t5[:, 0] == yb).sum()); top5 += int((t5 == yb[:, None]).any(1).sum())
    model.train()
    return top1 / X.shape[0], top5 / X.shape[0]

def name(precision, layout, compiled):
    return f"torch-resnet18-{precision}-{layout}-{'compile' if compiled else 'eager'}"

def bench(B=512, warmup=30, steps=100):
    torch.backends.cudnn.benchmark = True
    configs = [("fp32", "nchw", False), ("fp32", "cl", False), ("bf16", "nchw", False),
               ("bf16", "cl", False), ("fp32", "cl", True), ("bf16", "cl", True)]
    print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(),
          "cudnn.allow_tf32", torch.backends.cudnn.allow_tf32, flush=True)
    for p, l, c in configs:
        model, step, mf, _ = make(p, l, c)
        x = torch.rand(B, 3, 64, 64, device="cuda").to(memory_format=mf)
        y = torch.randint(0, NCLASS, (B,), device="cuda")
        tb = SummaryWriter(os.path.join(TB_DIR, name(p, l, c) + "-synthetic"))
        t0 = time.time()
        for _ in range(warmup): step(x, y)
        torch.cuda.synchronize(); warm = time.time() - t0
        t0 = time.time(); tw = t0
        for s in range(1, steps + 1):
            loss = step(x, y)
            if s % 10 == 0:
                torch.cuda.synchronize(); now = time.time()
                tb.add_scalar("train/img_per_s", 10 * B / (now - tw), s); tw = now
                tb.add_scalar("train/loss", float(loss), s)
        torch.cuda.synchronize(); dt = time.time() - t0
        tb.close()
        print(f"{p:5s} {l:5s} {'compile' if c else 'eager':8s}  {B*steps/dt:8.0f} img/s  "
              f"{1e3*dt/steps:7.1f} ms/step   (warm-up {warm:5.1f} s)", flush=True)
        del model, step; torch.cuda.empty_cache()

def train(precision, layout, compiled, epochs, limit=None, B=512):
    torch.backends.cudnn.benchmark = True
    Xs, ys = load_train(); Xv, yv = load_val()
    loader = Loader(Xs, ys, B, limit=limit)
    model, step, mf, bf16 = make(precision, layout, compiled)
    run = name(precision, layout, compiled) + "-b512-" + time.strftime("%Y%m%d-%H%M")
    tb = SummaryWriter(os.path.join(TB_DIR, run))
    print(f"train {run}: {len(loader)} steps/epoch, params {sum(p.numel() for p in model.parameters())}", flush=True)
    rows, gstep, t_rate, n_rate, losses = [], 0, time.time(), 0, []
    for epoch in range(1, epochs + 1):
        eloss = 0.0; nb = 0; t0 = time.time(); t_sub = 0.0
        for x, y in loader:
            loss = step(x.to(memory_format=mf), y)
            gstep += 1; nb += 1; n_rate += 1
            if gstep % 10 == 0 or nb == len(loader):          # read the loss back every 10 steps
                l = float(loss); losses.append((gstep, l)); eloss += l * 10
                tb.add_scalar("train/loss", l, gstep)
            if n_rate == 50:
                torch.cuda.synchronize()
                tb.add_scalar("train/img_per_s", 50 * B / (time.time() - t_rate), gstep)
                t_rate = time.time(); n_rate = 0
            if gstep % SUBSET_EVERY == 0:
                te = time.time()
                s1, s5 = evaluate(model, Xv[:SUBSET_N], yv[:SUBSET_N], mf, bf16)
                tb.add_scalar("val_subset/top1", 100*s1, gstep); tb.add_scalar("val_subset/top5", 100*s5, gstep)
                de = time.time() - te; t_sub += de; t_rate += de
        torch.cuda.synchronize()
        dt = time.time() - t0 - t_sub
        top1, top5 = evaluate(model, Xv, yv, mf, bf16)
        eloss /= max(nb, 1)
        rows.append((epoch, eloss, top1, top5, dt, nb * B / dt))
        for tag, v in (("val/top1", 100*top1), ("val/top5", 100*top5), ("train/epoch_loss", eloss), ("time/epoch_s", dt)):
            tb.add_scalar(tag, v, gstep)
        t_rate = time.time(); n_rate = 0
        print(f"Epoch {epoch:2d}/{epochs} | loss {eloss:.4f} | val top-1 {100*top1:5.2f}% top-5 {100*top5:5.2f}% "
              f"| {dt:6.1f}s | {nb*B/dt:6.0f} img/s | {nb} batches", flush=True)
    tb.close()
    tag = name(precision, layout, compiled).removeprefix("torch-resnet18-")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"train_epochs-torch-{tag}.csv"), "w") as f:
        f.write("epoch,loss,val_top1,val_top5,seconds,img_per_s\n")
        for r in rows: f.write(",".join(str(v) for v in r) + "\n")
    with open(os.path.join(RESULTS, f"train_loss-torch-{tag}.csv"), "w") as f:
        f.write("step,loss\n")
        for s, l in losses: f.write(f"{s},{l}\n")
    print(f"best val top-1 {100*max(r[2] for r in rows):.2f}%  top-5 {100*max(r[3] for r in rows):.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "train", "check"])
    ap.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--layout", default="cl", choices=["nchw", "cl"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)       # max steps per epoch (smoke tests)
    a = ap.parse_args()
    if a.mode == "check":                                      # CPU only: data and model sanity
        Xs, ys = load_train(); Xv, yv = load_val()
        print("train shards", len(Xs), "images", sum(x.shape[0] for x in Xs), "shape", Xs[0].shape,
              "labels", min(int(y.min()) for y in ys), max(int(y.max()) for y in ys))
        print("val", Xv.shape, "labels", int(yv.min()), int(yv.max()), "pixel range",
              float(Xv[:100].min()), float(Xv[:100].max()))
        print("params", sum(p.numel() for p in resnet18_64().parameters()))
    elif a.mode == "bench":
        bench()
    else:
        train(a.precision, a.layout, a.compile, a.epochs, a.limit)
