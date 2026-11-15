# DDQN: Double DQN (+ n-step, Huber, replay, target net) in Julia, on Lunar Lander & CartPole

Gymnasium's `LunarLander-v3` (discrete: 4 actions) driven from Julia via PyCall; the Q-function is a
[Lux](https://lux.csail.mit.edu) network trained with Zygote + Adam. Built up step by step, learning
classic value-based RL toward DQN. The same learner also runs `CartPole-v1` (see *Environments*).

## What's implemented

1. **Q backed by a neural network** — semi-gradient Q-learning.
   Q_θ(s, ·) : ℝ⁸ → ℝ⁴ (`Dense(8→64, tanh) → Dense(64→4)`). For a transition (s, a, r, s′, d):

   - target  t = r + γ (1 − d) maxₐ′ Q_θ(s′, a′) — a **constant** (built from `learner.p`, not tracked by Zygote)
   - loss    ℓ = ½ (t − Q_θ(s, a))², gradient −δ ∇_θ Q_θ(s, a) only through the prediction
   - d = `terminated`, **not** `terminated || truncated`: a time-limit truncation still bootstraps

2. **ε-greedy with annealing.** ε falls linearly ε₀ → ε_min over the first `ε_episodes` episodes
   (`anneal!`), then stays at ε_min. Defaults 1 → 0.05 over 200 episodes.

3. **Experience replay.** Every transition goes into a fixed-capacity ring buffer (`Replay`, 50 000).
   After `warmup` (1000) transitions, every `train_every` (4) env steps: sample B = 64 uniformly,
   take one Adam step on

   ℒ = (1/B) Σᵢ ½ (tᵢ − Q_θ(sᵢ, aᵢ))²

   Q(sᵢ, aᵢ) is picked per column with a one-hot mask. Valid because Q-learning is off-policy: the max
   in t doesn't care which (older, more random) policy produced the transition.

4. **Target network.** A frozen copy θ⁻ of the parameters builds the target; θ⁻ ← θ every
   `target_every` (C = 1000) env steps:

   t = r + γ (1 − d) maxₐ′ Q_θ⁻(s′, a′)

   Between syncs the labels are fixed, so each period is plain regression toward 𝒯Q_θ⁻ (one step of
   approximate value iteration) instead of chasing a target that moves with every gradient step.
   Pinning the target also makes the loss meaningful within a period:
   𝔼[(y − Q_θ)²] = 𝔼[(y − 𝒯Q_θ⁻)²] (label noise) + 𝔼[(𝒯Q_θ⁻ − Q_θ)²] (fit error).
   Acting still uses the current θ; only the TD target uses θ⁻.

   **Polyak (soft) target**, `polyak=true`: instead of the hard copy, every env step
   θ⁻ ← τθ + (1 − τ)θ⁻, i.e. θ⁻ = τ Σₖ (1 − τ)ᵏ θₜ₋ₖ, an exponential moving average (low-pass filter)
   of θ. Mean lag ≈ 1/τ steps vs C/2 for the hard copy, so τ = 2/C matches it: default τ = 0.002 ↔
   C = 1000. Trade-off: no jumps at syncs, but the target is never exactly pinned (the clean
   label-noise + fit-error split above holds only approximately). DDPG / TD3 / SAC use it (τ ≈ 0.005).
   Code: `update_target!`, `fmap` over the parameter trees, blended in place.

5. **Huber loss.** Per transition, with δ = t − Q_θ(s, a) and q = min(|δ|, κ):

   h(δ) = ½q² + κ(|δ| − q)  =  ½δ² for |δ| ≤ κ,  κ(|δ| − ½κ) beyond

   so the gradient −h′(δ) ∇_θ Q_θ(s, a) uses h′(δ) = clamp(δ, −κ, κ) instead of δ. Squared loss lets one
   large residual (a ±100 crash/landing reward, or a stale target right after a sync) dominate the
   batch; Huber caps each transition's pull at κ (a large κ such as `1f6` recovers squared loss; not `Inf`,
   which gives Inf·0 = NaN). Near convergence (|δ| ≤ κ) it is still least squares.

   κ must match the reward scale. Atari's κ = 1 assumes rewards clipped to [−1, 1]; here |δ| ≫ 1, so
   κ = 1 is effectively L1 regression (fits the median, drops residual size) and it *hurt*: greedy
   snapshots +205 → +108 over 3 seeds. Default κ = 30 caps only the spikes.

6. **Double DQN.** The online network picks the next action, the target network scores it:

   a* = argmaxₐ′ Q_θ(s′, a′),   t = r + γ (1 − d) Q_θ⁻(s′, a*)

   A plain max over noisy estimates is biased up, 𝔼[maxₐ Q̂(s′,a)] ≥ maxₐ 𝔼[Q̂(s′,a)] (it picks the
   luckiest error), and bootstrapping propagates that bias backwards. The target network removes the
   *moving* label, not this. With pick and score from different networks, θ's luck in choosing a*
   isn't rewarded by θ⁻'s value of it. Code: one extra forward pass on s′ with θ (outside the
   gradient), `argmax(Qn; dims=1)` gives a CartesianIndex per column, `Qn⁻[a′]` reads θ⁻ there.

7. **n-step lookahead** (`n`, default 10). Sum n real rewards, then bootstrap once, n steps later:

   t = Rₜ⁽ⁿ⁾ + γⁿ (1 − d) V(sₜ₊ₙ),   Rₜ⁽ⁿ⁾ = Σₖ₌₀ⁿ⁻¹ γᵏ rₜ₊ₖ,   V = the Double-DQN value above

   A landing's +100 reaches n states back per update instead of 1, and the (possibly wrong) Q is
   weighted by γⁿ instead of γ. The γᵏ inside R keep the fixed point at Q* (Q = 𝔼[Σ γᵏ r]).
   The price is a small off-policy bias: rₜ₊₁ … rₜ₊ₙ₋₁ came from the ε-greedy (possibly old) policy in
   the buffer, not the greedy one; small n keeps it small. One horizon n, no λ: TD(λ) instead averages
   all n with weights (1 − λ)λⁿ⁻¹ — a choice made for its O(1) recursion, which replay doesn't need.

   Code: `on_step!` queues (s, a, r); when n are queued, `emit!` stores the oldest as one transition
   (s₀, a₀, R, s′ = sₜ₊ₙ, d, g = γⁿ). At episode end every queued one is flushed with its shorter m:
   terminated → d = 1 (no bootstrap), truncated → d = 0 and g = γᵐ. Replay stores **g** per
   transition (the γ for that transition's lookahead), so `train_batch!` just computes
   t = R + g(1 − d)V and never needs n. With n = 1, g = γ: identical to plain 1-step (verified bit-for-bit).

## Layout

| File | What |
|:---|:---|
| `src/setup.jl` | top level: includes `ddqn.jl`; `create_learner(e)`, `new_learner(e)`, `simple_train(e)`, `simple_infer(e)` |
| `src/envs.jl` | `EnvName` enum (`LUNAR_LANDER`, `CARTPOLE`), `SPECS` table, `get_env`, `get_model`, `outcome` |
| `src/ddqn.jl` | `Learner`, `action`, `on_step!`, `train_batch!`, `run_episode`, `train!`, `infer` |
| `src/replay.jl` | `Replay` ring buffer (`push!`, `sample`), included by `ddqn.jl` |
| `src/utils.jl` | helpers, included by `ddqn.jl`: `learning_curve` |
| `Project.toml` | Julia deps: Lux, Zygote, Optimisers, Functors, PyCall, ProgressMeter, UnicodePlots, Revise, CUDA/cuDNN |
| `pyproject.toml`, `uv.lock` | Python side: `gymnasium[box2d]==1.3.0` in `.venv` |
| `experiments/bench.jl` | CPU benchmark behind *Backend* (action, training step, env step, loop split) |
| `experiments/reactant-host-action.patch` | the Reactant version's host-side action, on top of commit `b75a0ca` |

## Setup

```bash
uv sync                                                     # .venv on the *system* python3.12 (see notes)
PYTHON=$PWD/.venv/bin/python julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("PyCall")'
```

## Run

```bash
julia --project          # single-threaded, see notes
```

```julia
using Revise; includet("src/setup.jl")   # tracks setup.jl → ddqn.jl → utils.jl, replay.jl
env, l = create_learner(LUNAR_LANDER)   # (Gymnasium env, fresh Learner); or CARTPOLE
train!(l, env)                       # train 1000 episodes; live progress block; returns the Learner
train!(l, env; episodes=500, plots=true)   # + learning curve in the terminal at the end
infer(l, env; episodes=100)        # greedy evaluation (ε = 0, no learning), no window
w = get_env(l.envname; render=true); infer(l, w); w.close()   # 5 greedy episodes in the pygame window
simple_train(CARTPOLE)             # train, plot → resources/cartpole.png, save → runs/cartpole.jls
simple_infer(CARTPOLE)             # load runs/cartpole.jls and fly it in a window
```

Hyperparameters are `Learner` keywords, passed through `create_learner(e; n=3, η=1f-3, …)`
(per-env overrides live in `SPECS[e].hyper`):
`γ, ε₀, ε_min, ε_episodes, η, capacity, batch, warmup, train_every, target_every, polyak, τ, κ, n`.

## Environments

`EnvName` picks the problem; everything env-specific is one row of `SPECS` in `src/envs.jl`. The DQN
code only sees the state size d and action count K, Q_θ : ℝᵈ → ℝᴷ.

| `EnvName` | Gymnasium | d | K | reward | success | solved | overrides |
|:---|:---|:---|:---|:---|:---|:---|:---|
| `LUNAR_LANDER` | `LunarLander-v3` | 8 | 4 | shaped, ±100 terminal | `:landed` | 200 | — |
| `CARTPOLE` | `CartPole-v1` | 4 | 2 | +1 per step, cap 500 | `:balanced` (reached the cap) | 475 | η = 10⁻³ |
| `ACROBOT` | `Acrobot-v1` | 6 | 3 | −1 per step, cap 500 | `:swung_up` (tip cleared the bar) | −100 | η = 10⁻³ |
| `MOUNTAIN_CAR` | `MountainCar-v0` | 2 | 3 | −1 per step, cap 200 | `:summit` (x ≥ 0.5) | −110 | — |

CartPole's success is a *truncation*: d = 0 there, so the target still bootstraps, which is right
(the pole was still up). Acrobot and MountainCar are the opposite: success is the *termination*
(d = 1, no bootstrap past the goal), and the cap is failure.

- **Acrobot**: obs = (cos θ₁, sin θ₁, cos θ₂, sin θ₂, θ̇₁, θ̇₂), torque −1 / 0 / +1 on the elbow.
  Underactuated, but −1 per step is dense enough for ε-greedy to find the goal.
- **MountainCar**: obs = (x, v), push left / none / right. **Sparse**: a random policy almost never
  reaches the flag, so every episode returns −200 and Q learns nothing useful until one succeeds.
  An exploration problem, not an approximation one: n-step returns push a success back n states per
  update. v ∈ [−0.07, 0.07] is ~15× smaller than x ∈ [−1.2, 0.6] and nothing normalizes it.

  The plain `Learner` defaults solve it; "tuning" made it worse (1000 episodes, one seed each):

  | settings | first success | successes | last-100 avg | greedy, 100 episodes |
  |:---|:---|:---|:---|:---|
  | η = 10⁻³, n = 3, ε over 400 episodes | ~ep 350 | 568 / 1000 | −143.8 | 100/100, −129.1 |
  | defaults: η = 10⁻², n = 10, ε over 200 episodes | ~ep 110 | 862 / 1000 | −115.5 | 100/100, **−101.1** |

  n = 10 vs 3 is the big one: in a sparse problem, how fast the terminal signal flows backward is
  everything. The faster ε decay also helps once that first success has propagated: a greedier
  policy keeps going back to the flag and fills the buffer with successful transitions.

Adding an env = one `EnvName` value **appended at the end** (checkpoints store `envname`) + one
`SPECS` row + an `outcome` case. Older checkpoints without `envname` load as `LUNAR_LANDER`. Files are
named after the env: `runs/<env>.jls`, `resources/<env>.png` (`lunar_lander`, `cartpole`,
`acrobot`, `mountain_car`).

## Reading the output

- **return** of an episode; Gymnasium calls the env **solved** at an average return ≥ 200 over
  100 consecutive episodes (the dashed line on the plot). A clean landing scores ≈ +200 to +300.
- **success** = the episode ended `terminated` with final reward exactly +100 (came to rest).
  The terminal reward *replaces* the step reward: +100 rest, −100 body contact or |x| ≥ 1.
- **mean ℓ** is the mean batch loss. It is *not* a progress measure (the target moves with Q);
  watch the returns.
- Training returns include ε noise and ongoing learning. The honest number is a **greedy, frozen
  evaluation**: ε = 0, no updates, fresh seeds.

## Results so far (1000 episodes, greedy eval of 50 episodes at snapshots 800 / 900 / 1000, 3 seeds)

| | train avg, last 100 | greedy snapshots (mean) | snapshots ≥ 200 |
|:---|:---|:---|:---|
| online (current transition only) | ≈ 146 | ≈ +83 | 2 / 9 |
| replay | ≈ 133 | ≈ +79 | 5 / 9 |
| replay + target network | ≈ 221 | ≈ +205 (worst +143) | 6 / 9 |
| + Huber κ = 1 | ≈ 138 | ≈ +108 | 3 / 9 |
| + Huber κ = 30 | ≈ 205 | ≈ +196 | 6 / 9 |
| + Double DQN (κ = 30) | ≈ 214 | ≈ +206 | 5 / 9 |
| + n-step, n = 3 | ≈ 214 | ≈ +219 (finals 234 / 253 / 276) | 7 / 9 |
| + n-step, n = 5 | ≈ 248 | ≈ +209 | 6 / 9 |

Double DQN lowered mean maxₐQ over buffered states from ≈ 90 to ≈ 62 on every seed (less max-bias)
with returns unchanged: with 4 actions the bias rarely changes *which* action is greedy. n = 3 reached
a 100-episode average of 200 ≈ 50 episodes sooner (≈ 442 vs ≈ 493); n = 5 trains higher but its greedy
snapshots are noisier (more off-policy bias). Differences of 10–15 points are within 3-seed noise.

Both swing hard between snapshots (−68 … +229 within one run): replay finds solved-level policies
more often but loses them again — the **moving target** (t uses the same θ being trained). The
target network fixes most of that: snapshots swing far less (worst +143 vs −68).
Without annealing (fixed ε = 0.15) online learning plateaued at ≈ +50–70; with η = 1e-3 instead of
1e-2, nothing much was learned in 1000 episodes.

## Backend: Zygote, not Reactant + Enzyme

The learner runs eagerly: Lux on plain CPU arrays, gradients from Zygote. A Reactant + Enzyme version,
with the training step (both target passes, Huber loss, reverse pass, Adam) compiled to one XLA
program, is about 15 % slower here, so this one stays.

Per env step, Lunar Lander, `julia -t 1`, CPU, one training step (B = 256) every 4 env steps, timed
inside the real loop for 60 s each:

| | Zygote (this code) | Reactant, compiled action | Reactant, host-side action |
|:---|---:|---:|---:|
| action ($Q_\theta(s, \cdot)$, one state) | 1.1 μs | 68.7 μs | 11.0 μs |
| `env.step` (Gymnasium via PyCall) | 51 μs | 78 μs | 57 μs |
| `on_step!` (store + ¼ training step) | 49 μs | 63 μs | 51 μs |
| one training step, alone | 140–156 μs | 152–154 μs | 124–140 μs |
| **env steps/s** | **9 100** | **4 500** | **7 800** |

- **Every call is overhead.** An action is $8 \cdot 64 + 64 \cdot 4 = 768$ multiply-adds and a training
  step ≈ 2 MFLOP: a few μs of arithmetic on one core. A call costs ~100–150 μs either way, Zygote's
  tape and allocations or XLA's buffer copies and dispatch, so compiling the training step makes it
  no faster.
- **Acting is where Reactant loses.** Through XLA, one state's forward pass costs 35–70 μs instead of
  1 μs, more than the env step. Running it as plain Lux on a host copy of θ, refreshed by `copyto!`
  (~17 μs) after each training step, recovers most of it: the "host-side action" column.
- **`env.step` is slower with Reactant loaded** (51 → 57–78 μs), likely XLA's own CPU threads
  competing with Python's Box2D (not verified).
- **The crossover.** With a fixed per-call cost $t_0 \approx 100$ μs, the time per call is
  $t \approx t_0 + W/R$ for $W$ FLOP at rate $R$; compilation raises $R$ (fusion, multithreaded kernels)
  but only pays once $W/R \gtrsim t_0$, i.e. $W \gtrsim$ 5–10 MFLOP per call on this CPU (an estimate,
  not measured). The ImageNet-64 ResNet-18 step is ~7 TFLOP, far past it, and gains 2.8× from
  Reactant on the GPU. Bigger nets or batches, K gradient steps per call, or vectorized envs (N
  states per action call) would move DDQN past it too.
- **Where the Reactant version is:** commit `b75a0ca` (`ddqn enzyme`), with the host-side action on top
  as `experiments/reactant-host-action.patch` (`git apply` on that commit). The benchmark builds the
  same learner in both versions and times `action`, `train_batch!`, `env.step` and the loop.

## Next (classic "max out Q with a NN", in order)

1. Dueling heads, prioritized replay, distributional (C51), noisy nets (the rest of Rainbow).

Plus: bigger net (2×128 ReLU), anneal ε by steps, keep the best greedy snapshot; compare Polyak
τ = 0.002 against the hard copy C = 1000.

## Notes / gotchas

- **Python must be the system `/usr/bin/python3.12`**, not uv's managed CPython: its `libpython`
  needs an executable stack, which glibc ≥ 2.41 refuses to `dlopen` (`pyproject.toml` pins
  `python-preference = "only-system"`).
- **PyCall's Python is global to the Julia depot** (`Pkg.build("PyCall")` with `PYTHON=…`). After
  renaming/moving this folder, rebuild it.
- **Threads**: `~/.zshrc` sets `JULIA_NUM_THREADS=auto` (48). Nothing here uses threads, and it
  exposed a Julia 1.12.3 JIT race with the REPL's background hint task (segfault). Run with
  `JULIA_NUM_THREADS=1` or disable hints: `Base.active_repl.options.hint_tab_completes = false`.
- **First call compiles** (~10–20 s on CPU); stay in one REPL session — Revise recompiles only what
  you edit.
- **CPU vs GPU**: `tdev` selects the training device. For this tiny net the CPU is faster (GPU time
  is kernel-launch overhead); switch `tdev = gdev` once batches / nets are bigger.
