# DDQN: Double DQN (+ n-step, Huber, replay, target net) in Julia, on Lunar Lander & CartPole

Gymnasium's `LunarLander-v3` (discrete: 4 actions) driven from Julia via PyCall; the Q-function is a
[Lux](https://lux.csail.mit.edu) network trained with Adam, compiled to XLA by [Reactant](https://github.com/EnzymeAD/Reactant.jl)
and differentiated by [Enzyme](https://enzyme.mit.edu) (see *Reactant + Enzyme*). Built up step by step, learning
classic value-based RL toward DQN. The same learner also runs `CartPole-v1` (see *Environments*).

## What's implemented

1. **Q backed by a neural network** — semi-gradient Q-learning.
   $Q_\theta(s, \cdot) : \mathbb{R}^8 \to \mathbb{R}^4$ (`Dense(8→64, tanh) → Dense(64→4)`). For a
   transition $(s, a, r, s', d)$:

   - target $t = r + \gamma (1 - d) \max_{a'} Q_\theta(s', a')$ — a **constant** (`ignore_derivatives`: no gradient flows through it)
   - loss $\ell = \tfrac{1}{2} \left(t - Q_\theta(s, a)\right)^2$, gradient $-\delta \, \nabla_\theta Q_\theta(s, a)$ only through the prediction
   - $d$ = `terminated`, **not** `terminated || truncated`: a time-limit truncation still bootstraps

2. **ε-greedy with annealing.** $\varepsilon$ falls linearly $\varepsilon_0 \to \varepsilon_{\min}$ over
   the first `ε_episodes` episodes (`anneal!`), then stays at $\varepsilon_{\min}$. Defaults
   $1 \to 0.05$ over 200 episodes.

3. **Experience replay.** Every transition goes into a fixed-capacity ring buffer (`Replay`, 50 000).
   After `warmup` (1000) transitions, every `train_every` (4) env steps: sample $B = 64$ uniformly,
   take one Adam step on

   $$
   \mathcal{L} = \frac{1}{B} \sum_i \tfrac{1}{2} \left(t_i - Q_\theta(s_i, a_i)\right)^2
   $$

   $Q(s_i, a_i)$ is picked per column with a one-hot mask. Valid because Q-learning is off-policy: the
   max in $t$ doesn't care which (older, more random) policy produced the transition.

4. **Target network.** A frozen copy $\theta^-$ of the parameters builds the target; $\theta^- \leftarrow \theta$
   every `target_every` ($C = 1000$) env steps:

   $$
   t = r + \gamma (1 - d) \max_{a'} Q_{\theta^-}(s', a')
   $$

   Between syncs the labels are fixed, so each period is plain regression toward $\mathcal{T} Q_{\theta^-}$
   (one step of approximate value iteration) instead of chasing a target that moves with every
   gradient step. Pinning the target also makes the loss meaningful within a period:

   $$
   \mathbb{E}\left[(y - Q_\theta)^2\right] = \underbrace{\mathbb{E}\left[(y - \mathcal{T} Q_{\theta^-})^2\right]}_{\text{label noise}} + \underbrace{\mathbb{E}\left[(\mathcal{T} Q_{\theta^-} - Q_\theta)^2\right]}_{\text{fit error}}
   $$

   Acting still uses the current $\theta$; only the TD target uses $\theta^-$.

   **Polyak (soft) target**, `polyak=true`: instead of the hard copy, every env step

   $$
   \theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-, \qquad \text{i.e.} \quad \theta^- = \tau \sum_k (1 - \tau)^k \theta_{t-k},
   $$

   an exponential moving average (low-pass filter) of $\theta$. Mean lag $\approx 1/\tau$ steps vs
   $C/2$ for the hard copy, so $\tau = 2/C$ matches it: default $\tau = 0.002 \leftrightarrow C = 1000$.
   Trade-off: no jumps at syncs, but the target is never exactly pinned (the clean label-noise +
   fit-error split above holds only approximately). DDPG / TD3 / SAC use it ($\tau \approx 0.005$).
   Code: `update_target!`, `fmap` over the parameter trees; the blend is one compiled program (`Polyak`).

5. **Huber loss.** Per transition, with $\delta = t - Q_\theta(s, a)$ and $q = \min(|\delta|, \kappa)$:

   $$
   h(\delta) = \tfrac{1}{2} q^2 + \kappa (|\delta| - q) =
   \begin{cases} \tfrac{1}{2} \delta^2 & |\delta| \le \kappa \\ \kappa \left(|\delta| - \tfrac{1}{2}\kappa\right) & \text{otherwise} \end{cases}
   $$

   so the gradient $-h'(\delta) \, \nabla_\theta Q_\theta(s, a)$ uses $h'(\delta) = \operatorname{clamp}(\delta, -\kappa, \kappa)$
   instead of $\delta$. Squared loss lets one large residual (a ±100 crash/landing reward, or a stale
   target right after a sync) dominate the batch; Huber caps each transition's pull at $\kappa$ (a large
   $\kappa$ such as `1f6` recovers squared loss; not `Inf`, which gives $\infty \cdot 0 = \text{NaN}$).
   Near convergence ($|\delta| \le \kappa$) it is still least squares.

   $\kappa$ must match the reward scale. Atari's $\kappa = 1$ assumes rewards clipped to $[-1, 1]$; here
   $|\delta| \gg 1$, so $\kappa = 1$ is effectively L1 regression (fits the median, drops residual size)
   and it *hurt*: greedy snapshots +205 → +108 over 3 seeds. Default $\kappa = 30$ caps only the spikes.

6. **Double DQN.** The online network picks the next action, the target network scores it:

   $$
   a^* = \arg\max_{a'} Q_\theta(s', a'), \qquad t = r + \gamma (1 - d) \, Q_{\theta^-}(s', a^*)
   $$

   A plain max over noisy estimates is biased up,
   $\mathbb{E}\left[\max_a \hat Q(s', a)\right] \ge \max_a \mathbb{E}\left[\hat Q(s', a)\right]$ (it picks
   the luckiest error), and bootstrapping propagates that bias backwards. The target network removes
   the *moving* label, not this. With pick and score from different networks, $\theta$'s luck in choosing
   $a^*$ isn't rewarded by $\theta^-$'s value of it. Code: one extra forward pass on $s'$ with $\theta$
   (outside the gradient). The pick is a mask rather than an argmax + gather, which compiles to plain
   elementwise ops and reductions:

   $$
   M_{a i} = \mathbb{1}\left[Q_\theta(s'_i, a) = \max_{a'} Q_\theta(s'_i, a')\right], \qquad
   V_i = \frac{\sum_a M_{a i} \, Q_{\theta^-}(s'_i, a)}{\sum_a M_{a i}}
   $$

   which is $Q_{\theta^-}(s'_i, a^*_i)$ whenever the max is unique (an exact tie between float outputs
   would average over the tied actions).

7. **n-step lookahead** (`n`, default 10). Sum $n$ real rewards, then bootstrap once, $n$ steps later:

   $$
   t = R_t^{(n)} + \gamma^n (1 - d) \, V(s_{t+n}), \qquad R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k}
   $$

   with $V$ the Double-DQN value above. A landing's +100 reaches $n$ states back per update instead of
   1, and the (possibly wrong) $Q$ is weighted by $\gamma^n$ instead of $\gamma$. The $\gamma^k$ inside
   $R$ keep the fixed point at $Q^*$ ($Q = \mathbb{E}\left[\sum_k \gamma^k r\right]$). The price is a small
   off-policy bias: $r_{t+1}, \ldots, r_{t+n-1}$ came from the ε-greedy (possibly old) policy in the
   buffer, not the greedy one; small $n$ keeps it small. One horizon $n$, no $\lambda$: TD($\lambda$)
   instead averages all $n$ with weights $(1 - \lambda)\lambda^{n-1}$ — a choice made for its $O(1)$
   recursion, which replay doesn't need.

   Code: `on_step!` queues $(s, a, r)$; when $n$ are queued, `emit!` stores the oldest as one transition
   $(s_0, a_0, R, s' = s_{t+n}, d, g = \gamma^n)$. At episode end every queued one is flushed with its
   shorter $m$: terminated → $d = 1$ (no bootstrap), truncated → $d = 0$ and $g = \gamma^m$. Replay
   stores **$g$** per transition (the $\gamma$ for that transition's lookahead), so `train_batch!` just
   computes $t = R + g(1 - d)V$ and never needs $n$. With $n = 1$, $g = \gamma$: identical to plain
   1-step (verified bit-for-bit).

## Reactant + Enzyme

Each piece of device work is a single compiled XLA program, traced on first use and cached:

| program | what | built by |
|:---|:---|:---|
| training step | both target forward passes on $s'$, the Huber loss, Enzyme's reverse pass, Adam's update | `Training.single_train_step!(AutoEnzyme(), DDQNLoss(κ), data, ts)` |
| action | $Q_\theta(s, \cdot)$ on a $d \times 1$ batch | `@compile model(x, θ, st)`, cached in `learner.xla[:q]` |
| Polyak | $\theta^- \leftarrow \tau\theta + (1 - \tau)\theta^-$ | `@compile Polyak(τ)(θ⁻, θ)`, cached under `(:polyak, τ)` |

- **The objective is a Lux objective**, `(model, θ, st, data) → (ℒ, st′, stats)`, with
  `data = (s, mask, r, g, d, s′, θ⁻)`. The target is built inside it and wrapped in
  `Reactant.ignore_derivatives`, so the semi-gradient is exact and Enzyme doesn't differentiate the
  two extra forward passes.
- **θ lives in the TrainState** (`learner.ts`, read via `online(l)`). The step may reuse θ's buffers,
  so nothing keeps a reference to θ across a step. The hard target sync copies (`fmap(copy, θ)`),
  and the Polyak blend returns new buffers, so θ⁻ never aliases θ.
- **The host keeps** the Gymnasium env, the replay buffer and minibatch sampling. Each training step
  copies one minibatch to the device ($(2d + K + 3) B$ floats) and one scalar $\ell$ back.
- **Compiled programs are fixed** to their shapes and to the constants traced into them: $B$, $\kappa$,
  $\eta$, $\tau$. Changing $\tau$ compiles a new Polyak program; changing `batch`, `κ` or `η` on an
  existing learner does not take effect (build a new one). Editing `DDQNLoss` under Revise likewise
  needs a new learner.
- **Backend**: `DDQN_BACKEND=cpu` (default) runs on XLA's CPU client, `DDQN_BACKEND=gpu` on CUDA, read
  once when `ddqn.jl` loads. For $8 \to 64 \to 4$ and $B = 256$ the GPU buys nothing (each call is
  launch latency). On the GPU, XLA preallocates 75% of memory: set `XLA_REACTANT_GPU_PREALLOCATE=false`
  to share the card with another job.
- **Checkpoints** hold host copies only (θ, θ⁻, Adam's $(m, v, \beta_1^t, \beta_2^t)$ per array, η);
  `load_learner` restores them into a fresh TrainState. Files from the earlier Zygote version (CPU
  arrays, an `Optimisers` state tree) load the same way.

## Layout

| File | What |
|:---|:---|
| `src/setup.jl` | top level: includes `ddqn.jl`; `create_learner(e)`, `new_learner(e)`, `simple_train(e)`, `simple_infer(e)` |
| `src/envs.jl` | `EnvName` enum (`LUNAR_LANDER`, `CARTPOLE`), `SPECS` table, `get_env`, `get_model`, `outcome` |
| `src/ddqn.jl` | `Learner`, `action`, `on_step!`, `train_batch!`, `run_episode`, `train!`, `infer` |
| `src/replay.jl` | `Replay` ring buffer (`push!`, `sample`), included by `ddqn.jl` |
| `src/utils.jl` | helpers, included by `ddqn.jl`: `learning_curve` |
| `src/checkpoint.jl` | `save_learner`, `load_learner`, included by `ddqn.jl` |
| `Project.toml` | Julia deps: Lux, Reactant, Enzyme, Optimisers, Functors, PyCall, ProgressMeter, UnicodePlots, CairoMakie, Revise |
| `pyproject.toml`, `uv.lock` | Python side: `gymnasium[box2d]==1.3.0` in `.venv` |

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
code only sees the state size $d$ and action count $K$, $Q_\theta : \mathbb{R}^d \to \mathbb{R}^K$.

| `EnvName` | Gymnasium | d | K | reward | success | solved | overrides |
|:---|:---|:---|:---|:---|:---|:---|:---|
| `LUNAR_LANDER` | `LunarLander-v3` | 8 | 4 | shaped, ±100 terminal | `:landed` | 200 | — |
| `CARTPOLE` | `CartPole-v1` | 4 | 2 | +1 per step, cap 500 | `:balanced` (reached the cap) | 475 | $\eta = 10^{-3}$ |
| `ACROBOT` | `Acrobot-v1` | 6 | 3 | −1 per step, cap 500 | `:swung_up` (tip cleared the bar) | −100 | $\eta = 10^{-3}$ |
| `MOUNTAIN_CAR` | `MountainCar-v0` | 2 | 3 | −1 per step, cap 200 | `:summit` ($x \ge 0.5$) | −110 | — |

CartPole's success is a *truncation*: $d = 0$ there, so the target still bootstraps, which is right
(the pole was still up). Acrobot and MountainCar are the opposite: success is the *termination*
($d = 1$, no bootstrap past the goal), and the cap is failure.

- **Acrobot**: obs $= (\cos\theta_1, \sin\theta_1, \cos\theta_2, \sin\theta_2, \dot\theta_1, \dot\theta_2)$, torque $-1 / 0 / +1$ on the elbow.
  Underactuated, but −1 per step is dense enough for ε-greedy to find the goal.
- **MountainCar**: obs $= (x, v)$, push left / none / right. **Sparse**: a random policy almost never
  reaches the flag, so every episode returns −200 and $Q$ learns nothing useful until one succeeds.
  An exploration problem, not an approximation one: n-step returns push a success back $n$ states per
  update. $v \in [-0.07, 0.07]$ is ~15× smaller than $x \in [-1.2, 0.6]$ and nothing normalizes it.

  The plain `Learner` defaults solve it; "tuning" made it worse (1000 episodes, one seed each):

  | settings | first success | successes | last-100 avg | greedy, 100 episodes |
  |:---|:---|:---|:---|:---|
  | $\eta = 10^{-3}$, $n = 3$, $\varepsilon$ over 400 episodes | ~ep 350 | 568 / 1000 | −143.8 | 100/100, −129.1 |
  | defaults: $\eta = 10^{-2}$, $n = 10$, $\varepsilon$ over 200 episodes | ~ep 110 | 862 / 1000 | −115.5 | 100/100, **−101.1** |

  $n = 10$ vs 3 is the big one: in a sparse problem, how fast the terminal signal flows backward is
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
  The terminal reward *replaces* the step reward: +100 rest, −100 body contact or $|x| \ge 1$.
- **mean ℓ** is the mean batch loss $\ell$. It is *not* a progress measure (the target moves with $Q$);
  watch the returns.
- Training returns include ε noise and ongoing learning. The honest number is a **greedy, frozen
  evaluation**: $\varepsilon = 0$, no updates, fresh seeds.

## Results so far (1000 episodes, greedy eval of 50 episodes at snapshots 800 / 900 / 1000, 3 seeds)

| | train avg, last 100 | greedy snapshots (mean) | snapshots ≥ 200 |
|:---|:---|:---|:---|
| online (current transition only) | ≈ 146 | ≈ +83 | 2 / 9 |
| replay | ≈ 133 | ≈ +79 | 5 / 9 |
| replay + target network | ≈ 221 | ≈ +205 (worst +143) | 6 / 9 |
| + Huber $\kappa = 1$ | ≈ 138 | ≈ +108 | 3 / 9 |
| + Huber $\kappa = 30$ | ≈ 205 | ≈ +196 | 6 / 9 |
| + Double DQN ($\kappa = 30$) | ≈ 214 | ≈ +206 | 5 / 9 |
| + n-step, $n = 3$ | ≈ 214 | ≈ +219 (finals 234 / 253 / 276) | 7 / 9 |
| + n-step, $n = 5$ | ≈ 248 | ≈ +209 | 6 / 9 |

Double DQN lowered mean $\max_a Q$ over buffered states from ≈ 90 to ≈ 62 on every seed (less max-bias)
with returns unchanged: with 4 actions the bias rarely changes *which* action is greedy. $n = 3$ reached
a 100-episode average of 200 ≈ 50 episodes sooner (≈ 442 vs ≈ 493); $n = 5$ trains higher but its greedy
snapshots are noisier (more off-policy bias). Differences of 10–15 points are within 3-seed noise.

Both swing hard between snapshots (−68 … +229 within one run): replay finds solved-level policies
more often but loses them again — the **moving target** ($t$ uses the same $\theta$ being trained). The
target network fixes most of that: snapshots swing far less (worst +143 vs −68).
Without annealing (fixed $\varepsilon = 0.15$) online learning plateaued at ≈ +50–70; with $\eta = 10^{-3}$ instead of
$10^{-2}$, nothing much was learned in 1000 episodes.

## Next (classic "max out Q with a NN", in order)

1. Dueling heads, prioritized replay, distributional (C51), noisy nets (the rest of Rainbow).

Plus: bigger net (2×128 ReLU), anneal $\varepsilon$ by steps, keep the best greedy snapshot; compare Polyak
$\tau = 0.002$ against the hard copy $C = 1000$.

## Notes / gotchas

- **Python must be the system `/usr/bin/python3.12`**, not uv's managed CPython: its `libpython`
  needs an executable stack, which glibc ≥ 2.41 refuses to `dlopen` (`pyproject.toml` pins
  `python-preference = "only-system"`).
- **PyCall's Python is global to the Julia depot** (`Pkg.build("PyCall")` with `PYTHON=…`). After
  renaming/moving this folder, rebuild it.
- **Threads**: `~/.zshrc` sets `JULIA_NUM_THREADS=auto` (48). Nothing here uses threads, and it
  exposed a Julia 1.12.3 JIT race with the REPL's background hint task (segfault). Run with
  `JULIA_NUM_THREADS=1` or disable hints: `Base.active_repl.options.hint_tab_completes = false`.
- **First call compiles**: Julia's JIT plus one XLA compile per program (a few seconds each, at the
  first action and the first training step); stay in one REPL session. Revise recompiles Julia code
  you edit, but a learner keeps the XLA programs it already built (see *Reactant + Enzyme*).
