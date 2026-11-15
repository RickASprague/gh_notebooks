# Decision Making

Sequential decision making: an agent chooses actions aₜ in states sₜ to maximize expected return

  G = Σₜ γᵗ rₜ,

usually framed as a Markov decision process (𝒮, 𝒜, P(s′ | s, a), r, γ). The optimal action-value
function satisfies the Bellman optimality equation

  Q*(s, a) = 𝔼[ r + γ maxₐ′ Q*(s′, a′) | s, a ].

Two branches, split by whether the model P is used:

- **`rl/`**: reinforcement learning, model-free. Learn Q (or a policy) from sampled transitions only.
- planning (model-based, e.g. MCTS): search with a model of P. Not here yet.

## Projects

| Project | What |
|:---|:---|
| [`rl/ddqn`](rl/ddqn) | Double DQN in Julia (Lux + Zygote): replay, target network, Huber loss, n-step returns. Solves LunarLander, CartPole, Acrobot and MountainCar from Gymnasium via PyCall. |
