# Decision Making

Sequential decision making: an agent chooses actions $a_t$ in states $s_t$ to maximize the expected
return

$$
G = \sum_{t=0}^{\infty} \gamma^t r_t, \qquad 0 \le \gamma \le 1,
$$

usually framed as a Markov decision process $(\mathcal{S}, \mathcal{A}, P(s' \mid s, a), r, \gamma)$.
The optimal action-value function satisfies the Bellman optimality equation

$$
Q^*(s, a) = \mathbb{E}\left[\, r + \gamma \max_{a'} Q^*(s', a') \;\middle|\; s, a \,\right].
$$

Two branches, split by whether the model $P$ is used:

- **`rl/`**: reinforcement learning, model-free. Learn $Q$ (or a policy) from sampled transitions only.
- planning (model-based, e.g. MCTS): search with a model of $P$. Not here yet.

## Projects

| Project | What |
|:---|:---|
| [`rl/ddqn`](rl/ddqn) | Double DQN in Julia (Lux + Zygote): replay, target network, Huber loss, n-step returns. Solves LunarLander, CartPole, Acrobot and MountainCar from Gymnasium via PyCall. |
