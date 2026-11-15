# Which Gymnasium problem a learner solves. Everything env-specific lives in SPECS; the DQN code
# reads only the state size d and action count K from here (Q_θ : ℝᵈ → ℝᴷ).
#
#   LUNAR_LANDER  LunarLander-v3  d = 8, K = 4   r ∈ ~[−100, 100] per step, solved at 200
#   CARTPOLE      CartPole-v1     d = 4, K = 2   r = +1 per step, capped at 500, solved at 475
#   ACROBOT       Acrobot-v1      d = 6, K = 3   r = −1 per step until the tip clears the bar, capped at 500, solved at −100
#   MOUNTAIN_CAR  MountainCar-v0  d = 2, K = 3   r = −1 per step until x ≥ 0.5, capped at 200, solved at −110
#
# Acrobot obs: cos θ₁, sin θ₁, cos θ₂, sin θ₂, θ̇₁, θ̇₂; actions: torque −1 / 0 / +1 on the elbow.
# MountainCar obs: x ∈ [−1.2, 0.6], v ∈ [−0.07, 0.07]; actions: push left / none / right.
#   Sparse: a random policy almost never reaches the flag, so every episode is −200 until one does.
#   This is an exploration problem, not an approximation one. The default n = 10 carries the first
#   success back 10 states per update; that, not a slower ε decay, is what gets it going.
#   Note v is ~15× smaller than x and nothing normalizes it.
#
# New members go at the END: checkpoints store envname, and appending keeps existing values.

@enum EnvName LUNAR_LANDER CARTPOLE ACROBOT MOUNTAIN_CAR

#   id       Gymnasium id
#   obs, actions   d, K
#   solved   "solved" level of the 100-episode average (dashed line on the plot)
#   ylim     plot's minimum y-range, widened to the data
#   hyper    Learner keywords that override the defaults for this env
const SPECS = Dict(
    LUNAR_LANDER => (; id="LunarLander-v3", obs=8, actions=4, solved=200, ylim=(-400, 320),
                       hyper=(;)),
    CARTPOLE     => (; id="CartPole-v1", obs=4, actions=2, solved=475, ylim=(0, 520),
                       hyper=(; η=1f-3)),
    ACROBOT      => (; id="Acrobot-v1", obs=6, actions=3, solved=-100, ylim=(-520, 0),
                       hyper=(; η=1f-3)),
    MOUNTAIN_CAR => (; id="MountainCar-v0", obs=2, actions=3, solved=-110, ylim=(-210, 0),
                       hyper=(;)),   # defaults solve it (greedy −101); η=1f-3, n=3, ε_episodes=400 only got −129
)

spec(e::EnvName) = SPECS[e]

# Files are named after the env: LUNAR_LANDER → runs/lunar_lander.jls, resources/lunar_lander.png
envfile(e::EnvName) = lowercase(string(e))
learner_path(e::EnvName) = "runs/$(envfile(e)).jls"
plot_path(e::EnvName) = "resources/$(envfile(e)).png"

function get_env(e::EnvName; render=false)
    gym = pyimport("gymnasium")
    env = gym.make(spec(e).id; render_mode=(render ? "human" : nothing))
    # action_space.n is a numpy int64, which PyCall leaves as a PyObject; .item() → Python int → Int
    dims = (Int(env.observation_space.shape[1]), Int(env.action_space.n.item()))
    dims == (spec(e).obs, spec(e).actions) ||
        error("$(spec(e).id): gym says (d, K) = $dims, SPECS says $((spec(e).obs, spec(e).actions))")
    return env
end

# Q_θ : ℝᵈ → ℝᴷ, one hidden layer of 64 tanh units for both.
get_model(e::EnvName) = Lux.Chain(Lux.Dense(spec(e).obs => 64, tanh), Lux.Dense(64 => spec(e).actions))

# How an episode ended. LunarLander: the terminal reward replaces the step reward (+100 came to
# rest, −100 body hit / out of bounds). CartPole: terminated = pole fell or cart left the track;
# truncated = survived the 500-step cap, which is the goal. Acrobot / MountainCar: terminated =
# reached the goal (the only way out early); truncated = ran out the clock.
function outcome(e::EnvName, r, terminated, truncated)
    e == CARTPOLE && return truncated ? :balanced : :fell
    e == ACROBOT && return terminated ? :swung_up : :timeout
    e == MOUNTAIN_CAR && return terminated ? :summit : :timeout
    return truncated ? :timeout : (r == 100 ? :landed : :crashed)
end

success(o::Symbol) = o ∈ (:landed, :balanced, :swung_up, :summit)
