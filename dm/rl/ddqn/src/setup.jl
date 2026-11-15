include("ddqn.jl")              # also includes envs.jl: EnvName (LUNAR_LANDER, CARTPOLE, ACROBOT, MOUNTAIN_CAR), get_env, get_model

# A fresh Learner for env e: get_model(e), the env's hyperparameter overrides (SPECS.hyper), then kw.
new_learner(e::EnvName; kw...) = Learner(get_model(e); envname=e, spec(e).hyper..., kw...)

# (env, learner) for e, e.g.  env, l = create_learner(CARTPOLE); train!(l, env)
create_learner(e::EnvName; render=false, kw...) = (get_env(e; render), new_learner(e; kw...))

# Train from scratch, plot (terminal + resources/<env>.png), save the learner to runs/<env>.jls.
function simple_train(e::EnvName=LUNAR_LANDER; episodes=1000, plots=BothPlots, path=learner_path(e), kw...)
	env, learner = create_learner(e; kw...)
	try
		train!(learner, env; episodes, plots)
	finally
		env.close()
		save_learner(path, learner)          # saved even if interrupted (Ctrl-C) mid-training
		println("learner → ", path)
	end
	return learner
end

# Load a saved learner and just fly it: greedy (ε = 0), no learning, rendered in a window.
# The env comes from the checkpoint (learner.envname).
function simple_infer(e::EnvName=LUNAR_LANDER; path=learner_path(e), episodes=5)
	learner = load_learner(path)
	env = get_env(learner.envname; render=true)
	try
		infer(learner, env; episodes)
	finally
		env.close()
	end
	return learner
end
