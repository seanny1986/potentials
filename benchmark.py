import gym
#from stable_baselines3 import TD3
import training_loops.training_loops as tl
#from common.multiprocessing_env import SubprocVecEnv
#from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.fan_2d as tenv
import agents.modules as mod
import agents.online_agents as onag
import agents.utilities as utils
import torch
#from stable_baselines3.common.callbacks import EvalCallback


if __name__ == '__main__':
    #envs = [tl.make_fan_2d() for i in range(16)]
    #envs = SubprocVecEnv(envs)

    # Separate evaluation env
    t_env = tenv.FanTrajectoryEnv2D()
    state_dim = t_env.observation_space.shape[0]
    action_dim = t_env.action_space.shape[0]

    # Use deterministic actions for evaluation
    #eval_callback = EvalCallback(t_env, best_model_save_path='./logs/',
    #                            log_path='./logs/', eval_freq=1000,
    #                            deterministic=True, render=True)
    #model = TD3('MlpPolicy', t_env, verbose=1)
    #model.learn(total_timesteps=2500000, callback=eval_callback)

    replay_memory = utils.ReplayMemory(1e6)
    q_fn = mod.ValueNet(state_dim + action_dim, 256, 1, num_heads=1)
    q_fn_targ = mod.ValueNet(state_dim + action_dim, 256, 1, num_heads=1)
    beta = mod.DeterministicPolicy(state_dim, 256, action_dim)
    agent = onag.TD3(beta, q_fn, q_fn_targ, replay_memory)
    pol_opt = torch.optim.Adam(beta.parameters(), lr=1e-4)
    q_opt = torch.optim.Adam(q_fn.parameters(), lr=1e-4)
    sac_data = tl.train_online(t_env, agent, pol_opt, q_opt, None, batch_size=128, iterations=2500, log_interval=10)
