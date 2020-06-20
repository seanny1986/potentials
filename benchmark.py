import gym
from stable_baselines3 import PPO
import training_loops.training_loops as tl
#from common.multiprocessing_env import SubprocVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import envs.traj_2d as tenv
from stable_baselines3.common.callbacks import EvalCallback


if __name__ == '__main__':
    envs = [tl.make_traj_2d() for i in range(16)]
    envs = SubprocVecEnv(envs)

    # Separate evaluation env
    t_env = tenv.TrajectoryEnv2D()

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(t_env, best_model_save_path='./logs/',
                                log_path='./logs/', eval_freq=1000,
                                deterministic=True, render=True)
    model = PPO('MlpPolicy', envs, verbose=1)
    model.learn(total_timesteps=2500000, callback=eval_callback)