import torch
import matplotlib.pyplot as plt
import numpy as np
import time

import envs.waypoint_2d as wp_2d
import envs.waypoint_3d as wp_3d
import envs.nh_waypoint_3d as nh_wp_3d

import envs.traj_2d as traj_2d
import envs.soft_2d as soft_2d
import envs.term_2d as term_2d

import envs.soft_3d as soft_3d
import envs.term_3d as term_3d

import gym
import gym_aero

import config as cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_waypoint_2d():
    def _thunk():
        env = wp_2d.WaypointEnv2D()
        return env
    return _thunk

def make_waypoint_3d():
    def _thunk():
        env = wp_3d.WaypointEnv3D()
        return env
    return _thunk

def make_nh_waypoint_3d():
    def _thunk():
        env = nh_wp_3d.WaypointEnv3D()
        return env
    return _thunk

def make_traj_2d():
    def _thunk():
        env = traj_2d.TrajectoryEnv2D()
        return env
    return _thunk

def make_soft_2d():
    def _thunk():
        env = soft_2d.SoftTrajectoryEnv2D()
        return env
    return _thunk

def make_term_2d():
    def _thunk():
        env = term_2d.TrajectoryEnvTerm2D()
        return env
    return _thunk

def make_traj_3d():
    def _thunk():
        env = gym.make("Trajectory-v0")
        return env
    return _thunk

def make_soft_3d():
    def _thunk():
        env = soft_3d.TrajectoryEnv3D()
        return env
    return _thunk

def make_term_3d():
    def _thunk():
        env = term_3d.TrajectoryEnvTerm()
        return env
    return _thunk

def plot(episodes, rewards, fname=None):
    plt.figure(figsize=(10,10))
    plt.subplot(111)
    plt.title("Iterations: %s. Cumulative Reward: %s" % (episodes[-1], rewards[-1]))
    plt.plot(episodes, rewards)
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Reward")
    if fname is None: plt.show()
    else: plt.savefig(fname+".pdf")

def test(env, agent, render=False):
    state = torch.Tensor(env.reset()).to(device)
    reward_sum = 0
    done = False
    if render:
            env.render()
            time.sleep(0.05)
    while not done:
        action = agent.select_action(state.unsqueeze(0), deterministic=True)
        #print("action: ", action)
        action = action.squeeze(0)
        next_state, reward, done, info = env.step(action.cpu().data.numpy())
        #print(info)
        if render:
            env.render()
            time.sleep(0.05)
        reward_sum += reward
        next_state = torch.Tensor(next_state).to(device)
        state = next_state
    #if render: env.close()
    return reward_sum

def train_mp(envs, t_env, agent, opt, batch_size, iterations, log_interval, t_runs=30, render=False, fname=None):
    rews = []
    eps = []
    test_rew_best = np.mean([test(t_env, agent, render=render) for _ in range(t_runs)])
    eps.append(0)
    rews.append(test_rew_best)
    print()
    print("Iterations: ", 0)
    print("Time steps: ", 0)
    print("Reward: ", test_rew_best)
    print()
    state = torch.Tensor(envs.reset()).to(device)
    for ep in range(1, iterations+1):
        s_, a_, ns_, r_, lp_, masks = [], [], [], [], [], []
        goal_pos, next_goal_pos = [], []
        t = 0
        while t < batch_size:
            actions, log_probs, entropies = agent.select_action(state)
            g_pos = torch.Tensor(envs.get_goal_positions()).to(device)
            next_state, reward, done, info = envs.step(actions.cpu().data.numpy())
            dones = [[not d] for d in done]
            reward = torch.Tensor(reward).unsqueeze(1).to(device)
            next_state = torch.Tensor(next_state).to(device)
            next_g_pos = torch.Tensor(envs.get_goal_positions()).to(device)

            reward += entropies.sum(dim=-1, keepdim=True)

            s_.append(state)
            goal_pos.append(g_pos)
            next_goal_pos.append(next_g_pos)
            a_.append(actions)
            ns_.append(next_state)
            r_.append(reward)
            lp_.append(log_probs)
            masks.append(torch.Tensor(dones).to(device))
            
            state = next_state
            t += 1
        trajectory = {
                    "states" : s_,
                    "goal_position": goal_pos,
                    "next_goal_position": next_goal_pos,
                    "actions" : a_,
                    "rewards" : r_,
                    "next_states" : ns_,
                    "masks" : masks,
                    "log_probs" : lp_,
                    }
        agent.update(opt, trajectory)
        if ep % log_interval == 0:
            eps.append(len(envs)*batch_size*ep)
            test_rew = np.mean([test(t_env, agent, render=render) for _ in range(t_runs)])
            rews.append(test_rew)
            #plot(eps, rews)
            print("Iterations: ", ep)
            print("Time steps: ", len(envs)*batch_size*ep)
            print("Reward: ", test_rew)
            if (test_rew > test_rew_best) and (fname is not None): 
                print("Saving best parameters in"+fname+".pth.tar")
                torch.save(agent.state_dict(), fname+".pth.tar")
                test_rew_best = test_rew
            print()
    if fname is not None: torch.save(agent.state_dict(), fname+"final.pth.tar")
    #plot(eps, rews, fname=fname)
    return eps, rews, agent

def test_lstm(env, agent, render=False):
    state = torch.Tensor(env.reset()).to(device)
    reward_sum = 0
    hidden_b = None
    hidden_c = None
    done = False
    while not done:
        if render:
            env.render()
            time.sleep(0.05)
        action, _, hidden_b, hidden_c, _, _ = agent.select_action(state.view(1,-1), hidden_b, hidden_c)
        action = action.squeeze(0)
        next_state, reward, done, info = env.step(action.cpu().data.numpy())
        reward_sum += reward
        next_state = torch.Tensor(next_state).to(device)
        state = next_state
    #if render: env.close()
    return reward_sum

def train_mp_lstm(envs, t_env, agent, opt, batch_size, iterations, log_interval, t_runs=10, render=False, fname=None):
    rews = []
    eps = []
    test_rew_best = np.mean([test_lstm(t_env, agent, render=render) for _ in range(t_runs)])
    eps.append(0)
    rews.append(test_rew_best)
    print()
    print("Iterations: ", 0)
    print("Time steps: ", 0)
    print("Reward: ", test_rew_best)
    print()
    state = torch.Tensor(envs.reset()).to(device)
    for ep in range(1, iterations+1):
        s_, a_, ns_, r_, v_, lp_, m_ = [], [], [], [], [], [], []
        h_b, h_c = [], []
        goal_pos, next_goal_pos = [], []
        t = 0
        while t < batch_size:
            actions, values, hidden_b, hidden_c, log_probs, entropies = agent.select_action(state)
            g_pos = torch.Tensor(envs.get_goal_positions()).to(device)
            next_state, reward, done, info = envs.step(actions.cpu().data.numpy())
            for j, d in enumerate(done):
                if d == 1:
                    hidden_b[0][:, j, :] *= (1-d)
                    hidden_b[1][:, j, :] *= (1-d)
                    hidden_c[0][:, j, :] *= (1-d)
                    hidden_c[1][:, j, :] *= (1-d)
            masks = torch.Tensor([[not d] for d in done]).to(device)
            reward = torch.Tensor(reward).unsqueeze(1).to(device)
            next_state = torch.Tensor(next_state).to(device)
            next_g_pos = torch.Tensor(envs.get_goal_positions()).to(device)

            reward += entropies.sum(dim=-1, keepdim=True)

            s_.append(state)
            goal_pos.append(g_pos)
            next_goal_pos.append(next_g_pos)
            a_.append(actions)
            ns_.append(next_state)
            r_.append(reward)
            v_.append(values)
            lp_.append(log_probs)
            m_.append(masks)
            h_b.append(hidden_b)
            h_c.append(hidden_c)
            
            state = next_state
            t += 1
        trajectory = {
                    "states" : s_,
                    "goal_position": goal_pos,
                    "next_goal_position": next_goal_pos,
                    "actions" : a_,
                    "rewards" : r_,
                    "next_states" : ns_,
                    "values" : v_,
                    "masks" : m_,
                    "log_probs" : lp_,
                    "hiddens_b" : h_b,
                    "hiddens_c" : h_c
                    }
        agent.update(opt, trajectory)
        if ep % log_interval == 0:
            eps.append(len(envs)*batch_size*ep)
            test_rew = np.mean([test_lstm(t_env, agent, render=render) for _ in range(t_runs)])
            rews.append(test_rew)
            #plot(eps, rews)
            print("Iterations: ", ep)
            print("Time steps: ", len(envs)*batch_size*ep)
            print("Reward: ", test_rew)
            if (test_rew > test_rew_best) and (fname is not None): 
                print("Saving best parameters in"+fname+".pth.tar")
                torch.save(agent.state_dict(), fname+".pth.tar")
                test_rew_best = test_rew
            print()
    if fname is not None: torch.save(agent.state_dict(), fname+"final.pth.tar")
    #plot(eps, rews, fname=fname)
    return eps, rews, agent

def test_term(env, agent, render=True):
    state = torch.Tensor(env.reset()).to(device)
    reward_sum = 0
    term_reward_sum = 0
    done = False
    while not done:
        if render:
            env.render()
            time.sleep(0.05)
        action, _, _, _ = agent.select_action(state.unsqueeze(0))
        action = action.squeeze(0)
        next_state, reward, done, info = env.step(action.cpu().data.numpy())
        term_rew = info["term_rew"]
        reward_sum += reward
        term_reward_sum += term_rew
        next_state = torch.Tensor(next_state).to(device)
        state = next_state
    return reward_sum, term_reward_sum

def train_term_mp(envs, t_env, agent, opt, batch_size, iterations, log_interval, t_runs=10, render=True, fname=None):
    rews = []
    term_rews = []
    eps = []
    test_rews = [test_term(t_env, agent, render=render) for _ in range(t_runs)]
    test_rew_best = np.mean([tr[0] for tr in test_rews])
    test_rew_term_best = np.mean([tr[1] for tr in test_rews])
    eps.append(0)
    rews.append(test_rew_best)
    term_rews.append(test_rew_term_best)
    print()
    print("Iterations: ", 0)
    print("Time steps: ", 0)
    print("Reward: ", test_rew_best)
    print("Term Reward: ", test_rew_term_best)
    print()
    state = torch.Tensor(envs.reset()).to(device)
    for ep in range(1, iterations+1):
        s_, a_, ns_, r_, v_, lp_,  masks = [], [], [], [], [], [], []
        goal_pos, next_goal_pos = [], []
        t_s_, t_a_, t_r_, t_v_, t_lp_ = [], [], [], [], []
        gs = []
        t = 0
        while t < batch_size:
            actions, values, log_probs, entropies = agent.select_action(state)
            g_pos = torch.Tensor(envs.get_goal_positions()).to(device)
            next_state, reward, done, info = envs.step(actions.cpu().data.numpy())
            
            term_rew = [i["term_rew"] for i in info]
            goal_switch = [i["goal_switch"] for i in info]
            dones = [not d for d in done]

            term_rew = torch.Tensor(term_rew).unsqueeze(1).to(device)
            goal_switch = torch.Tensor(goal_switch).unsqueeze(1).to(device)
            mask = torch.Tensor(dones).unsqueeze(1).to(device)
            reward = torch.Tensor(reward).unsqueeze(1).to(device)
            next_state = torch.Tensor(next_state).to(device)
            next_g_pos = torch.Tensor(envs.get_goal_positions()).to(device)

            reward += 1e-2 * entropies[0].sum(dim=-1, keepdim=True)
            term_rew += 1e-2 * entropies[1].unsqueeze(1)

            s_.append(state)
            goal_pos.append(g_pos)
            next_goal_pos.append(next_g_pos)
            a_.append(actions[:,:-1])
            ns_.append(next_state)
            r_.append(reward)
            v_.append(values[0])
            lp_.append(log_probs[0])
            masks.append(mask)
            
            t_s_.append(state)
            t_lp_.append(log_probs[1].unsqueeze(1))
            t_r_.append(term_rew)
            t_v_.append(values[1])
            t_a_.append(actions[:,-1])

            gs.append(goal_switch)
            
            state = next_state
            g_pos = next_g_pos
            t += 1
        trajectory = {
                    "states" : s_,
                    "goal_position": goal_pos,
                    "next_goal_position": next_goal_pos,
                    "actions" : a_,
                    "rewards" : r_,
                    "next_states" : ns_,
                    "values" : v_,
                    "masks" : masks,
                    "log_probs" : lp_,
                    "term_states" : t_s_,
                    "term_log_probs" : t_lp_,
                    "term_rew" : t_r_,
                    "term_val" : t_v_,
                    "terminations" : t_a_,
                    "goal_switch" : gs
                    }
        agent.update(opt, trajectory)
        if ep % log_interval == 0:
            eps.append(len(envs)*batch_size*ep)
            test_rew = [test_term(t_env, agent, render=render) for _ in range(t_runs)]
            mean_test_rew = np.mean([tr[0] for tr in test_rew])
            mean_term_rew = np.mean([tr[1] for tr in test_rew])
            rews.append(mean_test_rew)
            term_rews.append(mean_term_rew) 
            print("Iterations: ", ep)
            print("Time steps: ", len(envs)*batch_size*ep)
            print("Reward: ", mean_test_rew)
            print("Term Reward: ", mean_term_rew)
            if (mean_test_rew > test_rew_best) and (fname is not None): 
                print("Saving best parameters in"+fname+"term.pth.tar")
                torch.save(agent.state_dict(), fname+"term.pth.tar")
                test_rew_best = mean_test_rew
            print()
    if fname is not None: torch.save(agent.state_dict(), fname+"term_final.pth.tar")
    #plot(eps, rews, fname=fname)
    #plot(eps, term_rews, fname="term_"+fname)
    return eps, rews, term_rews, agent