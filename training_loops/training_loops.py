import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

import envs.waypoint_2d as wp_2d
import envs.waypoint_3d as wp_3d

import envs.fan_2d as fan_2d
import envs.traj_2d as traj_2d
import envs.soft_2d as soft_2d
import envs.term_2d as term_2d

import envs.traj_3d as traj_3d
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

def make_fan_2d():
    def _thunk():
        env = fan_2d.FanTrajectoryEnv2D()
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
        env = traj_3d.TrajectoryEnv3D()#gym.make("RandomWaypointNH-v0")
        return env
    return _thunk

def make_soft_3d():
    def _thunk():
        env = soft_3d.SoftTrajectoryEnv3D()
        return env
    return _thunk

def make_term_3d():
    def _thunk():
        env = term_3d.TrajectoryEnvTerm3D()
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
    sigmas = []
    while not done:
        #print("state: ", state)
        mu, sigma = agent.test_action(state.unsqueeze(0))
        action, sigma = mu.squeeze(0).cpu().data.numpy(), sigma.squeeze(0).cpu().data.numpy()
        #print("action", mu)
        next_state, reward, done, info = env.step(action)
        sigmas.append(sigma)
        #print(info)
        #input()
        if render:
            env.render()
            time.sleep(0.05)
        reward_sum += reward
        next_state = torch.Tensor(next_state).to(device)
        state = next_state
    #if render: env.close()
    mean_sigma = np.mean(sigmas)
    #print(reward_sum)
    return reward_sum, mean_sigma

def train_mp(logger, envs, t_env, agent, opt, batch_size, iterations, log_interval, t_runs, render=False, fname=None):
    rews = []
    eps = []
    test_res = [test(t_env, agent, render=render) for _ in range(t_runs)]
    test_rew_best = np.mean([t[0] for t in test_res])
    sigma = np.mean([t[1] for t in test_res])
    logger.info(" ")
    logger.info("Iterations: {}".format(0))
    logger.info("Time steps: {}".format(0))
    logger.info("Reward: {:.5f}".format(test_rew_best))
    logger.info("Test sigma: {:.5f}".format(sigma))
    logger.info(" ")
    eps.append(0)
    rews.append(test_rew_best)
    print()
    print("Iterations: ", 0)
    print("Time steps: ", 0)
    print("Reward: ", test_rew_best)
    print("Test sigma: ", sigma)
    print()
    state = torch.Tensor(envs.reset()).to(device)
    for ep in range(1, iterations+1):
        s_, a_, ns_, r_, lp_, masks = [], [], [], [], [], []
        t = 0
        while t < batch_size:
            actions, log_probs, entropies = agent.select_action(state)
            next_state, reward, done, info = envs.step(actions.cpu().data.numpy())
            dones = [[not d] for d in done]
            reward = torch.Tensor(reward).unsqueeze(1).to(device)
            next_state = torch.Tensor(next_state).to(device)
            reward += 0.5 * entropies.sum(dim=-1, keepdim=True)
 
            s_.append(state)
            a_.append(actions)
            ns_.append(next_state)
            r_.append(reward)
            lp_.append(log_probs)
            masks.append(torch.Tensor(dones).to(device))
            
            state = next_state
            t += 1
        trajectory = {
                    "states" : s_,
                    "actions" : a_,
                    "rewards" : r_,
                    "next_states" : ns_,
                    "masks" : masks,
                    "log_probs" : lp_,
                    }
        agent.update(opt, trajectory)
        if ep % log_interval == 0:
            eps.append(len(envs)*batch_size*ep)
            test_res = [test(t_env, agent, render=render) for _ in range(t_runs)]
            test_rew = np.mean([t[0] for t in test_res])
            sigma = np.mean([t[1] for t in test_res])
            logger.info("Iterations: {}".format(ep))
            logger.info("Time steps: {}".format(len(envs)*batch_size*ep))
            logger.info("Reward: {:.5f}".format(test_rew))
            logger.info("Test sigma: {:.5f}".format(sigma))
            logger.info(" ")
            rews.append(test_rew)
            #plot(eps, rews)
            print("Iterations: ", ep)
            print("Time steps: ", len(envs)*batch_size*ep)
            print("Reward: ", test_rew)
            print("Test sigma: ", sigma)
            if (test_rew > test_rew_best) and (fname is not None):
                logger.info("Saving best parameters in " + fname + ".pth.tar")
                print("Saving best parameters in " + fname + ".pth.tar")
                torch.save(agent.state_dict(), fname+".pth.tar")
                test_rew_best = test_rew
            print()
    final_name = fname + "_" + str(datetime.datetime.now()) + "_final.pth.tar"
    logger.info("Saving final model as " + final_name)
    if fname is not None: torch.save(agent.state_dict(), final_name)
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

def train_term_mp(logger, envs, t_env, agent, opt, batch_size, iterations, log_interval, t_runs=10, render=True, fname=None):
    rews = []
    term_rews = []
    eps = []
    test_rews = [test_term(t_env, agent, render=render) for _ in range(t_runs)]
    test_rew_best = np.mean([tr[0] for tr in test_rews])
    test_rew_term_best = np.mean([tr[1] for tr in test_rews])
    logger.info(" ")
    logger.info("Iterations: {}".format(0))
    logger.info("Time steps: {}".format(0))
    logger.info("Reward: {:.5f}".format(test_rew_best))
    logger.info(" ")
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
        t_s_, t_a_, t_r_, t_v_, t_lp_ = [], [], [], [], []
        t = 0
        while t < batch_size:
            actions, values, log_probs, entropies = agent.select_action(state)
            next_state, reward, done, info = envs.step(actions.cpu().data.numpy())
            
            term_rew = [i["term_rew"] for i in info]
            dones = [not d for d in done]

            term_rew = torch.Tensor(term_rew).unsqueeze(1).to(device)
            mask = torch.Tensor(dones).unsqueeze(1).to(device)
            reward = torch.Tensor(reward).unsqueeze(1).to(device)
            next_state = torch.Tensor(next_state).to(device)

            reward += 1e-2 * entropies[0].sum(dim=-1, keepdim=True)
            term_rew += 1e-2 * entropies[1].unsqueeze(1)

            s_.append(state)
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
            
            state = next_state
            t += 1
        trajectory = {
                    "states" : s_,
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
                    }
        agent.update(opt, trajectory)
        if ep % log_interval == 0:
            eps.append(len(envs)*batch_size*ep)
            test_rew = [test_term(t_env, agent, render=render) for _ in range(t_runs)]
            mean_test_rew = np.mean([tr[0] for tr in test_rew])
            mean_term_rew = np.mean([tr[1] for tr in test_rew])
            logger.info("Iterations: {}".format(ep))
            logger.info("Time steps: {}".format(len(envs)*batch_size*ep))
            logger.info("Reward: {:.5f}".format(mean_test_rew))
            logger.info("Term Reward: {:.5f}".format(mean_term_rew))
            logger.info(" ")
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


def train_online(env, agent, pol_opt, q_opt, v_opt, steps=1000, warmup=1000, batch_size=128, iterations=500, log_interval=10, t_runs=10):
    # warmup to add transitions to replay memory
    T = 0
    while T < warmup:
        state = torch.Tensor(env.reset()).to(device)
        done = False
        t = 0
        while not done:
            action, _, _ = agent.select_action(state)
            action = action.detach()
            next_state, reward, done, info = env.step(action.cpu().data.numpy())
            mask = 1 if t == env._max_episode_steps else float(not done)
            mask = torch.Tensor([mask]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor(next_state).to(device)
            agent.replay_memory.push(state, action, next_state, reward, mask)
            state = next_state
            t += 1
        T += t
    
    print("Warmup finished, training agent.")
    
    test_rew_best = np.mean([test(env, agent, None) for _ in range(t_runs)])
    data = []
    data.append(test_rew_best)
    print()
    print("Iterations: ", 0)
    print("Time steps: ", 0)
    print("Reward: ", test_rew_best)
    print()
    
    # run training loop
    for ep in range(1, int(iterations + 1)):
        T = 0
        R = 0
        n = 1
        while T < steps:
            done = False
            state = torch.Tensor(env.reset()).to(device)
            r = 0
            t = 0
            while not done:
                action, _, _ = agent.select_action(state)
                action = action.detach()
                next_state, reward, done, info = env.step(action.cpu().data.numpy())
                r += reward
                mask = 1 if t == env._max_episode_steps else float(not done)
                mask = torch.Tensor([mask]).to(device)
                reward = torch.Tensor([reward]).to(device)
                next_state = torch.Tensor(next_state).to(device)
                agent.replay_memory.push(state, action, next_state, reward, mask)
                agent.update(pol_opt, q_opt, v_opt, batch_size)
                state = next_state
                t += 1
            T += t
            R = (R*(n-1)+r)/n
            n += 1
        print("Batch {},  mean reward: {:.4f}".format(ep, R))        
        if ep % log_interval == 0:
            test_rew = np.mean([test(env, agent, render=True) for _ in range(t_runs)])
            print("------------------------------")
            print("Iterations: ", ep)
            print("Time steps: ", ep * steps)
            print("Test Reward: ", test_rew)
            print("------------------------------")
            data.append(test_rew)
    return data