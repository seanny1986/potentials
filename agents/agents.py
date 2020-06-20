import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch import autograd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0., std=0.1)
        torch.nn.init.constant_(m.bias, 0.1)

class SimplePolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimplePolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.mu = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, output_dim)).to(device)

        self.logvar = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, output_dim)).to(device)
        
    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar.exp().sqrt()


class ValueFunctionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ValueFunctionLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.__lstm = nn.LSTM(input_dim, hidden_dim, 2)
        self.__value = nn.Linear(hidden_dim, 1)
    
    def step(self, x, hidden=None):
        hx, cx = self.__lstm(x.unsqueeze(0), hidden)
        val = self.__value(hx.squeeze(0))
        return val, cx

    def forward(self, x, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(x)
        vals = torch.zeros(steps, 1, 1)
        for i in range(steps):
            if force or i == 0:
                input = x[i]
            value, hidden = self.step(input, hidden)
            vals[i] = value
        return vals, hidden


class GaussianPolicyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GaussianPolicyLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.__lstm = nn.LSTM(input_dim, hidden_dim, 2)
        self.__mu = nn.Linear(hidden_dim, output_dim)
        self.__logvar = nn.Linear(hidden_dim, output_dim)
    
    def step(self, x, hidden=None):
        hx, cx = self.__lstm(x.unsqueeze(0), hidden)
        mu = self.__mu(hx.squeeze(0))
        logvar = self.__logvar(hx.squeeze(0))
        return mu, logvar.exp().sqrt(), cx

    def forward(self, x, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(x)
        mus = torch.zeros(steps, 1, self.output_dim)
        sigmas = torch.zeros(steps, 1, self.output_dim)
        for i in range(steps):
            if force or i == 0:
                input = x[i]
            mu, sigma, hidden = self.step(input, hidden)
            mus[i] = mu
            sigmas[i] = sigma
        return mus, sigmas, hidden


class SimpleA2C(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dim=2, lookahead=1):
        super(SimpleA2C, self).__init__()
        self.beta = SimplePolicy(input_dim, hidden_dim, output_dim).to(device)
        self.critic = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, 1)).to(device)

    def select_action(self, x, deterministic=False):
        if deterministic == True:
            mu, sigma = self.beta(x)
            #print("Action: ", mu)
            return mu
        else:
            mu, sigma = self.beta(x)
            dist = Normal(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
            entropy = torch.sum(entropy, dim=-1, keepdim=True)
            #print("Action: ", action)
            return action, log_prob, entropy

    def get_phi(self, trajectory, critic, gamma=0.99):
        states = torch.stack(trajectory["states"]).to(device)
        rewards = torch.stack(trajectory["rewards"]).to(device)
        next_states = torch.stack(trajectory["next_states"]).to(device)
        masks = torch.stack(trajectory["masks"]).to(device)
        returns = torch.Tensor(rewards.size()).to(device)
        deltas = torch.Tensor(rewards.size()).to(device)
        prev_return = 0
        for j in reversed(range(rewards.size(0))):
            next_val = critic(next_states[j])
            bootstrap = next_val * (1-masks[j])
            returns[j] = rewards[j] + gamma * (prev_return * masks[j] + bootstrap)
            prev_return = returns[j]
        deltas = returns.detach() - critic(states)
        return deltas.view(-1,1), returns.view(-1,1)

    def update(self, optim, trajectory, iters=4):
        log_probs = torch.stack(trajectory["log_probs"]).to(device)
        for i in range(iters):
            deltas, _ = self.get_phi(trajectory, self.critic)
            phi = (deltas-deltas.mean())/deltas.std()
            crit_loss = torch.mean(deltas ** 2)
            pol_loss = -torch.mean(log_probs.view(-1,1) * phi.detach())
            loss = pol_loss+crit_loss
            optim.zero_grad()
            loss.backward()
            optim.step()


class Agent(SimpleA2C):
    def __init__(self, input_dim, hidden_dim, output_dim, dim=2, lookahead=1):
        super(Agent, self).__init__(input_dim, hidden_dim, output_dim, dim, lookahead)
        self.pi = SimplePolicy(input_dim, hidden_dim, output_dim).to(device)              
        hard_update(self.pi, self.beta)

    def get_gaussian_policy_loss(self, trajectory, pi, critic):
        states = torch.stack(trajectory["states"])
        actions = torch.stack(trajectory["actions"])
        fixed_log_probs = torch.stack(trajectory["log_probs"]).detach()
        mus, stds = pi(states)
        dist = Normal(mus, stds)
        log_probs = torch.sum(dist.log_prob(actions), dim=-1)
        deltas, returns = self.get_phi(trajectory, critic)
        phi = (deltas / returns.std())
        ratio = torch.exp(log_probs.view(-1, 1) - fixed_log_probs.view(-1,1))
        loss = -ratio * phi.detach()
        return loss.mean()

    def trpo_update(self, trajectory, policy_loss_fn, pi, beta, critic, fvp, max_kl=1e-2):
        print("Updating agent.")
        states = torch.stack(trajectory["states"]).to(device)
        actions = torch.stack(trajectory["actions"]).to(device)
        policy_loss = policy_loss_fn(trajectory, pi, critic)
        grads = torch.autograd.grad(policy_loss, pi.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        stepdir = conjugate_gradient(fvp, -loss_grad, states, actions, pi, beta)
        shs = 0.5*(stepdir.dot(fvp(stepdir, states, actions, pi, beta)))
        lm = torch.sqrt(max_kl/shs)
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)
        old_params = get_flat_params_from(beta)
        _, params = linesearch(trajectory, pi, critic, policy_loss_fn, old_params, fullstep, expected_improve)
        set_flat_params_to(pi, params)
        hard_update(beta, pi)

    def update(self, opt, trajectory, iters=4):
        for _ in range(iters):
            deltas, _ = self.get_phi(trajectory, self.critic)
            opt.zero_grad()
            loss = torch.mean(deltas ** 2)
            loss.backward()
            opt.step()
        self.trpo_update(trajectory, self.get_gaussian_policy_loss, self.pi, self.beta, self.critic, gaussian_fvp)

       
def gaussian_fvp(gradient_vector, states, actions, pi, beta):
    mus_pi, sigmas_pi = pi(states)
    mus_beta, sigmas_beta = beta(states)
    vars_pi = sigmas_pi.pow(2)
    vars_beta = sigmas_beta.pow(2)
    kl = sigmas_pi.log()-sigmas_beta.log()+(vars_beta+(mus_beta-mus_pi).pow(2))/(2.*vars_pi)-0.5
    kl = torch.sum(kl, dim=-1).mean()
    grads = torch.autograd.grad(kl, pi.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    kl_v = flat_grad_kl.dot(gradient_vector)
    grads = torch.autograd.grad(kl_v, pi.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
    return flat_grad_grad_kl

def conjugate_gradient(fvp, gradient_vector, states, actions, pi, beta, n_steps=10, residual_tol=1e-10):
    """
    Estimate the function Fv = g, where F is the FIM, and g is the gradient.
    Since dx ~= F^{-1}g for a stochastic process, v is dx. The CG algorithm 
    assumes the function is locally quadratic. In order to ensure our step 
    actually improves the policy, we need to do a linesearch after this.
    """
    x = torch.zeros(gradient_vector.size()).to(device)
    r = gradient_vector.clone()
    p = gradient_vector.clone()
    rdotr = torch.dot(r, r)
    for i in range(n_steps):
        fisher_vector_product = fvp(p, states, actions, pi, beta)
        alpha = rdotr/p.dot(fisher_vector_product)
        x = x + alpha * p
        r -= alpha*fisher_vector_product
        new_rdotr = r.dot(r)
        tau = new_rdotr/rdotr
        p = r + tau * p
        rdotr = new_rdotr
        if rdotr <= residual_tol:
            break
    return x

def linesearch(trajectory, pi, critic, policy_loss, old_params, fullstep, expected_improve, max_backtracks=10, accept_ratio=.1):
    """
    Conducts an exponentially decaying linesearch to guarantee that our update step improves the
    model. 
    """
    set_flat_params_to(pi, old_params)
    fval = policy_loss(trajectory, pi, critic).data
    steps = 0.5**torch.arange(max_backtracks).to(device).float()
    for n, stepfrac in enumerate(steps):
        xnew = old_params+stepfrac*fullstep
        set_flat_params_to(pi, xnew)
        newfval = policy_loss(trajectory, pi, critic).data
        actual_improve = fval-newfval
        expected_improve = expected_improve*stepfrac
        ratio = actual_improve/expected_improve
        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            return True, xnew
    return False, old_params

def get_flat_params_from(model):
    """
    Get flattened parameters from a network. Returns a single-column vector of network weights.
    """
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    """
    Take a single-column vector of network weights, and manually set the weights of a given network
    to those contained in the vector.
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind+flat_size].view(param.size()))
        prev_ind += flat_size

def hard_update(target, source):
    """
    Updates a target network based on a source network. I.e. it makes N* == N for two networks
    N* and N.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)