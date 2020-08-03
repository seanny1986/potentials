import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch import autograd
import numpy as np
import agents.agents as agents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0., std=0.1)
        torch.nn.init.constant_(m.bias, 0.1)


class CompositePolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CompositePolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.termination_score = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        termination_score = torch.softmax(self.termination_score(x), dim=-1)
        return termination_score


class TerminationPolicyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TerminationPolicyLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.__lstm = nn.LSTM(input_dim, hidden_dim, 1)
        self.__score = nn.Linear(hidden_dim, output_dim)
    
    def step(self, x, hidden=None):
        hx, cx = self.__lstm(x.unsqueeze(1), hidden)
        score = F.softmax(self.__score(hx.squeeze(1)), dim=-1)
        return score, cx

    def forward(self, x, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(x)
        scores = torch.zeros(steps, 1, 1)
        for i in range(steps):
            if force or i == 0:
                input = x[i]
            score, hidden = self.step(input, hidden)
            scores[i] = score
        return scores, hidden


class TerminationPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TerminationPolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.termination_score = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        termination_score = torch.softmax(self.termination_score(x), dim=-1)
        return termination_score

class TermA2C(agents.SimpleA2C):
    def __init__(self, input_dim, hidden_dim, output_dim, dim):
        super(TermA2C, self).__init__(input_dim, hidden_dim, output_dim, dim)
        self.term_beta = TerminationPolicy(input_dim, 64, 2).to(device)
        self.term_critic = nn.Sequential(nn.Linear(input_dim, 64),
                                         nn.Tanh(),
                                         nn.Linear(64, 64),
                                         nn.Tanh(),
                                         nn.Linear(64, 1)).to(device)

    def terminate(self, x):
        score = self.term_beta(x)
        dist = Categorical(score)
        term = dist.sample()
        logprob = dist.log_prob(term)
        entropy = dist.entropy()
        return term, logprob, entropy
    
    def select_action(self, x):
        term, log_prob_term, entropy_term = self.terminate(x)
        mu, sigma = self.beta(x)
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
        entropy = torch.sum(entropy, dim=-1, keepdim=True)
        full_action = torch.cat([action.float(), term.unsqueeze(1).float()], dim=-1)
        value = self.critic(x)
        return full_action, (value, None), (log_prob, log_prob_term), (entropy, entropy_term)

    def update(self, optim, trajectory, iters=1):
        log_probs = torch.stack(trajectory["log_probs"])
        term_log_probs = torch.stack(trajectory["term_log_probs"])
        
        traj = {
                    "states" : trajectory["states"],
                    "rewards" : trajectory["rewards"],
                    "masks" : trajectory["masks"],
                    "next_states" : trajectory["next_states"]}
        
        term_traj = {
                    "states" : trajectory["states"],
                    "rewards" : trajectory["term_rew"],
                    "masks" : trajectory["masks"],
                    "next_states" : trajectory["next_states"]}

        for i in range(iters):
            deltas, returns = self.get_phi(traj, self.critic)
            term_deltas, term_returns = self.get_phi(term_traj, self.term_critic)
            phi = deltas / returns.std()
            term_phi = term_deltas / term_returns.std()
            crit_loss = torch.mean(deltas ** 2)
            term_crit_loss = torch.mean(term_deltas ** 2)
            pol_loss = -torch.mean(log_probs.view(-1,1) * phi.detach())
            term_pol_loss = -torch.mean(term_log_probs.view(-1,1)*term_phi.detach())
            loss = pol_loss + term_pol_loss + crit_loss + term_crit_loss
            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()


class Agent(TermA2C):
    def __init__(self, input_dim, hidden_dim, output_dim, dim=2):
        super(Agent, self).__init__(input_dim, hidden_dim, output_dim, dim)
        self.pi = agents.SimplePolicy(input_dim, hidden_dim, output_dim).to(device)
        self.term_pi = TerminationPolicy(input_dim, 64, 2).to(device)        
        hard_update(self.pi, self.beta)
        hard_update(self.term_pi, self.term_beta)

    def trpo_update(self, trajectory, policy_loss_fn, pi, beta, critic, fvp, max_kl=1e-2):
        states = torch.stack(trajectory["states"])
        policy_loss = policy_loss_fn(trajectory, pi, critic)
        grads = torch.autograd.grad(policy_loss, pi.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        stepdir = conjugate_gradient(fvp, -loss_grad, states, pi, beta)
        shs = 0.5 * (stepdir.dot(fvp(stepdir, states, pi, beta)))
        lm = torch.sqrt(max_kl/shs)
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)
        old_params = get_flat_params_from(beta)
        _, params = linesearch(trajectory, pi, critic, policy_loss_fn, old_params, fullstep, expected_improve)
        set_flat_params_to(pi, params)
        hard_update(beta, pi)

    def update(self, opt, trajectory, iters=3):
        print("Updating agent.")
        def get_gaussian_policy_loss(trajectory, pi, critic):
            states = torch.stack(trajectory["states"])
            actions = torch.stack(trajectory["actions"])
            deltas, returns = self.get_phi(trajectory, critic)
            phi = deltas / returns.std()
            mus, stds = pi(states)
            dist = Normal(mus, stds)
            log_probs = torch.sum(dist.log_prob(actions), dim=-1)
            log_probs = log_probs.view(-1, 1)
            loss = -log_probs * phi.detach()
            return loss.mean()

        def get_categorical_policy_loss(trajectory, pi, critic):
            states = torch.stack(trajectory["states"])
            actions = torch.stack(trajectory["actions"])
            term_rew = torch.stack(trajectory["term_rew"]).view(-1, 1)
            scores_pi = pi(states)
            dist = Categorical(scores_pi)
            log_probs = dist.log_prob(actions)
            log_probs = log_probs.view(-1, 1)
            loss = -log_probs*term_rew.detach()
            return loss.mean()

        traj = {
                    "states" : trajectory["states"],
                    "actions" : trajectory["actions"],
                    "rewards" : trajectory["rewards"],
                    "masks" : trajectory["masks"],
                    "next_states" : trajectory["next_states"]}
        
        term_traj = {
                    "states" : trajectory["states"],
                    "actions" : trajectory["terminations"],
                    "rewards" : trajectory["rewards"],
                    "term_rew" : trajectory["term_rew"],
                    "masks" : trajectory["masks"],
                    "next_states" : trajectory["next_states"]}

        for _ in range(iters):
            deltas, _ = self.get_phi(traj, self.critic)
            #term_deltas, _ = self.get_phi(term_traj, self.term_critic)
            opt.zero_grad()
            crit_loss = torch.mean(deltas ** 2)
            #term_crit_loss = torch.mean(term_deltas ** 2)
            crit_loss.backward()
            opt.step()
        self.trpo_update(traj, 
                        get_gaussian_policy_loss, 
                        self.pi, 
                        self.beta, 
                        self.critic, 
                        gaussian_fvp)
        self.trpo_update(term_traj, 
                        get_categorical_policy_loss, 
                        self.term_pi, 
                        self.term_beta, 
                        self.critic, 
                        categorical_fvp)



def gaussian_fvp(gradient_vector, states, pi, beta):
    """
    Calculates the Fisher Vector Product using the KL-divergence of two Independent Gaussian
    Policies.
    """
    mus_pi, logsigmas_pi = pi(states)
    mus_beta, logsigmas_beta = beta(states)
    sigmas_pi = torch.exp(logsigmas_pi)
    sigmas_beta = torch.exp(logsigmas_beta)
    dist_pi = Normal(mus_pi, sigmas_pi)
    dist_beta = Normal(mus_beta, sigmas_beta)
    kl = torch.distributions.kl_divergence(dist_beta, dist_pi)
    kl = torch.mean(torch.sum(kl, dim=-1))
    grads = torch.autograd.grad(kl, pi.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    kl_v = flat_grad_kl.dot(gradient_vector)
    grads = torch.autograd.grad(kl_v, pi.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
    return flat_grad_grad_kl

def categorical_fvp(gradient_vector, states, pi, beta):
    scores_pi = pi(states)
    scores_beta = beta(states)
    dist_pi = Categorical(scores_pi)
    dist_beta = Categorical(scores_beta)
    kl = torch.distributions.kl_divergence(dist_beta, dist_pi)
    kl = torch.mean(torch.sum(kl, dim=-1))
    grads = torch.autograd.grad(kl, pi.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
    kl_v = flat_grad_kl.dot(gradient_vector)
    grads = torch.autograd.grad(kl_v, pi.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
    return flat_grad_grad_kl

def conjugate_gradient(fvp, gradient_vector, states, pi, beta, n_steps=10, residual_tol=1e-10):
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
        fisher_vector_product = fvp(p, states, pi, beta)
        alpha = rdotr/p.dot(fisher_vector_product)
        x += alpha*p
        r -= alpha*fisher_vector_product
        new_rdotr = r.dot(r)
        tau = new_rdotr/rdotr
        p = r+tau*p
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