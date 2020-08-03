import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal, Categorical
import agents.utilities as utils
#import settings as cfg
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

artanh = lambda z : 0.5 * torch.log((1 + z)/(1 - z + 1e-10) + 1e-10)
invsig = lambda z : -torch.log((1 - z) / (z + 1e-10) + 1e-10)


class CategoricalPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CategoricalPolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.score = nn.Sequential(
                                nn.Linear(input_dim, hidden_dim),
                                nn.Tanh(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.Tanh(),
                                nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        score = self.score(x)
        return torch.softmax(score, dim=-1)
    
    def get_dist(self, states):
        score = self.forward(states)
        dist = Categorical(score)
        return dist

    def select_action(self, x):
        dist = self.get_dist(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def get_log_prob(self, states, actions):
        dist = self.get_dist(states)
        lp = dist.log_prob(actions)
        return lp
    
    def get_entropy(self, states):
        dist = self.get_dist(states)
        entropy = dist.entropy()
        return entropy


class DeterministicPolicy(nn.Module):
    """
    Deterministic policy function for continuous control tasks. This policy relies
    on noise injection to explore the space (i.e. OU noise as is used in DDPG-based
    algorithms)
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeterministicPolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mu = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                               nn.Tanh(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.Tanh(),
                               nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        mu = self.mu(x)
        return mu

    def select_action(self, x):
        return self.forward(x)

class CompositePolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, cont_output_dim, cat_output_dim):
        super(CompositePolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cont_output_dim = cont_output_dim
        self.cat_output_dim = cat_output_dim
        self.score = nn.Sequential(
                                nn.Linear(input_dim, hidden_dim),
                                nn.Tanh(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.Tanh(),
                                nn.Linear(hidden_dim, cat_output_dim))
        self.mu = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                               nn.Tanh(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.Tanh(),
                               nn.Linear(hidden_dim, cont_output_dim))
        self.logsigma = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                               nn.Tanh(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.Tanh(),
                               nn.Linear(hidden_dim, cont_output_dim))

    def forward(self, x):
        mu = self.mu(x)
        logsigma = torch.clamp(self.logsigma(x), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        score = torch.softmax(self.score(x), dim=-1)
        return mu, logsigma, score
    
    def get_cont_dist(self, states):
        mu, logsigma, _ = self.forward(states)
        sigma = torch.exp(logsigma)
        dist = Normal(mu, sigma)
        return dist
    
    def get_cat_dist(self, states):
        _, _, score = self.forward(states)
        dist = Categorical(score)
        return dist

    def select_action(self, x):
        cont_dist = self.get_cont_dist(x)
        cat_dist = self.get_cat_dist(x)
        cont_action = cont_dist.sample()
        cat_action = cat_dist.sample()
        log_prob = torch.sum(cont_dist.log_prob(cont_action), dim=-1, keepdim=True) + cat_dist.log_prob(cat_action)
        entropy = cont_dist.entropy() + cat_dist.entropy()
        return torch.cat([cont_action, cat_action], dim=-1), log_prob, entropy

    def get_log_prob(self, states, actions):
        cont_dist = self.get_cont_dist(states)
        cat_dist = self.get_cat_dist(states)
        cont_action = actions[:,:self.cont_output_dim]
        cat_action = actions[:, self.cont_output_dim:]
        lp = torch.sum(cont_dist.log_prob(cont_action), dim=-1, keepdim=True) + cat_dist.log_prob(cat_action)
        return lp
    
    def get_entropy(self, states):
        cont_dist = self.get_cont_dist(states)
        cat_dist = self.get_cat_dist(states)
        entropy = torch.sum(cont_dist.entropy(), dim=-1, keepdim=True) + cat_dist.entropy()
        return entropy


class SquashedDeterministicPolicy(DeterministicPolicy):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SquashedDeterministicPolicy, self).__init__()
    
    def forward(self, x):
        return torch.tanh(super(SquashedDeterministicPolicy, self).forward(x))


class IndependentGaussianPolicy(DeterministicPolicy):
    """
    Gaussian policy function for continuous control tasks. Assumes all actions are
    independent (i.e. diagonal covariance matrix).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IndependentGaussianPolicy, self).__init__(input_dim, hidden_dim, output_dim)   
        self.logsigma = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                               nn.Tanh(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.Tanh(),
                               nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        mu = self.mu(x)
        logsigma = torch.clamp(self.logsigma(x), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, logsigma
    
    def get_dist(self, states):
        mu, logsigma = self.forward(states)
        sigma = torch.exp(logsigma)
        dist = Normal(mu, sigma)
        return dist

    def select_action(self, x):
        dist = self.get_dist(x)
        action = dist.sample()
        log_prob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
        entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)
        return action, log_prob, entropy

    def get_log_prob(self, states, actions):
        dist = self.get_dist(states)
        return torch.sum(dist.log_prob(actions), dim=-1, keepdim=True)
    
    def get_entropy(self, states):
        dist = self.get_dist(states)
        entropy = dist.entropy()
        return torch.sum(entropy, dim=-1, keepdim=True)


class SquashedGaussianPolicy(IndependentGaussianPolicy):
    """
    Sqaushed Gaussian policy function for continuous control tasks. Assumes all actions are
    independent (i.e. diagonal covariance matrix), and uses a tanh squashing function to
    constrain all actions to [-1, 1]. In order to recover the log_prob, we need to subtract
    log(1 -tanh(action)^2) from the original log_prob (to understand why, see normalizing
    flows and the change of variables formula).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SquashedGaussianPolicy, self).__init__(input_dim, hidden_dim, output_dim)
        
    def select_action(self, x):
        action, log_prob, entropy = super(SquashedGaussianPolicy, self).select_action(x)
        action = torch.tanh(action)
        log_prob -= torch.sum(torch.log((1 - action.pow(2)) + 1e-10), dim=-1, keepdim=True)
        return action, log_prob, entropy
    
    def get_log_prob(self, states, actions):
        log_prob = super(SquashedGaussianPolicy, self).get_log_prob(states, artanh(actions))
        log_prob -= torch.sum(torch.log((1 - actions.pow(2)) + 1e-10), dim=-1, keepdim=True)
        return log_prob


class ShapedPolicy(IndependentGaussianPolicy):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ShapedPolicy, self).__init__(input_dim, hidden_dim, output_dim)
        
    def select_action(self, x):
        action, log_prob, entropy = super(ShapedPolicy, self).select_action(x)
        action = torch.sigmoid(action) ** 0.5
        log_prob -= torch.sum(torch.log(0.5 * action * (1 - action ** 0.5)), dim=-1, keepdim=True)
        return action, log_prob, entropy
    
    def get_log_prob(self, states, actions):
        log_prob = super(ShapedPolicy, self).get_log_prob(states, invsig(actions ** 2))
        log_prob -= torch.sum(torch.log(0.5 * actions * (1 - actions ** 0.5)), dim=-1, keepdim=True)
        return log_prob


class MVGaussianPolicy(nn.Module):
    """
    Multivariate Gaussian policy function for continuous control tasks. Assumes all actions 
    are correlated. To do this, we output a triangular scaling matrix, where the diagonal
    elements are constrained to being positive. The downside of this type of policy is that
    the number of outputs from the network scales with the square of the number of actions.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MVGaussianPolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tril_dim = int(output_dim * (output_dim + 1) / 2)

        self.mu = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, output_dim))
        self.cov = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, self.tril_dim))
    
    def select_action(self, x):
        dist = self.get_dist(x)
        action = dist.sample()
        log_prob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
        entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)
        return action, log_prob, entropy

    def reshape_output(self, mu, cov):
        tril = torch.zeros((mu.size(0), mu.size(1), mu.size(1)))
        tril_indices = torch.tril_indices(row=mu.size(1), col=mu.size(1), offset=0)
        tril[:, tril_indices[0], tril_indices[1]] = cov
        diag_indices = np.diag_indices(tril.shape[1])
        tril[:, diag_indices[0], diag_indices[1]] = torch.exp(tril[diag_indices[0], diag_indices[1]])
        return tril

    def forward(self, x):
        mu = self.mu(x)
        cov = self.cov(x)
        return mu, cov
    
    def get_dist(self, states):
        mu, cov = self.forward(states)
        tril = self.reshape_output(mu, cov)
        dist = MultivariateNormal(mu, scale_tril=tril)
        return dist
    
    def get_log_prob(self, states, actions):
        dist = self.get_dist(states)
        return dist.log_prob(actions)
    
    def get_entropy(self, states):
        dist = self.get_dist(states)
        return dist.entropy()
        

class SquashedMVGaussianPolicy(MVGaussianPolicy):
    """
    Sqaushed Multivariate Gaussian policy function for continuous control tasks. Assumes all 
    actions are correlated, and uses a tanh squashing function to constrain all actions to 
    [-1, 1]. In order to recover the log_prob, we need to subtract log(1 - tanh(action)^2) 
    from the original log_prob (to understand why, see normalizing flows and the change of 
    variables formula).
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SquashedMVGaussianPolicy, self).__init__(input_dim, hidden_dim, output_dim)
        
    def select_action(self, x):
        action, log_prob, entropy = super(SquashedMVGaussianPolicy, self).select_action(x)
        action = torch.tanh(action)
        log_prob -= torch.sum(torch.log((1 - action.pow(2)) + 1e-10))
        return action, log_prob, entropy

    def get_log_prob(self, states, actions):
        log_prob = super(SquashedMVGaussianPolicy, self).get_log_prob(states, artanh(actions))
        log_prob -= torch.sum(torch.log((1 - actions.pow(2)) + 1e-10), dim=-1, keepdim=True)
        return log_prob

        
class ValueNet(nn.Module):
    """
    Simple parameterized value function. We use this for both state value functions,
    and state-action value functions. We can spawn either one or multiple heads and
    use the TD3 trick of selecting the most conservative (minimum) value function
    estimate. If the number of heads is 1, then our value function is the same as a
    standard implementation.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1):
        super(ValueNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_dim)) for _ in range(num_heads)])
        
    def forward(self, x):
        values = [h(x) for h in self.heads]
        return values
    
    def get_min_value(self, x):
        values = self.forward(x)
        data = torch.min(torch.cat(values, dim=-1), dim=-1, keepdim=True)
        min_v = data[0]
        return min_v

    def get_max_value(self, x):
        values = self.forward(x)
        data = torch.max(torch.cat(values, dim=-1), dim=-1, keepdim=True)
        max_v = data[0]
        return max_v
    
    def get_nearest_value(self, x, targ_vals):
        values = self.forward(x)
        data = torch.cat(values, dim=-1)
        dist = torch.norm(targ_vals - data, dim=-1, p=None)
        nearest = dist.topk(1, largest=False)
        return nearest.values
    
    def get_highest_correlation(self, x, targ_vals):
        values = self.forward(x)
        data = torch.cat(values, dim=-1)
        vx = data - torch.mean(data, dim=0)
        vy = targ_vals - torch.mean(targ_vals)
        cov_xy = torch.sum(vx * vy.expand_as(vx), dim=0)
        var_x = torch.sum(vx ** 2, dim=0)
        var_y = torch.sum(vy ** 2, dim=0)
        denom = torch.sqrt(var_x) * torch.sqrt(var_y)
        pearsons =  cov_xy / denom
        best = torch.max(pearsons, dim=-1, keepdim=True)
        idx = best[1]
        return data[:, idx]
