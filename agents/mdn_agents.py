import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class AgentMDN(torch.nn.Module):
    def __init__(self, agents, hidden_dim=128):
        super(AgentMDN, self).__init__()
        self.agents = torch.nn.ModuleList([a for a in agents])
        self.num_policies = len(self.agents)
        self.num_actions = self.agents[0].output_dim
        self.input_dim = int(self.agents[0].input_dim)
        self.mixing = nn.Sequential(
                                    nn.Linear(self.input_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim, self.num_policies))
        self.value = nn.Sequential(
                                    nn.Linear(self.input_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim, 1))
    
    def get_integrated_value(self, x):
        return self.pos_val_fn(x)
        
    def forward(self, x):
        mixing = F.softmax(self.mixing(x), dim=-1)
        value = self.value(x)
        return mixing, value        

    def select_action(self, x):
        scores, value = self.forward(x)
        cat_dist = Categorical(scores)
        pol_idx = cat_dist.sample().int()
        actions = []
        for i in range(pol_idx.size(0)):
            mu, sigma = self.agents[pol_idx[i]](x[i])
            norm_dist = Normal(mu, sigma)
            action = norm_dist.sample()
            actions.append(action)
        actions = torch.stack(actions)
        p = torch.zeros(pol_idx.size(0), self.num_actions)
        for i, a in enumerate(self.agents):
            mu, sigma = a(x)
            dist = Normal(mu, sigma)
            lp = dist.log_prob(actions)
            p += scores[:, i].unsqueeze(1)*torch.exp(lp)
        log_probs = torch.sum(torch.log(p), dim=-1)
        return action, value, log_probs, 0

    def get_phi(self, trajectory, critic, gamma=0.99):
        rewards = torch.stack(trajectory["rewards"]).to(device)
        next_states = torch.stack(trajectory["next_states"]).to(device)
        values = torch.stack(trajectory["values"]).to(device)
        masks = torch.stack(trajectory["masks"]).to(device)
        masks = masks.unsqueeze(2)
        returns = torch.Tensor(rewards.size()).to(device)
        deltas = torch.Tensor(rewards.size()).to(device)
        prev_return = 0
        for j in reversed(range(rewards.size(0))):
            next_val = critic(next_states[j])
            if j == rewards.size(0)-1:
                bootstrap = next_val
            else:
                bootstrap = (next_val*(1-masks[j]))
            returns[j] = rewards[j]+gamma*(prev_return*masks[j]+bootstrap)
            prev_return = returns[j]
        deltas = returns.detach()-values
        return (deltas.squeeze(2)).view(-1,1), (returns.squeeze(2)).view(-1,1)
    
    def update(self, optim, trajectory, iters=1):
        log_probs = torch.stack(trajectory["log_probs"]).to(device)
        traj = {
                    "states" : trajectory["states"],
                    "actions" : trajectory["actions"],
                    "log_probs" : trajectory["log_probs"],
                    "rewards" : trajectory["rewards"],
                    "masks" : trajectory["masks"],
                    "values" : trajectory["values"],
                    "next_states" : trajectory["next_states"]}
    
        val_fn_preds = [self.pos_val_fn(state) for state in trajectory["goal_position"]]
        val_fn_traj = {
                    "states" : trajectory["goal_position"],
                    "rewards" : trajectory["rewards"],
                    "masks" : trajectory["masks"],
                    "values" : val_fn_preds,
                    "next_states" : trajectory["next_goal_position"]}
                    
        for i in range(iters):
            deltas, _ = self.get_phi(traj, self.critic)
            val_fn, _ = self.get_phi(val_fn_traj, self.pos_val_fn)
            phi = (deltas-deltas.mean())/deltas.std()
            crit_loss = torch.mean(deltas**2)
            val_fn_loss = torch.mean(val_fn**2)
            pol_loss = -torch.mean(log_probs.view(-1,1)*phi.detach())
            loss = pol_loss+crit_loss+val_fn_loss
            optim.zero_grad()
            if i < iters-1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optim.step()
        