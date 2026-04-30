import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from matplotlib import pyplot as plt
import random


# general parameters
SEED = 42

# agent parameters
HIDDEN_DIMS_DEFAULT = [32, 32]

# training parameters
MAX_STEPS = int(1e6)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    ''' simple mlp '''
    def __init__(self, x_dim, hidden_dims, y_dim):
        super().__init__()
        dims = [x_dim] + hidden_dims + [y_dim]
        self.fcs = nn.ModuleList() # list that is tracked in model.parameters()
        for i in range(len(dims) - 1):
            self.fcs.append(nn.Linear(dims[i], dims[i+1], dtype=torch.float32))
        
    def forward(self, x):
        for i in range(len(self.fcs) - 1):
            x = F.relu(self.fcs[i](x))
        return self.fcs[-1](x)
    

def get_env(id='Ant-v5'):
    ''' instantiates env, returns object and act/obs spaces'''
    env = gym.make(id)
    act_space = env.action_space
    obs_space = env.observation_space
    return env, act_space, obs_space


def train(env, policy_net, value_net):
    ''' trains agent on env '''
    # control variables
    steps = 0

    # reset env
    obs, _ = env.reset(seed=SEED)

    print(obs)

    # training loop
    while steps < MAX_STEPS:
        break


if __name__ == '__main__':
    # set seed for a deterministic run
    set_seed(SEED)

    # init environment
    env, act_space, obs_space = get_env()

    # agent - for policy, we output mean and std for each action dimension
    policy_net = MLP(obs_space.shape[0], HIDDEN_DIMS_DEFAULT, act_space.shape[0] * 2)
    value_net = MLP(obs_space.shape[0], HIDDEN_DIMS_DEFAULT, 1) 

    train(env, policy_net, value_net)