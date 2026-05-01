import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from matplotlib import pyplot as plt
import random


# general parameters
SEED = 42

# agent parameters
HIDDEN_DIMS_DEFAULT = [32, 32]
BUFFER_SIZE = 256

# training parameters
MAX_STEPS = int(1e6)
VERBOSE_EVERY = int(1e4)


# GENERAL

def set_seed(seed):
    ''' makes execution deterministic '''
    random.seed(seed)
    torch.manual_seed(seed)


# PPO

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
    
def sample_action(obs, policy_net):
    ''' samples actions from ppo network'''
    # 1. get raw outputs (dim is 2 * action dimension)
    output = policy_net(obs)
    # 2. give meaning to the outputs -> get 8 means and log(stds) for gaussian dists
    mean, log_std = output.chunk(2, dim=-1) # [1, 2, 3, 4, ...] -> [[1, 2], [3, 4], ...]
    # 3. std can not be negative, lets exp it (negatives become near zero values)
    std = log_std.exp()
    # 4. build a gaussian distribution
    normal = Normal(mean, std)
    # 5. sample action from the dist
    act = normal.rsample() # rsample is used because we need gradients to flow through the sample
    # 6. log prob - how probable the action was needs to be considered on policy gradient step
    log_prob = normal.log_prob(act).sum(-1) # we sum the log prob of each dimension
    # 7. higher entropy means a fat dist, not peaked, highly uncertain, we reward high entropy in ppo (exploration)
    entropy = normal.entropy().sum(-1)

    # !!! (-1) is the last dimention, if obs is a one dimentional tensor (obs_shape, )
    # we wouldnt need it, it is the same thing as (0), but once we have batches, the shape of
    # the obs will be (batch_size, obs_shape), which means our act will be (batch_size, act_shape)
    # so we need to make sure we are summing across dimentions of the action and not the batch with (-1) indexing

    return act, log_prob, entropy
    
def add_to_buffer(i, buffer, ):
    ''' adds transition to buffer '''
    return None

# ENVIRONMENT 

def get_env(id='Ant-v5'):
    ''' instantiates env, returns object and act/obs shapes'''
    env = gym.make(id)
    act_shape = env.action_space.shape[0]
    obs_shape = env.observation_space.shape[0]
    return env, act_shape, obs_shape


# TRAINING

def train(env, policy_net, value_net, buffer):
    ''' trains agent on env '''
    # control variables
    steps = 0
    episode = 0

    # reset env
    obs, _ = env.reset(seed=SEED)

    # training loop
    while steps < MAX_STEPS:
        # fill in buffer with trajectories
        for i in range(BUFFER_SIZE):
            # get action
            obs = torch.tensor(obs, dtype=torch.float32)
            act, log_prob, _ = sample_action(obs, policy_net) # entropy from old policy not used during learning step

            # step env
            obs_, rew, term, trun, _ = env.step(act.detach().numpy())
            
            # fill in buffer

            # is the episode over?
            if term or trun:
                obs, _ = env.reset() 
                episode += 1
            else: # update obs
                obs = obs_

            # update control variables
            steps += 1

            # 6. verbose
            if steps % VERBOSE_EVERY == 0:
                print(f'Step: {steps}, Episode: {episode} \
                    ...')
                
        # learn from collected trajectories
        # ...



if __name__ == '__main__':
    # set seed for a deterministic run
    set_seed(SEED)

    # init environment
    env, act_shape, obs_shape = get_env()

    # env specifications verbose
    print(f'--- Training PPO in Mujoco Locomotion Environments ---\n')
    print(f'Environment Info:')
    print(f'ID: {env.spec.id}')
    print(f'Observation Space: {obs_shape}')
    print(f'Action Space: {act_shape}\n')

    # init agent - for policy, we output mean and std for each action dimension
    policy_net = MLP(obs_shape, HIDDEN_DIMS_DEFAULT, act_shape * 2)
    value_net = MLP(obs_shape, HIDDEN_DIMS_DEFAULT, 1)

    # agent specifications verbose
    print('Agent Networks:')
    print(policy_net)
    print(value_net, end='\n')

    # init buffer
    buffer = {} # dict will be lighter then creating clas just for storage
    buffer['obs'] = torch.zeros((BUFFER_SIZE, obs_shape), dtype=torch.float32)
    buffer['act'] = torch.zeros((BUFFER_SIZE, act_shape), dtype=torch.float32)
    buffer['log_prob'] = torch.zeros((BUFFER_SIZE), dtype=torch.float32)
    buffer['reward'] = torch.zeros((BUFFER_SIZE), dtype=torch.float32)
    buffer['obs_'] = torch.zeros((BUFFER_SIZE, obs_shape), dtype=torch.float32)
    buffer['term'] = torch.zeros((BUFFER_SIZE), dtype=torch.bool)
    buffer['trun'] = torch.zeros((BUFFER_SIZE), dtype=torch.bool)

    # train
    print(f'Starting training for {MAX_STEPS}')
    train(env, policy_net, value_net, buffer)