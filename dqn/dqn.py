import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import copy


# AGENT PARAMETERS
HIDDEN_DIM = 32
EPS_MIN = 0.001
EPS_DECAY = 0.999
MEM_SIZE = int(32000)
BATCH_SIZE = int(64)
LR = 0.001

# TRAINING PARAMETERS
MAX_STEPS = 10000

# OTHER CONFIGURATIONS
verbose_every = 1000


class MLP(nn.Module):
    ''' simple mlp for dqn'''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.layer2 = nn.Linear(HIDDEN_DIM, output_dim)

    def forward(self, x):
        ''' this is outputing q-values - how good would each action be? '''
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    

class Memory():
    ''' simple memory buffer'''
    def __init__(self, size, obs_space, act_space):

        self.pointer = 0
        self.size = size     

        self.obss= torch.zeros((size, obs_space), dtype=torch.float32)
        self.acts = torch.zeros((size), dtype=torch.int)
        self.obs_s = torch.zeros((size, obs_space), dtype=torch.float32)
        self.rews = torch.zeros((size), dtype=torch.float32)
        self.dones = torch.zeros((size), dtype=torch.bool)

    def write(self, obs, act, rew, obs_, done):
        i = self.pointer % self.size

        self.obss[i] = obs
        self.acts[i] = act
        self.obs_s[i] = obs_
        self.rews[i] = rew
        self.dones[i] = done

        self.pointer += 1

    def sample(self, batch_size):
        if self.pointer < batch_size:
            return None # not enough entries in the buffer
        
        # sample batch_size indices
        sample_limit = self.pointer if self.pointer < self.size else self.size
        indices = torch.randint(0, sample_limit, (batch_size, ))

        return (
            self.obss[indices],
            self.acts[indices],
            self.rews[indices],
            self.obs_s[indices],
            self.dones[indices]
        )


def get_env(id:str='CartPole-v1'):
    ''' instantiates env, returns object and act/obs spaces'''
    env = gym.make(id, render_mode="rgb_array")
    act_space = env.action_space.n
    obs_space = env.observation_space.shape[0]
    return env, act_space, obs_space


def train(env, q, qt, mem, act_space):

    # control variables / bookkeeping
    n_steps = 0
    n_episodes = 0
    eps = 1.0

    # initial reset
    obs, _ = env.reset() # _ ignores info

    # iterates environment
    while n_steps < MAX_STEPS:

        # select action
        if torch.rand((1,)).item() < eps:
            # randomize action (exploration)
            act = torch.randint(0, act_space, (1,)).item()
        else:
            # forward q_net
            act = torch.argmax(q(torch.tensor(obs, dtype=torch.float32)))

        # step
        obs_, rew, term, trun, _  = env.step(act.item() if type(act) != int else act) # _ ignores info
        done = term or trun

        # buffer update
        mem.write(
            torch.tensor(obs, dtype=torch.float32), 
            torch.tensor(act, dtype=torch.int), 
            torch.tensor(rew, dtype=torch.float32), 
            torch.tensor(obs_, dtype=torch.float32), 
            torch.tensor(done, dtype=torch.bool)
            )

        # episode end
        if done: # episode over
            n_episodes += 1
            obs, _ = env.reset() # _ ignores info
        else:
            obs = obs_

        # dqn var updates
        eps = eps * EPS_DECAY if eps * EPS_DECAY > EPS_MIN else EPS_MIN

        # dqn learning
        sample = mem.sample(BATCH_SIZE)
        if sample is not None:
            obss, acts, rews, obs_s, dones = sample
            # I AM HERE
            
        # bookkepping
        n_steps += 1

        # verbose
        if n_steps % verbose_every == 0:
            print(f' Step: {n_steps}\n Episode: {n_episodes}\n \
                  ...')
            

if __name__ == '__main__':
    ''' creates environment, neural network, trains it, evals it'''

    # create env
    env, act_space, obs_space = get_env()

    # create q network agent
    q = MLP(obs_space, act_space)
    qt = copy.deepcopy(q)
    mem = Memory(MEM_SIZE, obs_space, act_space)

    train(env, q, qt, mem, act_space)