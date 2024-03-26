import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda")
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, state, action, next_state, reward):
        # Convert states to tensors here, if not already
        reward = torch.tensor([reward], device=device, dtype=torch.float32)

        if next_state is not None:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        self.memory.append(Transition(state, action, next_state, reward))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_channels, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(11*11*32, 256)  
        self.fc2 = nn.Linear(256, outputs)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
class DQN_Agent:
    def __init__(self, n_actions, n_observations, input_channels):

        self.BATCH_SIZE = 10000
        self.GAMMA = 0.99
        self.EPS_START = 1.0
        self.EPS_END = 0.01
        self.EPS_DECAY = 300000
        self.TAU = 0.05
        self.LR = 0.0004

        self.n_actions = n_actions
        self.policy_net = DQN(input_channels, n_actions).to(device)
        self.target_net = DQN(input_channels, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(50000)
        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        # Reshape tensor to match [N, C, H, W] format
        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display
            display.clear_output(wait=True)
            display.display(plt.gcf())
    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.scores, label='Scores')
        plt.plot(self.mean_scores, label='Mean Scores')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.losses, label='Losses')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()