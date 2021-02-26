from typing import Union

import torch
from torch import nn

import torch.nn.functional as F


# architecture from Levine paper:
class SoftQNetwork(nn.Module):
    def __init__(self,h, w, num_actions):
        super(SoftQNetwork,self).__init__()
        self.in_channels = 3
        self.out_channels = [32,32,32]
        self.kernel_sizes = [4,3,3]
        self.strides = [2,2,2]
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels[0], kernel_size=self.kernel_sizes[0], stride=self.strides[0]),
            #nn.BatchNorm2d(out_channels[0])
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[0], out_channels=self.out_channels[1], kernel_size=self.kernel_sizes[1], stride=self.strides[1]),
            #nn.BatchNorm2d(out_channels[1])
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[1], out_channels=self.out_channels[2], kernel_size=self.kernel_sizes[2], stride=self.strides[2]),
            #nn.BatchNorm2d(out_channels[2])
            nn.ReLU()
        )
        self.actionFC = nn.Sequential(
            nn.Linear(num_actions,33),
            nn.ReLU()
        )
        self.finishFC = nn.Sequential(
            nn.Linear(1617,32),
            nn.Linear(32,32),
            nn.Linear(32,1)
        )


    def forward(self, image, action, timestep):
        timestep=timestep.unsqueeze(1)
        convolution_output = self.convolution(image)
        time_tiled = torch.repeat_interleave(timestep,49,dim=0).reshape((timestep.shape[0],1,7,7))
        state_output = torch.cat((time_tiled,convolution_output),dim=1)

        action_output = self.actionFC(action).unsqueeze(2).unsqueeze(2)

        state_action = torch.add(state_output,action_output)
        state_action_flat = torch.flatten(state_action,start_dim=1)
        q = self.finishFC(state_action_flat)

        # timestep_tensor = torch.tensor([[[timestep]]])
        # action_tensor = torch.tensor([action])
        # convolution_output = self.convolution(image)
        # time_tiled = torch.repeat_interleave(timestep,7,dim=0).repeat_interleave(7,dim=1)
        # state_output = torch.cat((time_tiled,convolution_output),dim=2)
        # action_output = self.actionFC(action_tensor).unsqueeze(0).unsqueeze(0)
        # state_action = torch.add(state_output,action_output)
        # q = self.finishFC(state_action)
        return q


class ValueNetwork(nn.Module):
    def __init__(self,h, w, num_actions):
        super(ValueNetwork,self).__init__()
        self.in_channels = 3
        self.out_channels = [32,32,32]
        self.kernel_sizes = [4,3,3]
        self.strides = [2,2,2]
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels[0], kernel_size=self.kernel_sizes[0], stride=self.strides[0]),
            #nn.BatchNorm2d(out_channels[0])
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[0], out_channels=self.out_channels[1], kernel_size=self.kernel_sizes[1], stride=self.strides[1]),
            #nn.BatchNorm2d(out_channels[1])
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[1], out_channels=self.out_channels[2], kernel_size=self.kernel_sizes[2], stride=self.strides[2]),
            #nn.BatchNorm2d(out_channels[2])
            nn.ReLU()
        )

        self.finishFC = nn.Sequential(
            nn.Linear(1617,32),
            nn.Linear(32,32),
            nn.Linear(32,1)
        )


    def forward(self, image, timestep):
        timestep=timestep.unsqueeze(1)
        convolution_output = self.convolution(image)
        time_tiled = torch.repeat_interleave(timestep,49,dim=0).reshape((timestep.shape[0],1,7,7))
        state_output = torch.cat((time_tiled,convolution_output),dim=1)

        state_action = torch.add(state_output,action_output)
        state_action_flat = torch.flatten(state_action,start_dim=1)
        q = self.finishFC(state_action_flat)

        return q







class ValueNetwork(nn.Module):
    def __init__(self,h, w, num_actions):
        super(ValueNetwork,self).__init__()
        self.in_channels = 3
        self.out_channels = [32,32,32]
        self.kernel_sizes = [4,3,3]
        self.strides = [2,2,2]
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels[0], kernel_size=self.kernel_sizes[0], stride=self.strides[0]),
            #nn.BatchNorm2d(out_channels[0])
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[0], out_channels=self.out_channels[1], kernel_size=self.kernel_sizes[1], stride=self.strides[1]),
            #nn.BatchNorm2d(out_channels[1])
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[1], out_channels=self.out_channels[2], kernel_size=self.kernel_sizes[2], stride=self.strides[2]),
            #nn.BatchNorm2d(out_channels[2])
            nn.ReLU()
        )

        self.finishFC = nn.Sequential(
            nn.Linear(1617,32),
            nn.Linear(32,32),
            nn.Linear(32,1)
        )


    def forward(self, image, timestep):
        timestep=timestep.unsqueeze(1)
        convolution_output = self.convolution(image)
        time_tiled = torch.repeat_interleave(timestep,49,dim=0).reshape((timestep.shape[0],1,7,7))
        state_output = torch.cat((time_tiled,convolution_output),dim=1)

        state_action = torch.add(state_output,action_output)
        state_action_flat = torch.flatten(state_action,start_dim=1)
        q = self.finishFC(state_action_flat)

        return q






###########
## Here taken from github: https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/sac.py#L195
class ValueNetwork____(nn.Module):
    def __init__(self, state_dim, hidden_dim, activation=F.relu, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.activation = activation

    def forward(self, state):
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork____(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, activation=F.relu, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.activation = activation

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork____(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, activation=F.relu, init_w=3e-3, log_std_min=-20,
                 log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = 10.
        self.num_actions = num_actions
        self.activation = activation

    def forward(self, state):
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))

        mean = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std