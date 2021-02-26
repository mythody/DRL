from typing import Union

import torch
from torch import nn

import torch.nn.functional as F


######
## LEVINE NETWORK: Q(s,a) --> Scalar-Reward, for continuous action-space
#################
class qNetworkStateAndAction(nn.Module):
    def __init__(self,h, w, num_actions):
        super(qNetworkStateAndAction,self).__init__()
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




####
## DQN-on-policy: Q(s) -> V(a) network. for discrete actions. one convolution on img space.
################
class DQN(nn.Module):
    def __init__(self, c, h, w, outputs, STACK_SIZE=1):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(STACK_SIZE*c, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
        linear_input_size = convw * convh * 64
        self.linear = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs) #, outputs

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        return self.head(x)

'''
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from Piazza
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
        return action_distribution

'''










Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
        init_method=None,
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        curr_layer = nn.Linear(in_size, size)
        if init_method is not None:
            curr_layer.apply(init_method)
        layers.append(curr_layer)
        layers.append(activation)
        in_size = size

    last_layer = nn.Linear(in_size, output_size)
    if init_method is not None:
        last_layer.apply(init_method)

    layers.append(last_layer)
    layers.append(output_activation)
        
    return nn.Sequential(*layers)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
