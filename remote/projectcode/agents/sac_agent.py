import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
from torch.distributions import Normal


from PIL import Image
from projectcode.infrastructure import pytorch_util as ptu
from projectcode.infrastructure.utils import ReplayMemory
from torch import distributions
import matplotlib.pyplot as plt
from projectcode.infrastructure.sac_utils import ValueNetwork, SoftQNetwork, PolicyNetwork



class SACAgent(object):
    def __init__(self, env, agent_params):

        self.agent_params = agent_params

        if torch.cuda.is_available() and self.agent_params['use_gpu']:
            self.device = torch.device("cuda:" + str(self.agent_params['gpu_id']))
            print("Using GPU id {}".format(self.agent_params['gpu_id']))
        else:
            self.device = torch.device("cpu")
            print("GPU not detected. Defaulting to CPU.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env

        self.params = agent_params
        self.batch_size = agent_params['BATCH_SIZE']

        #self.critic = DQNCritic(agent_params, self.optimizer_spec)
        #self.actor = ArgMaxPolicy(self.criticc)
        img_size = self.params['obs_size']
        LEARNING_RATE = self.params['LEARNING_RATE'] #1e-4 # argsis


        self.preprocess = T.Compose([T.ToPILImage(),
                            T.Resize(img_size, interpolation=Image.CUBIC),
                            T.ToTensor()])

        if(self.params['obs_type']=='BW'):
            self.preprocess = T.Compose([T.ToPILImage(),
                                T.Grayscale(num_output_channels=1),
                                T.Resize(img_size, interpolation=Image.CUBIC),
                                T.ToTensor()])

         #Get screen size so that we can initialize layers correctly based on shape
         #returned from pybullet (48, 48, 3).
        init_screen = self.get_screen()
        _, n_channels, screen_height, screen_width = init_screen.shape

        # Get number of actions from gym action space
        if(self.env._isDiscrete):
            self.n_actions = env.action_space.n
            print("n_actions: ", env.action_space.n)
            print(type(env.action_space.n))
        else:
            self.n_actions = 4 #env.action_space.high.size #-1 because we do not control gripper-closeness
            print("n_actions: ", env.action_space.high.size)
            print(type(env.action_space.high.size))


        #### Define networks, loss, lrs, optimizers for ValueNet, soft-q-net-1, soft-q-net-1, policy-net
        self.value_net = ValueNetwork(n_channels, screen_height, screen_width).to(self.device)
        self.target_value_net = ValueNetwork(n_channels, screen_height, screen_width).to(self.device)

        self.soft_q_net1 = SoftQNetwork(n_channels, screen_height, screen_width, self.n_actions).to(self.device)
        self.soft_q_net2 = SoftQNetwork(n_channels, screen_height, screen_width, self.n_actions).to(self.device)
        self.policy_net = PolicyNetwork(n_channels, screen_height, screen_width, self.n_actions, self.device).to(self.device)

        print('(Target) Value Network: ', self.value_net)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        value_lr = 3e-4
        soft_q_lr = 3e-4
        policy_lr = 3e-4

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.memory = ReplayMemory(self.params['MEMORY_CAPACITY'], device=self.device)
        print("Initial_memory: ", self.params['initial_memory'])
        if(self.params['initial_memory']):
            # Load memory data
            print("Initial Memmory Path: ", self.params['initial_memory_path'])
            print("cam view: ", self.params['cam_view_option'])
            self.memory.load_memory(memory_path=self.params['initial_memory_path'])

        self.eps_threshold = 0

        self.action_range = 1.

        '''
        self.policy_net = ptu.DQN(n_channels, screen_height, screen_width, self.n_actions).to(self.device)
        self.target_net = ptu.DQN(n_channels, screen_height, screen_width, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.logstd = torch.nn.Parameter(torch.zeros(self.n_actions, dtype=torch.float32, device=ptu.device)).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(self.params['MEMORY_CAPACITY'])

        self.eps_threshold = 0

        # uniform_distribution = (r1 - r2) * torch.rand(a, b) + r2
        # see https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        '''

    def get_screen(self):
        global stacked_screens
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).

        #screen = self.env._get_observation(self.params['obs_type']).transpose((2, 0, 1))
        if(isinstance(self.params['obs_type'], list)):
            screen = self.env._get_observation(self.params['obs_type'][0]).transpose((2, 0, 1))
            for i in range(1, len(self.params['obs_type']) ):
                screen_i = self.env._get_observation(self.params['obs_type'][i]).transpose((2, 0, 1))
                screen = np.concatenate((screen, screen_i))
        else:
            screen = self.env._get_observation(self.params['obs_type']).transpose((2, 0, 1))
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)

        screen = np.ascontiguousarray(screen, dtype=np.float32)
        screen = screen / 255

        # Add joints_state or t_steps if wanted
        # t-steps: t_step/max_step conc. to screen.shape(dim 1,2)
        if(self.params['add_obs']=='t-steps'):
            curr_t = self.env._env_step
            maxtime = self.env._maxSteps
            curr_t_d_maxtime = curr_t / maxtime
            t_step_tensor = curr_t_d_maxtime * np.ones((1, screen.shape[1], screen.shape[2]), dtype=np.float32)
            screen = np.concatenate((screen, t_step_tensor))


        screen = torch.from_numpy(screen)
        preprocessed_screen = self.preprocess(screen).unsqueeze(0).to(self.device)
        # Resize, and add a batch dimension (BCHW)



        return preprocessed_screen

    def display_screen(self, state):
        #plt.imshow(state.cpu().numpy().transpose((0, 1, 2, 3)).squeeze(0).squeeze(0))
        sqeezed_and_transposed_img = state.cpu().numpy().squeeze(0).transpose((1,2,0))
        if(sqeezed_and_transposed_img.shape[2]==1):
            sqeezed_and_transposed_img = sqeezed_and_transposed_img.squeeze(2)
        plt.imshow(sqeezed_and_transposed_img)
        plt.show()

    eps_threshold = 0

    def select_action(self, state, i_episode, t):

        global steps_done
        global eps_threshold
        sample = random.random()
        self.eps_threshold = max(self.params['EPS_END'], self.params['EPS_START'] - i_episode / self.params['EPS_DECAY_LAST_FRAME'])

        deterministic = False
        if sample > self.eps_threshold:
            deterministic = True


        #state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        t_tensor = torch.FloatTensor([t]).to(self.device)
        mean, log_std = self.policy_net(state, t_tensor)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(self.device)
        action = self.action_range * torch.tanh(mean + std * z)
        action = torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]

        return action


    def optimize_model(self, soft_tau=1e-2):

        batch_size = self.params['BATCH_SIZE']

        alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)

        transitions = self.memory.sample(self.params['BATCH_SIZE'])
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.memory.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        time_step = torch.cat(batch.timestep)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        #state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.params['BATCH_SIZE'], device=self.device)

        masked_timestep = torch.masked_select(time_step, non_final_mask)
        try:
            next_state_values[non_final_mask] = self.target_value_net(non_final_next_states, masked_timestep).max(1)[0].detach()
        except:
            print("same errrrrror again ")
        # Compute the expected Q values
        target_q_value = (next_state_values * self.params['GAMMA']) + reward_batch # expected_state_action_values =

        #state = torch.FloatTensor(state).to(device)
        #next_state = torch.FloatTensor(next_state).to(device)
        #action = torch.FloatTensor(action).to(device)
        #reward = torch.FloatTensor(reward).unsqueeze(1).to(
        #    device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        #done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state_batch, action_batch, time_step)
        predicted_q_value2 = self.soft_q_net2(state_batch, action_batch, time_step)
        predicted_value = self.value_net(state_batch, time_step)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state_batch, time_step)

        #reward_batch = reward_scale * (reward_batch - reward_batch.mean(dim=0)) / reward_batch.std(dim=0)  # normalize with batch mean and std

        # Training Q Function
        #target_value = self.target_value_net(next_state)
        #target_q_value = reward_batch + (1 - done) * gamma * target_value  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1,
                                          target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Value Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state_batch, new_action, time_step), self.soft_q_net2(state_batch, new_action, time_step))
        target_value_func = predicted_new_q_value - alpha * log_prob  # for stochastic training, it equals to expectation over action
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Training Policy Function
        ''' implementation 1 '''
        self.policy_loss = (alpha * log_prob - predicted_new_q_value).mean()
        ''' implementation 2 '''
        # policy_loss = (alpha * log_prob - soft_q_net1(state, new_action)).mean()  # Openai Spinning Up implementation
        ''' implementation 3 '''
        # policy_loss = (alpha * log_prob - (predicted_new_q_value - predicted_value.detach())).mean() # max Advantage instead of Q to prevent the Q-value drifted high

        ''' implementation 4 '''  # version of github/higgsfield
        # log_prob_target=predicted_new_q_value - predicted_value
        # policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        # mean_lambda=1e-3
        # std_lambda=1e-3
        # mean_loss = mean_lambda * mean.pow(2).mean()
        # std_loss = std_lambda * log_std.pow(2).mean()
        # policy_loss += mean_loss + std_loss

        self.policy_optimizer.zero_grad()
        self.policy_loss.backward()
        self.policy_optimizer.step()

        # print('value_loss: ', value_loss)
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return q_value_loss1











#############################################################




    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of projectcode. the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # TODO store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer`
            # in dqn_utils.py
        self.replay_buffer_idx = TODO

        eps = self.exploration.value(self.t)

        # TODO use epsilon greedy exploration when selecting action
        perform_random_action = TODO
        if perform_random_action:
            # HINT: take random action 
                # with probability eps (see np.random.random())
                # OR if your current step number (see self.t) is less that self.learning_starts
            action = TODO
        else:
            # HINT: Your actor will take in multiple previous observations ("frames") in order
                # to deal with the partial observability of the environment. Get the most recent 
                # `frame_history_len` observations using functionality from the replay buffer,
                # and then use those observations as input to your actor. 
            action = TODO
        
        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        TODO

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        TODO

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        TODO

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # TODO fill in the call to the update function using the appropriate tensors
            log = self.critic.update(
                TODO
            )

            # TODO update the target network periodically 
            # HINT: your critic already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                TODO

            self.num_param_updates += 1

        self.t += 1
        return log
