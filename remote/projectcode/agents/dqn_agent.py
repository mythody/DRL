import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from PIL import Image
from projectcode.infrastructure import pytorch_util as ptu
from projectcode.infrastructure.utils import ReplayMemory
from torch import distributions




class DQNAgent(object):
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
        self.batch_size = agent_params['batch_size']

        #self.critic = DQNCritic(agent_params, self.optimizer_spec)
        #self.actor = ArgMaxPolicy(self.criticc)
        img_size = self.params['obs_size']
        LEARNING_RATE = self.params['LEARNING_RATE'] #1e-4 # argsis

        self.preprocess = T.Compose([T.ToPILImage(),
                            T.Grayscale(num_output_channels=1),
                            T.Resize(img_size, interpolation=Image.CUBIC),
                            T.ToTensor()])

        # Get screen size so that we can initialize layers correctly based on shape
        # returned from pybullet (48, 48, 3).
        init_screen = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape

        # Get number of actions from gym action space
        if(self.env._isDiscrete):
            self.n_actions = env.action_space.n
            print("n_actions: ", env.action_space.n)
            print(type(env.action_space.n))
        else:
            self.n_actions = 5 #env.action_space.high.size
            print("n_actions: ", env.action_space.high.size)
            print(type(env.action_space.high.size))

        self.policy_net = ptu.DQN(screen_height, screen_width, self.n_actions).to(self.device)
        self.target_net = ptu.DQN(screen_height, screen_width, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.logstd = torch.nn.Parameter(torch.zeros(self.n_actions, dtype=torch.float32, device=ptu.device)).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(self.params['MEMORY_CAPACITY'])

        self.eps_threshold = 0

        # uniform_distribution = (r1 - r2) * torch.rand(a, b) + r2
        # see https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch




    def get_screen(self):
        global stacked_screens
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env._get_observation(self.params['obs_type']).transpose((2, 0, 1))
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

        # Add joints_state or t_steps if wanted


        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return screen #self.preprocess(screen).unsqueeze(0).to(self.device)

    eps_threshold = 0

    def select_action(self, state, i_episode):
        global steps_done
        global eps_threshold
        sample = random.random()
        self.eps_threshold = max(self.params['EPS_END'], self.params['EPS_START'] - i_episode / self.params['EPS_DECAY_LAST_FRAME'])
        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.params['BATCH_SIZE']:
            return
        transitions = self.memory.sample(self.params['BATCH_SIZE'])
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.memory.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.params['BATCH_SIZE'], device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.params['GAMMA']) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

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
