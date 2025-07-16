from .base import BaseAgent
import torch
import torch.nn as nn
import torch.nn.functional as F # No need to instantiate
from src.agents.buffers import replayBuffer
import numpy as np
import random

class MlpQNetwork(nn.Module):
    """Simple MLP for approximating Q-values in vector-based environments."""
    def __init__(self, obs_size: int, action_size: int):
        """
        Args:
            obs_size (int): Dimension of the observation space.
            action_size (int): Number of discrete actions.
        """
        super().__init__()
        self.layer1 = nn.Linear(obs_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # No activation on the output layer for Q-values
        q_values = self.output_layer(x)
        return q_values


class DQNAgent(BaseAgent):
    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)
        agent_config = config['agent']

        # Writing hyperparams
        self.gamma = agent_config['gamma']
        self.lr = agent_config['lr']
        self.batch_size =  agent_config['batch_size']
        self.replay_buffer_size =  agent_config['replay_buffer_size']
        self.target_update_freq = agent_config['target_update_freq']
        self.epsilon_start = agent_config['epsilon_start']
        self.epsilon_end = agent_config['epsilon_end']
        self.epsilon_decay_steps = agent_config['epsilon_decay_steps']

        cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda_available else "cpu")

        network_class_name = agent_config['network']
        if network_class_name == 'MlpQNetwork':
            # Creating both Policy network and Target Network
            self.policy_net = MlpQNetwork(obs_size=obs_space.shape[0], action_size=act_space.n)
            self.target_net = MlpQNetwork(obs_size=obs_space.shape[0], action_size=act_space.n)

            # Moving both Networks to Device
            self.policy_net = self.policy_net.to(self.device)
            self.target_net = self.target_net.to(self.device)

        elif network_class_name == 'CnnQNetwork':
            # TODO: THIS COULD BE A POSSIBLE ERROR
            raise NotImplementedError("CNN Q Network not implemented yet")
        # build networks, replay buffer, hyperparams from config['agent']

        # Synchronizing Weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Setting target Network to evaluation mode
        self.target_net.eval()

        # Setting the optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), self.lr)

        # Instantiating the replay Buffer #TODO: CHECK REPLAY BUFFER IF MLP IS NOT THE NET
        self.buffer = replayBuffer.ReplayBuffer(self.replay_buffer_size, self.device)

        # Counter to track the episode
        self.steps_done = 0



    def select_action(self, state):
        # ε-greedy policy…
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1 * self.steps_done/ self.epsilon_decay_steps)

        random_number = random.random()
        if random_number < epsilon:
            self.act_space.sample()
        else:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

            output = self.policy_net.forward(state)


        pass

    def observe(self, s, a, r, s2, done):
        # push to replay…
        pass

    def update(self):
        # sample minibatch, gradient step…
        return {"loss": loss_value, "epsilon": self.epsilon}
