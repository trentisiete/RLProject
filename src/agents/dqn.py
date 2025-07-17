from sre_parse import State
from .base import BaseAgent
import torch
import torch.nn as nn
import torch.nn.functional as F # No need to instantiate
from src.agents.buffers import replayBuffer
from.src.utils.buffer import Transition
import numpy as np
import math
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
        
        # Calculate current epsilon using exponential decay
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
              math.exp(-1. * self.steps_done / self.epsilon_decay_steps)
        
        self.steps_done += 1

        if random.random() < epsilon:
            # Random Action from random Agent
            return self.act_space.sample()
        else:
            with torch.no_grad():

                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                # Torch.from_numpy converts to torch array
                # Unsqueeze convert the tensor from size N to (1, N) like a batch

                # Get the values from the Q_network passing the state_tensor
                # No need to directly write the "forward" method explicitly
                q_values = self.policy_net(state_tensor)

                # return the best action
                return torch.argmax(q_values).item()
        

    def observe(self, state, action, reward:float, next_state, done:bool):
        """Store a transition in the replay buffer."""
        # Packaging the data into a "Transition" object
        transition = Transition(state, action, reward, next_state, done)
        
        # Adding the "transition" object to Buffer
        self.buffer.add(transition)


    def update(self):
        # Sample a random batch from the buffer
        if len(self.buffer) < self.batch_size:
            return
        else:
            experiences = self.buffer.sample(self.batch_size)
        
        # Calculate the target Q-Values for that batch using the frozen target Network.
        


