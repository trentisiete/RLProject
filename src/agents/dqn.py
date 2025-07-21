from nt import access
from sre_parse import State
from .base import BaseAgent
import torch
import torch.nn as nn
import torch.nn.functional as F # No need to instantiate
from src.agents.buffers import replayBuffer
from src.utils.buffer import Transition
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

    def __init__(self, obs_space, act_space, agent_config:dict):
        super().__init__(obs_space, act_space, agent_config)

        # Writing hyperparams
        self.gamma = agent_config["hyperparams"]['gamma']
        self.lr = agent_config["hyperparams"]['lr']
        self.batch_size =  agent_config["hyperparams"]['batch_size']
        self.replay_buffer_size =  agent_config["hyperparams"]['replay_buffer_size']
        self.min_replay_buffer_size = agent_config['hyperparams']['min_replay_buffer_size']
        self.target_update_freq = agent_config["hyperparams"]['target_update_freq']
        self.epsilon_start = agent_config["hyperparams"]['epsilon_start']
        self.epsilon_end = agent_config["hyperparams"]['epsilon_end']
        self.epsilon_decay_steps = agent_config["hyperparams"]['epsilon_decay_steps']
        self.loss_fn = nn.SmoothL1Loss()

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


    def select_action(self, state, greedy:bool=False):
        """Selects an action using an epsilon-greedy policy or purely greedy policy."""
        
        # if it's in greedy mode, skip exploration mode
        if greedy:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()

        # The existing implementation remains the same
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
        """ y=r + γ * max_a(Q(s', a'))"""
        if len(self.buffer) < self.min_replay_buffer_size:
            return
        
        # Sample a random batch from the buffer
        experiences = self.buffer.sample(self.batch_size)
        states = experiences['states']
        actions = experiences['actions']
        rewards = experiences['rewards']
        next_states = experiences['next_states']
        dones = experiences["dones"]

        with torch.no_grad():
            # Get max Q-value for next states from the target network
            next_q_values = self.target_net(next_states).max(1)[0]
            
            # --- INICIO DE LA CORRECCIÓN 1 ---
            # Set Q-value to 0 for terminal states.
            # Convert 'dones' to a boolean tensor for indexing.
            next_q_values[dones.bool()] = 0.0
            # --- FIN DE LA CORRECCIÓN 1 ---

            # Compute the target Q-value
            target_q_values = rewards + (self.gamma * next_q_values)

        # Get predicted Q-values from the policy network
        # We need to select the Q-values for the actions that were actually taken
        predicted_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute loss and perform optimization
        loss = self.loss_fn(predicted_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        
        # --- INICIO DE LA CORRECCIÓN 2 ---
        # Call backward() on the loss tensor, not the loss function
        loss.backward()
        # --- FIN DE LA CORRECCIÓN 2 ---

        self.optimizer.step()
        
        # Update the target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # No need to return anything, but returning loss can be useful for logging
        return loss.item()
    
    def load(self, path: str):
        """Loads the policy network's weights from a file."""
        # Load the state dict from the file, mapping to the agent's current device
        state_dict = torch.load(path, map_location=self.device)
        
        # Apply the loaded weights to the policy network
        self.policy_net.load_state_dict(state_dict)
        
        # Synchronize the target network with the loaded policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Set both networks to evaluation mode
        self.policy_net.eval()
        self.target_net.eval()

    def save(self, path: str):
        """Saves the policy network's weights to a file."""
        print(f"Saving model state to {path}...")
        torch.save(self.policy_net.state_dict(), path)