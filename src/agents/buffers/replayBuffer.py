from abc import ABC, abstractmethod
from typing import NamedTuple, List
from src.utils.buffer import BaseBuffer, Transition
import random
import torch

# TODO: MAKE IT FULLY VECTORIZED
class ReplayBuffer(BaseBuffer):
    """Buffer used for DQN, store the Transition in a Data Structure of a fixed size, this is used to Train the RLAgent"""
    
    def __init__(self, capacity: int, device: torch.device):
        """Create the Data Structure of a fixed capacity (self.storage = List[Transition])"""
        self.capacity = capacity
        self.storage: List[Transition] = []
        self.pos = 0
        self.device = device

    def add(self, transition: Transition):
        """Logic to save the transition in the Buffer.
        It properly overwrites the oldest experiences once capacity is reached."""
        # If there is space, save the transition in the next pos
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            # If there is no space, change the data in the "next position"
            self.storage[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Take a random sample of the saved transitions"""
        batch = random.sample(self.storage, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            'states' : torch.stack([torch.from_numpy(s) for s in states]).float().to(self.device),
            'actions' : torch.tensor(actions, dtype=torch.long, device=self.device),
            'rewards' : torch.tensor(rewards, dtype=torch.float, device=self.device),
            'next_states': torch.stack([torch.from_numpy(s) for s in next_states]).float().to(self.device),
            'dones': torch.tensor(dones, dtype=torch.float, device=self.device),
        }

    def __len__(self):
        """Check lenght of storage"""
        return len(self.storage)

    def clear(self) -> None:
        """Removes all transitions from the buffer."""
        self.storage = []
        self.pos = 0
        return