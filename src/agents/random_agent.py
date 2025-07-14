import abc
from typing import Any, Dict
import gym
from src.agents.base import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self,
                observation_space: gym.spaces.Space,
                action_space: gym.spaces.Space,
                config: Dict[str, Any]):
        super().__init__(observation_space, action_space, config)
    
    def select_action(self, state: Any) -> Any:
        """Select a Random action of the action space"""
        return self.act_space.sample()
    
    def observe(self, state, action, reward, next_state, done):
        """Since this is a RandomAgent, we don't need to save Experiences"""
        pass
    def update(self):
        """RandomAgent does not need to update nothing because it doesn't learn"""
        pass
    def save(self, path: str):
        """RandomAgent does not need to save nothing"""
        pass
    def load(self, path: str):
        """RandomAgent does not need to load nothing"""
        pass