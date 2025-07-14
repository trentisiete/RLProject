import abc
from typing import Any, Dict
import gym

class BaseAgent(abc.ABC):
    def __init__(self, observation_space: gym.spaces.Space, # All spaces inherit from this Space class
                       action_space: gym.spaces.Space,
                       config: Dict[str, Any]):
        self.obs_space  = observation_space
        self.act_space  = action_space
        self.config     = config

    @abc.abstractmethod
    def select_action(self, state: Any) -> Any:
        """Given a state, return an action (possibly stochastic)."""
        raise NotImplementedError

    @abc.abstractmethod
    def observe(self, state: Any, action: Any, reward: float,
                next_state: Any, done: bool):
        """Store or process a transition (for learning). It's like adding an Experience"""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self) -> Dict[str, float]:
        """Periodically called to improve policy; return logging metrics. It's like Learn"""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: str):
        """Serialize networks/parameters to disk."""
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: str):
        """Load networks/parameters from disk."""
        raise NotImplementedError