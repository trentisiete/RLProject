from abc import ABC, abstractmethod
from typing import NamedTuple, List, Any
import random

class Transition(NamedTuple):
    """Agent experiences at each time"""
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool

class BaseBuffer(ABC):
    @abstractmethod
    def add(self, transition: Transition) -> None: ...
    @abstractmethod
    def sample(self, batch_size: int) -> List[Transition]: ...
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def clear(self) -> None:
        """Removes all transitions from the buffer."""