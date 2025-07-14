from abc import ABC, abstractmethod
from src.train import ExperimentResult

class BaseCallback(ABC):
    @abstractmethod
    def on_experiment_end(self, result: ExperimentResult) -> None:
        """Called once after each (env, seed) completes."""

    @abstractmethod
    def on_episode_end(self, episode: int, info: dict) -> None:
        """Optional: Called after every episode (for live plotting)."""


## Implement concrete callbacks:

# CsvLogger(BaseCallback) -> Directly replace _log from train
# TensorBoardLogger(BaseCallback)
# WandBLogger(BaseCallback)