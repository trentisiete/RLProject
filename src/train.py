from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List

import gym
import numpy as np

from src.utils.config import load_configs, make_env, make_agent


@dataclass
class ExperimentResult:
    """
    Data class to store results of a single experiment run.

    Attributes:
        env_name (str): Name of the environment used.
        seed (int): Random seed for reproducibility.
        total_episodes (int): Number of episodes run.
        avg_return (float): Average return over all episodes.
        metrics (Dict[str, Any]): Additional per-episode metrics (e.g., returns list).
    """
    env_name: str
    seed: int
    total_episodes: int
    avg_return: float
    metrics: Dict[str, Any]


class ExperimentRunner:
    """
    Manages and executes a suite of reinforcement learning experiments defined in configuration.

    This runner handles environment and agent creation, seeding, execution loops,
    logging of results, and optional callback hooks.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        env_factory: Callable[[str, Dict[str, Any]], gym.Env] = make_env,
        agent_factory: Callable[[gym.Env, Dict[str, Any]], Any] = make_agent,
        callbacks: List[Callable[[ExperimentResult], None]] = None,
    ) -> None:
        """
        Initialize the ExperimentRunner.

        Args:
            cfg (Dict[str, Any]): Merged configuration dict from YAML files.
            env_factory (Callable): Function to instantiate environments.
            agent_factory (Callable): Function to instantiate agents.
            callbacks (List[Callable]): Optional list of callbacks invoked after each run.
        """
        self.cfg = cfg
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.envs = cfg["experiment"]["envs"]
        self.seeds = cfg["experiment"]["seeds"]
        self.episodes = cfg["experiment"].get("episodes_per_env", 100)
        self.results: List[ExperimentResult] = []
        self.callbacks = callbacks or []

    def run(self) -> List[ExperimentResult]:
        """
        Execute all combinations of environments and seeds as defined in the configuration.

        Returns:
            List[ExperimentResult]: Collected results for each (env, seed) pair.
        """
        self._setup()
        for env_name in self.envs:
            for seed in self.seeds:
                result = self._run_single(env_name, seed)
                self.results.append(result)
                self._log(result)
                self._fire_callbacks(result)
        self._teardown()
        return self.results

    def _setup(self) -> None:
        """
        Prepare any necessary resources before running experiments.

        Creates the output directory if it does not exist.
        """
        output_dir = Path(self.cfg["experiment"].get("output_dir", "results/"))
        output_dir.mkdir(exist_ok=True, parents=True)

    def _run_single(self, env_name: str, seed: int) -> ExperimentResult:
        """
        Run a single experiment: instantiate environment and agent, train for N episodes.

        Args:
            env_name (str): Identifier for the environment.
            seed (int): Random seed for reproducibility.

        Returns:
            ExperimentResult: Summary of performance and metrics.
        """
        # Seed RNGs for reproducibility
        np.random.seed(seed)
        gym.logger.set_level(40)

        # Create environment and agent
        env = self.env_factory(env_name, self.cfg)
        env.seed(seed)
        agent = self.agent_factory(env, self.cfg)

        # Training loop
        returns: List[float] = []
        for episode in range(1, self.episodes + 1):
            state = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.observe(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                ep_return += reward
            returns.append(ep_return)

        avg_return = float(sum(returns) / len(returns))
        return ExperimentResult(
            env_name=env_name,
            seed=seed,
            total_episodes=self.episodes,
            avg_return=avg_return,
            metrics={"returns": returns},
        )

    def _log(self, result: ExperimentResult) -> None:
        """
        Persist the results of a single experiment to disk.

        Writes a CSV file with per-episode returns in the output directory.
        """
        out_dir = Path(self.cfg["experiment"].get("output_dir", "results/"))
        csv_path = out_dir / f"{result.env_name}_seed{result.seed}.csv"
        with csv_path.open("w") as f:
            f.write("episode,return\n")
            for idx, ret in enumerate(result.metrics["returns"], start=1):
                f.write(f"{idx},{ret}\n")

    def _fire_callbacks(self, result: ExperimentResult) -> None:
        """
        Invoke registered callbacks after each experiment run.

        Args:
            result (ExperimentResult): The result of the latest experiment.
        """
        for callback in self.callbacks:
            callback(result)

    def _teardown(self) -> None:
        """
        Clean up any resources after all experiments have completed.

        Currently a no-op, but useful for closing files or aggregating summaries.
        """
        pass


if __name__ == "__main__":
    config_dir = Path("configs/")
    cfg = load_configs(config_dir)
    runner = ExperimentRunner(cfg)
    all_results = runner.run()
    # Optionally: save master summary or print results
