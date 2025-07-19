from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List
import random
import torch
from tqdm import trange
from src.utils.config import load_configs, make_env, make_agent
import gymnasium as gym
import numpy as np



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

    # Reemplaza el método _run_single completo en tu archivo train.py

    def _run_single(self, env_name: str, seed: int) -> ExperimentResult:
        # Seed RNGs for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed) 

        # Create environment and agent
        env = make_env(self.cfg)
        agent = make_agent(env, self.cfg)

        # Training loop
        returns: List[float] = []
        
        # --- INICIO DE CAMBIOS ---

        # 1. El primer reset establece la semilla para toda la secuencia de episodios
        state, info = env.reset(seed=seed)

        for episode in trange(1, self.episodes + 1, desc=f"{env_name} (Seed {seed})"):
            # La variable 'state' ya está inicializada para el primer episodio.
            # Para los siguientes, se obtiene del reset al final del bucle anterior.
            
            terminated = False
            truncated = False
            ep_return = 0.0
            i = 0

            while not (terminated or truncated):
                action = agent.select_action(state)

                # 2. La llamada a step ahora devuelve 5 valores
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # La señal de 'done' es True si el episodio terminó por cualquier razón
                done = terminated or truncated

                agent.observe(state, action, reward, next_state, done)
                if i % self.cfg['agent']['train_frequency'] == 0:
                    agent.update()
                
                state = next_state
                ep_return += reward
                i += 1
            
            returns.append(ep_return)

            # 3. Hacemos reset para el siguiente episodio (si no es el último)
            if episode < self.episodes:
                state, info = env.reset()

        # --- FIN DE CAMBIOS ---

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
    print(all_results)
    # Optionally: save master summary or print results
