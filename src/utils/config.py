from pathlib import Path
import yaml
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import NormalizeObservation
from src.agents.dqn import DQNAgent
# from src.agents.ppo import PPOAgent
from src.agents.random_agent import RandomAgent 


AGENT_CLASSES = {
    'dqn':    DQNAgent,
    # 'ppo':    PPOAgent,
    'random': RandomAgent,
}

def load_yaml(path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_configs(config_dir: Path):
    agent_cfg      = load_yaml(config_dir / 'agent_config.yaml')
    env_cfg        = load_yaml(config_dir / 'env_config.yaml')
    experiment_cfg = load_yaml(config_dir / 'experiment_config.yaml')

    # Combine into one dict
    cfg = {
        'agent':      agent_cfg['agent'],
        'env':        env_cfg['env'],
        'experiment': experiment_cfg['experiment']
    }
    return cfg

def make_agent(env: Env, cfg:dict):
    agent_type = cfg['agent']['type'] # dqn, ppo,...
    cls = AGENT_CLASSES[agent_type] # class
    return cls(env.observation_space, env.action_space, cfg['agent'])

def make_env(cfg:dict):
    name = cfg['env']['name']
    env = gym.make(name) # TODO: THIS IS NOT GENERALIZED

    if cfg['env']['wrappers'].get('normalize_obs'):
        env = NormalizeObservation(env)
    # etc.
    return env
