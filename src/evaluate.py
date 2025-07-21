import gymnasium as gym
import torch
import yaml # Add yaml import
from pathlib import Path

# Keep make_agent, but we won't need load_configs from the train script
from src.agents.dqn import DQNAgent # Or a factory if you prefer

def watch_agent(eval_cfg:dict, agent_cfg: dict):
    """Loads a trained agent and runs it in an environment with rendering."""
    
    # INFO: The environments are not scalable
    # Create the environment
    env = gym.make(eval_cfg['eval']['env_name'], render_mode=eval_cfg['eval']['render_mode'])

    # INFO: Here we have to write the different agents
    # Create the agent instance using the original config
    if agent_cfg["type"] == "dqn":
        agent = DQNAgent(env.observation_space, env.action_space, agent_cfg)
    else:
        raise ValueError("Unknow agent type in agent_config.yaml")
    
    # Load the save model weights
    agent.load(eval_cfg['eval']['model_path'])

    print(f"Watching agent from {eval_cfg['eval']['model_path']}")
    
    # Run the evaluation Loop
    for episode in range(eval_cfg['eval']['num_episodes']):
        state, info = env.reset()
        terminated, truncated = False, False
        total_reward = 0

        while not (terminated or truncated):
            # INFO: Greedy is necessary to eval the real model
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward  ={total_reward}")
    
    env.close()

if __name__ == "__main__":
    # Load the evaluation-specific configuration
    with open("configs/evaluate_config.yaml", 'r') as f:
        eval_config = yaml.safe_load(f)

    # Load the agent configuration to ensure model architecture matches
    with open("configs/agent_config.yaml", 'r') as f:
        agent_config = yaml.safe_load(f)

    watch_agent(eval_config, agent_config)