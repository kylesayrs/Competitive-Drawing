from stable_baselines3 import PPO
from gym import Env

from competitive_drawing.train.reinforcement_learning.config import AgentConfig


def load_agent(agent_config: AgentConfig, environment: Env):
    agent = PPO(
        agent_config.policy,
        environment,
        policy_kwargs=agent_config.policy_kwargs,
        learning_rate=agent_config.learning_rate,
        n_steps=agent_config.n_steps,
        batch_size=agent_config.batch_size,
        n_epochs=agent_config.n_epochs,
        gamma=agent_config.gamma,
        gae_lambda=agent_config.gae_lambda,
        clip_range=agent_config.clip_range,
        verbose=agent_config.verbosity,
        device=agent_config.device,
    )

    # TODO: load from checkpoint

    return agent
