from typing import List

import torch
from stable_baselines3 import DDPG, make_vec_env, VecEnv

from competitive_drawing.train.rl_gan.config import (
    TrainingConfig, AgentConfig, CriticConfig, EnvironmentConfig
)
from competitive_drawing.train.rl_gan.solo_environment import SoloEnvironment
from competitive_drawing.train.rl_gan.Critic import Critic


def train_models(
    training_config: TrainingConfig,
    agent_config: AgentConfig,
    critic_config: CriticConfig,
    environment_config: EnvironmentConfig,
):
    environment = make_vec_env(
        SoloEnvironment,
        env_kwargs={"environment_config": environment_config},
        n_envs=1
    )

    agent_one = DDPG(**agent_config.dict())
    agent_two = DDPG(**agent_config.dict())
    critic = Critic(**critic_config.dict())

    rollout_callback = None
    agent_one._setup_learn(0)
    agent_two._setup_learn(0)

    alternating_images = None
    one_starts_first = True
    alternating_steps_left = environment_config.max_num_turns

    for episode_index in training_config.num_episodes:
        # this code needs jesus
        if episode_index % environment_config.max_num_turns:
            alternating_images = [
                torch.zeros(
                    (environment_config.image_size, environment_config.image_size),
                    dtype=torch.float32,
                    device=environment_config.device
                )
                for _ in environment.n_envs
            ]
            one_starts_first = not one_starts_first

        if one_starts_first:
            if episode_index % 2 == 0:
                agent = agent_one
            else:
                agent = agent_two
        else:
            if episode_index % 2 == 0:
                agent = agent_two
            else:
                agent = agent_one

        environment.reset(alternating_images, alternating_steps_left)
        alternating_steps_left -= 1

        agent.collect_rollouts(
            environment,
            train_freq=(1, "episode"),  # trick to only rollout one episode
            action_noise=agent.action_noise,
            callback=rollout_callback,
            learning_starts=agent.learning_starts,
            replay_buffer=agent.replay_buffer,
            log_interval=training_config.log_interval,
        )
        critic.collect_final_images(environment)

        set_to_first_images(alternating_images, environment)
        
        if episode_index % training_config.episodes_per_cycle == 0:
            agent_one.train(
                gradient_steps=agent_one.gradient_steps,
                batch_size=agent_one.batch_size,
            )
            agent_two.train(
                gradient_steps=agent_two.gradient_steps,
                batch_size=agent_two.batch_size,
            )
            critic.train()


def set_to_first_images(
    alternating_images: List[torch.tensor],
    environment: VecEnv,
):
    pass

if __name__ == "__main__":
    training_config = TrainingConfig()
    agent_config = AgentConfig()
    critic_config = CriticConfig()
    environment_config = EnvironmentConfig()

    train_models(
        training_config,
        agent_config,
        critic_config,
        environment_config
    )
