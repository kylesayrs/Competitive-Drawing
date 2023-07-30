from stable_baselines3 import DDPG, make_vec_env

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
    agent_one = DDPG(**agent_config.dict())
    agent_two = DDPG(**agent_config.dict())
    critic = Critic(**critic_config.dict())

    environment = make_vec_env(
        SoloEnvironment,
        env_kwargs={
            "environment_config": environment_config,
            "critic": critic
        },
        n_envs=1
    )

    rollout_callback = None

    for episode_index in training_config.num_episodes:
        agent = agent_one if episode_index % 2 == 0 else agent_two

        environment.reset(agent_number=(episode_index % 2) + 1)

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
