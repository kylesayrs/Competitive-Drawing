from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env

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
    critic = Critic(**critic_config.dict())

    environment = make_vec_env(
        SoloEnvironment,
        env_kwargs={
            "environment_config": environment_config,
            "critic": critic
        },
        n_envs=1
    )

    agents_dict = [
        {
            "agent": DDPG(
                None
                **agent_config.dict()
            ),
            "steps_before_train": 0
        }
        for _ in range(2)
    ]

    rollout_callback = None
    agents_dict[0]["agent"]._setup_learn(0)
    agents_dict[1]["agent"]._setup_learn(0)

    time_steps_before_train = 0
    for episode_index in training_config.num_episodes:
        agent_number = episode_index % 2
        agent = agents_dict[agent_number]["agent"]

        environment.reset(agent_number=agent_number)

        rollout = agent.collect_rollouts(
            environment,
            train_freq=(1, "episode"),  # trick to only rollout one episode
            action_noise=agent.action_noise,
            callback=rollout_callback,
            learning_starts=agent.learning_starts,
            replay_buffer=agent.replay_buffer,
            log_interval=training_config.log_interval,
        )
        critic.collect_final_images(environment)

        agents_dict[agent_number]["steps_before_train"] += rollout.episode_timesteps
        
        if episode_index % training_config.episodes_per_cycle == 0:
            agents_dict[0]["agent"].train(
                gradient_steps=agents_dict[0]["steps_before_train"],
                batch_size=agents_dict[0]["agent"].batch_size,
            )
            agents_dict[1]["agent"].train(
                gradient_steps=agents_dict[1]["steps_before_train"],
                batch_size=agents_dict[1]["agent"].batch_size,
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
