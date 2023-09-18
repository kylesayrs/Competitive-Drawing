from datetime import datetime

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

import wandb
from wandb.integration.sb3 import WandbCallback

from config import EnvironmentConfig, AgentConfig
from environment import StrokeEnvironment

def init_wandb(
    agent_config: AgentConfig,
    environment_config: EnvironmentConfig
):
    wandb.init(
        project="CD_RL_Static",
        mode=agent_config.wandb_mode,
        sync_tensorboard=False,
        config={
            "agent_config": agent_config.dict(),
            "environment_config": environment_config.dict(),
        }
    )


def makeCallbacks(agent_config: AgentConfig):
    callbacks = []

    if agent_config.wandb_mode != "disabled":
        callbacks.append(WandbCallback(
            verbose=agent_config.verbose,
        ))

    return callbacks


def train_agent(
    agent_config: AgentConfig,
    environment_config: EnvironmentConfig
):
    init_wandb(agent_config, environment_config)

    environment = make_vec_env(
        StrokeEnvironment,
        env_kwargs={
            "environment_config": environment_config
        },
        n_envs=agent_config.n_envs
    )

    model = PPO(
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

    model.learn(
        total_timesteps=agent_config.total_timesteps,
        log_interval=agent_config.log_interval,
        callback=makeCallbacks(agent_config),
        progress_bar=agent_config.progress_bar,
    )
    now_string = str(datetime.now()).replace(" ", "_")
    save_path = f"checkpoints/{now_string}.zip"
    model.save(save_path)
    print(f"saved model to {save_path}")


if __name__ == "__main__":
    agent_config = AgentConfig()
    environment_config = EnvironmentConfig()

    train_agent(agent_config, environment_config)
