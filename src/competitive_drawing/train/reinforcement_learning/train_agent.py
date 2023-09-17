from datetime import datetime

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

from competitive_drawing.train.reinforcement_learning.config import (
    EnvironmentConfig, AgentConfig
)
from competitive_drawing.train.reinforcement_learning.environment import StrokeEnvironment
from competitive_drawing.train.contrastive_learning.config import ModelsConfig
from competitive_drawing.train.reinforcement_learning.utils import load_agent

def init_wandb(
    agent_config: AgentConfig,
    models_config: ModelsConfig,
    environment_config: EnvironmentConfig
):
    wandb.init(
        project="CompetitiveDrawingRL",
        mode=agent_config.wandb_mode,
        sync_tensorboard=False,
        config={
            "agent_config": agent_config.dict(),
            "models_config": models_config.dict(),
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
    models_config: ModelsConfig,
    environment_config: EnvironmentConfig
):
    init_wandb(agent_config, models_config, environment_config)

    environment = make_vec_env(
        StrokeEnvironment,
        env_kwargs={
            "environment_config": environment_config,
            "models_config": models_config,
        },
        n_envs=agent_config.n_envs
    )

    agent = load_agent(agent_config, environment)

    agent.learn(
        total_timesteps=agent_config.total_timesteps,
        log_interval=agent_config.log_interval,
        callback=makeCallbacks(agent_config),
        progress_bar=agent_config.progress_bar,
    )
    now_string = str(datetime.now()).replace(" ", "_")
    save_path = f"checkpoints/{now_string}.zip"
    agent.save(save_path)
    print(f"saved model to {save_path}")


if __name__ == "__main__":
    models_config = ModelsConfig()
    agent_config = AgentConfig()
    environment_config = EnvironmentConfig()

    train_agent(agent_config, models_config, environment_config)
