from datetime import datetime

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback

from config import EnvironmentConfig, DDPGConfig, ModelConfig, PPOConfig
from environment import StrokeEnvironment

def init_wandb(
    model_config: ModelConfig,
    environment_config: EnvironmentConfig
):
    wandb.init(
        project="CD_RL_Static",
        mode=model_config.wandb_mode,
        sync_tensorboard=False,
        config={
            "model_config": model_config.dict(),
            "environment_config": environment_config.dict(),
        }
    )


def makeEvalCallback(model_config: ModelConfig):
    return EvalCallback(
        Monitor(StrokeEnvironment(environment_config)),
        n_eval_episodes=model_config.n_eval_episodes,
        eval_freq=model_config.eval_freq,
        render=model_config.eval_render,
    )


def makeCallbacks(model_config: ModelConfig):
    callbacks = []

    if model_config.wandb_mode != "disabled":
        callbacks.append(WandbCallback(
            verbose=model_config.verbose,
        ))

    if model_config.n_eval_episodes > 0:
        callbacks.append(makeEvalCallback(model_config))

    return callbacks


def train_agent(
    model_config: ModelConfig,
    environment_config: EnvironmentConfig
):
    init_wandb(model_config, environment_config)

    environment = make_vec_env(
        StrokeEnvironment,
        env_kwargs={
            "environment_config": environment_config
        },
        n_envs=model_config.n_envs
    )

    if isinstance(model_config, PPOConfig):
        model = PPO(  # TODO: Use DDPG
            model_config.policy,
            environment,
            policy_kwargs=model_config.policy_kwargs,
            learning_rate=model_config.learning_rate,
            n_steps=model_config.n_steps,
            batch_size=model_config.batch_size,
            n_epochs=model_config.n_epochs,
            gamma=model_config.gamma,
            gae_lambda=model_config.gae_lambda,
            clip_range=model_config.clip_range,
            verbose=model_config.verbose,
            device=model_config.device,
        )

    elif isinstance(model_config, DDPGConfig):
        model = DDPG(
            model_config.policy,
            environment,
            policy_kwargs=model_config.policy_kwargs,
            learning_starts=model_config.learning_starts,
            learning_rate=model_config.learning_rate,
            train_freq=model_config.train_freq,
            batch_size=model_config.batch_size,
            buffer_size=model_config.buffer_size,
            optimize_memory_usage=model_config.optimize_memory_usage,
            verbose=model_config.verbose,
            device=model_config.device
        )

    else:
        raise ValueError(f"Unknown model_config type {type(model_config)}")

    model.learn(
        total_timesteps=model_config.total_timesteps,
        log_interval=model_config.log_interval,
        callback=makeCallbacks(model_config),
        progress_bar=model_config.progress_bar,
    )
    now_string = str(datetime.now()).replace(" ", "_")
    save_path = f"checkpoints/{now_string}.zip"
    model.save(save_path)
    print(f"saved model to {save_path}")


if __name__ == "__main__":
    model_config = DDPGConfig()
    environment_config = EnvironmentConfig()

    train_agent(model_config, environment_config)
