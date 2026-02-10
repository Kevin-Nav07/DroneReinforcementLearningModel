# src/train_thrust_ppo_sweep.py
import os
import numpy as np
import gymnasium as gym

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from CrazyFlieEnvComplex import CrazyFlieEnv

# ---------- CLEAN ENV: domain randomization OFF for tuning ----------

DR_PARAMS = dict(
    obs_noise_std=0.0,
    obs_bias_std=0.0,
    action_noise_std=0.0,
    motor_scale_std=0.0,
    frame_skip=10,
    frame_skip_jitter=0,
)


def make_env(xml_path: str, target_z: float, max_steps: int, rank: int, seed: int):
    """
    Factory that creates a fresh clean env wrapped with Monitor.
    Each trial / process will call this, so every run is independent.
    """
    def _f():
        env = CrazyFlieEnv(
            xml_path=xml_path,
            target_z=target_z,
            max_steps=max_steps,
            n_stack=4,
            hover_required_steps=600,
            **DR_PARAMS,
        )
        # Let Gym handle seeding
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _f


def train():
    """
    Single training run for one set of hyperparameters.
    This is what wandb.agent(...) will call repeatedly.
    """
    here = os.path.dirname(__file__)

    # Paths
    xml_path = os.path.abspath(os.path.join(here, "..", "Assets", "bitcraze_crazyflie_2", "scene.xml"))
    base_models_dir = os.path.abspath(os.path.join(here, "..", "models", "Complex2_sweep"))
    base_logs_dir = os.path.abspath(os.path.join(here, "..", "logsComplex2_sweep"))

    os.makedirs(base_models_dir, exist_ok=True)
    os.makedirs(base_logs_dir, exist_ok=True)

    # Env constants
    TARGET_Z = 1.0
    MAX_STEPS = 1500
    N_ENVS = 8

    # ----- 1) Initialize wandb run -----
    run = wandb.init(
        project="crazyflie-ppo-clean",      # name it however you want
        config={
            # Default values (can be overridden by the sweep)
            "total_timesteps": 1_000_000,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.001,
            "vf_coef": 0.5,
            "hidden_size": 64,
            "seed": 0,
        },
        sync_tensorboard=True,  # sync SB3 TensorBoard to wandb
        monitor_gym=True,
        save_code=True,
    )

    config = wandb.config

    # ----- 2) Build clean vectorized env for THIS run -----
    env_fns = [
        make_env(xml_path, TARGET_Z, MAX_STEPS, rank=i, seed=config.seed)
        for i in range(N_ENVS)
    ]
    venv = DummyVecEnv(env_fns=env_fns)
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # Use a different logs dir per run
    logs_dir = os.path.join(base_logs_dir, run.name)
    os.makedirs(logs_dir, exist_ok=True)

    # ----- 3) Create PPO model with hyperparams from wandb config -----
    policy_kwargs = dict(net_arch=[config.hidden_size, config.hidden_size])

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=logs_dir,
        verbose=1,
    )

    # ----- 4) Train with WandbCallback -----
    model.learn(
        total_timesteps=config.total_timesteps,
        progress_bar=False,
        callback=WandbCallback(
            gradient_save_freq=0,
            model_save_path=base_models_dir,  # will save best models under this folder
            model_save_freq=0,                # you can set this >0 to save periodically
            verbose=1,
        ),
    )

    # Save final model + vecnorm for this run
    model_path = os.path.join(base_models_dir, f"{run.name}_final.zip")
    vecnorm_path = os.path.join(base_models_dir, f"{run.name}_vecnorm.pkl")
    model.save(model_path)
    venv.save(vecnorm_path)

    print(f"[{run.name}] Saved final model to {model_path}")
    print(f"[{run.name}] Saved VecNormalize to {vecnorm_path}")

    run.finish()


if __name__ == "__main__":
    # ----- 5) Define sweep config -----
    sweep_config = {
        "method": "bayes",  # or "random"
        "metric": {
            "name": "rollout/ep_rew_mean",
            "goal": "maximize",
        },
        "parameters": {
            "total_timesteps": {
                "value": 1_000_000  # same for all runs
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-3,
            },
            "n_steps": {
                "values": [1024, 2048, 4096],
            },
            "batch_size": {
                "values": [64, 128, 256],
            },
            "gamma": {
                "values": [0.98, 0.99, 0.995],
            },
            "gae_lambda": {
                "min": 0.9,
                "max": 0.99,
            },
            "clip_range": {
                "values": [0.1, 0.2, 0.3],
            },
            "ent_coef": {
                "values": [0.0, 0.001, 0.01],
            },
            "vf_coef": {
                "values": [0.3, 0.5, 0.7],
            },
            "hidden_size": {
                "values": [64, 128],
            },
            "seed": {
                "values": [0, 1, 2],
            },
        },
    }

    # ----- 6) Create sweep and launch agent -----
    sweep_id = wandb.sweep(sweep_config, project="crazyflie-ppo-clean")

    # 'count' is how many runs to do. Bump this up for overnight.
    wandb.agent(sweep_id, function=train, count=20)
