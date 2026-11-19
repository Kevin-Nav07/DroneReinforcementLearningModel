"""
train_from_scratch.py  (RECONSTRUCTION, written 2026-11-19)

The original script that produced models/Velocity_Final was never committed
and is lost. This file rebuilds it so the full pipeline can be rerun:

    train_from_scratch.py  ->  Velocity_Final-style model (20M steps)
    train_warmstart.py     ->  Velocity_WarmStart_HardDR_Long-style model (40M steps)

Run from src/ with the venv active:
    python -m training.train_from_scratch --run-name MyScratchRun
    python -m training.train_from_scratch --run-name Smoke --total-steps 20000 --n-envs 2   # quick test

──────────────────────────────────────────────────────────────────────────────
WHAT IS CONFIRMED (read from Velocity_Final's model zip and TensorBoard logs)
──────────────────────────────────────────────────────────────────────────────
  PPO MlpPolicy, net_arch [64, 64], 16 envs, 20M total steps
  n_steps 2048, batch 64, n_epochs 10, gamma 0.99, gae_lambda 0.95
  vf_coef 0.5, max_grad_norm 0.5, target_kl 0.02
  learning rate linear 1e-4 -> 2e-5   (logged train/learning_rate)
  clip range linear 0.20 -> 0.05      (logged train/clip_range)
  ent_coef linear 0.02 -> 0.005       (saved value 0.0104 at 12.8M fits exactly)
  one evaluation every 3.2M steps     (best model saved at 12.8M)
These match the archived TiltRobustness_v4 trainer, which this file is based on.

──────────────────────────────────────────────────────────────────────────────
WHAT IS ASSUMED (not recorded anywhere, so treat as a best guess)
──────────────────────────────────────────────────────────────────────────────
  Spawn and tilt settings are copied from train_warmstart.py, which was written
  right after Velocity_Final: 25 deg random spawn tilt, tilt_min_z 0.20,
  spawn height 0.01-1.80 m, xy range 1.0 m, safety radius 2.0 m.

  Domain randomisation: train_warmstart.py describes Velocity_Final's DR as
  "nominal" (always on) and "eval-proven" (reached later). The timing of the
  ramp below is a guess; episode length in the logs drops after about 10M
  steps, which is consistent with DR getting harder from there.
"""
import argparse
import copy
import multiprocessing
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from common.paths import SCENE_XML, model_dir, log_dir
from training.callbacks import (
    ZERO_DR, DebugCallback, EntropyCoefficientSchedule, MultiStageDRCallback,
    SaveVecNormOnBest, SyncedEvalCallback, linear_schedule, make_env, refuse_to_overwrite,
)

TARGET_Z = 1.0
MAX_STEPS = 1000     # 20 s episodes at 0.02 s per policy step
HOVER_REQ = 300      # 6 s of stable hover counts as success
EVAL_EVERY_STEPS = 3_200_000   # matches the original (eval_freq=200_000 per env x 16 envs)

# ── DR curriculum (ASSUMED timing, see docstring) ───────────────────────────
DR_STAGES = [
    # Ramp from nominal to the "eval-proven" values in the second half of training
    (10_000_000, 16_000_000, {
        "obs_noise_std":   0.06,
        "obs_bias_std":    0.07,
        "motor_scale_std": 0.20,
        "drag_lin_max":    0.10,
    }),
    (10_000_000, 12_000_000, {
        "frame_skip_jitter": 2,
    }),
]

# ── Env params shared by training and debug episodes ────────────────────────
SHARED = dict(
    hover_required_steps=HOVER_REQ,
    auto_landing=False,
    init_tilt_max_deg=25.0,   # random lean of 0-25 deg at spawn teaches tilt recovery
    tilt_min_z=0.20,          # tilted spawns are lifted to at least 0.20 m so recovery is possible
    # "Nominal" DR, on from step 0
    obs_noise_std=0.01,
    obs_bias_std=0.005,
    action_noise_std=0.0,
    motor_scale_std=0.03,
    # Torque noise is untrainable with this sim's actuator gear, so it stays off
    torque_bias_std=0.0,
    torque_gust_std=0.0,
    drag_lin_min=0.0,
    drag_lin_max=0.05,
    drag_quad_min=0.0,
    drag_quad_max=0.0,
    frame_skip_jitter=0,
)

TRAIN = dict(**SHARED, start_z_min=0.01, start_z_max=1.80, start_xy_range=1.0, safety_radius=2.0)

# Eval is always clean (no DR, no tilt, ground takeoff) so best_model selection is stable
EVAL = {**SHARED, **ZERO_DR, "init_tilt_max_deg": 0.0,
        "start_z_min": 0.01, "start_z_max": 0.01, "start_xy_range": 0.0, "safety_radius": 2.0}


def parse_args():
    """Command-line options. The defaults reproduce the original 20M-step run."""
    p = argparse.ArgumentParser(description="Train a hover policy with PPO from scratch.")
    p.add_argument("--run-name", default="Velocity_Scratch_Rebuild",
                   help="output folder name under models/ and logs/")
    p.add_argument("--total-steps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=16, help="parallel simulator processes")
    p.add_argument("--overwrite", action="store_true", help="allow replacing an existing best_model.zip")
    return p.parse_args()


def main():
    args = parse_args()
    models_dir = model_dir(args.run_name)
    logs_dir = log_dir(args.run_name)
    refuse_to_overwrite(models_dir, args.overwrite)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # ── Vectorised envs with observation normalisation ──────────────────────
    # VecNormalize keeps a running mean/std of every observation value and
    # rescales inputs to roughly mean 0, std 1, which makes PPO learn faster.
    train_venv = VecNormalize(
        SubprocVecEnv([make_env(SCENE_XML, TARGET_Z, MAX_STEPS, TRAIN) for _ in range(args.n_envs)],
                      start_method="spawn"),
        norm_obs=True, norm_reward=False, clip_obs=10.0)
    train_venv.training = True

    eval_venv = VecNormalize(
        SubprocVecEnv([make_env(SCENE_XML, TARGET_Z, MAX_STEPS, EVAL) for _ in range(args.n_envs)],
                      start_method="spawn"),
        norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)
    eval_venv.obs_rms = copy.deepcopy(train_venv.obs_rms)

    ent_start, ent_end = 0.02, 0.005

    # ── New model ───────────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        env=train_venv,
        learning_rate=linear_schedule(1e-4, 2e-5),
        clip_range=linear_schedule(0.20, 0.05),
        ent_coef=ent_start,           # EntropyCoefficientSchedule decays this during training
        target_kl=0.02,               # stops an update early if the policy changes too much
        n_steps=2048,                 # steps collected per env before each update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log=logs_dir,
        verbose=1,
        device="auto",
    )

    # ── Callbacks ───────────────────────────────────────────────────────────
    eval_cb = SyncedEvalCallback(
        train_venv=train_venv,
        eval_env=eval_venv,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=max(1, EVAL_EVERY_STEPS // args.n_envs),  # SB3 counts this per env
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        callback_on_new_best=SaveVecNormOnBest(train_venv, os.path.join(models_dir, "vecnormalize_best.pkl"), verbose=1),
    )
    ent_cb = EntropyCoefficientSchedule(args.total_steps, ent_start, ent_end)
    dr_cb = MultiStageDRCallback(train_venv, DR_STAGES, base_values=SHARED, verbose=1)  # keeps nominal DR on before stage 1
    dbg_cb = DebugCallback(SCENE_XML, TARGET_Z, MAX_STEPS, SHARED, every_n=500_000)

    sep = "=" * 78
    print(f"\n{sep}\n  From scratch -> {args.run_name}  ({args.total_steps:,} steps, {args.n_envs} envs)  [RECONSTRUCTION]")
    print("  LR 1e-4->2e-5 | clip 0.20->0.05 | kl=0.02 | ENT 0.02->0.005 | net [64,64]")
    print(f"{sep}\n")

    model.learn(
        total_timesteps=args.total_steps,
        progress_bar=True,
        callback=CallbackList([eval_cb, ent_cb, dr_cb, dbg_cb]),
    )

    model.save(os.path.join(models_dir, "final_model.zip"))
    train_venv.save(os.path.join(models_dir, "vecnormalize_final.pkl"))
    print(f"\nDone -> {models_dir}")
    train_venv.close()
    eval_venv.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed on Windows when child processes are spawned
    main()
