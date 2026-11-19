"""
train_warmstart.py  (formerly src/Running/TrainVelocity2.py, internally "TrainWarmStartLong")

This is the exact script that produced models/Velocity_WarmStart_HardDR_Long
(its save time matched the start of that run to the minute). The training
settings below are unchanged; the cleanup only moved shared code into
callbacks.py, moved paths into common/paths.py and added command-line options.

Run from src/ with the venv active:
    python -m training.train_warmstart --run-name MyWarmStart
    python -m training.train_warmstart --run-name Smoke --total-steps 20000 --n-envs 2   # quick test

It warm-starts from Velocity_Final/best_model.zip and pushes domain
randomisation BEYOND the eval-proven values for sim-to-real robustness.

──────────────────────────────────────────────────────────────────────────────
WHY ONE LONG RUN INSTEAD OF CHAINED RUNS
──────────────────────────────────────────────────────────────────────────────
  Loading a fully-trained PPO model with new hyperparameters caused immediate
  policy collapse (KL early-stopping at step 0, std dropping from 8.6 to 4.0,
  ep_rew_mean -2.4 from step 1). Extending a single run keeps the optimizer
  state consistent, lets the DR curriculum ramp from the warm-start, and gives
  the value function time to recalibrate.

──────────────────────────────────────────────────────────────────────────────
LESSONS APPLIED FROM PREVIOUS RUNS
──────────────────────────────────────────────────────────────────────────────
  v1 warm-start (failed): LR=5e-6 was too low. clip_fraction <0.01 in the
    final 3M steps means essentially zero learning.

  torque_bias permanently excluded: evaluation/torque_bias_test.py confirmed
    that even 10 µN·m flips the drone. Actuator gear=0.00001 makes any external
    body torque uncompensatable. The real drone handles rotor imbalance
    through its firmware PID, not the RL policy.

  tilt_min_z=0.20: below this the 25° init tilt is physically unrecoverable.

  max() in MultiStageDRCallback: later stages don't zero out earlier ones
    when their ramp starts at scale=0.

  EVAL always at zero DR: prevents noise variance from corrupting
    best_model selection.

──────────────────────────────────────────────────────────────────────────────
DR ESCALATION (beyond eval-proven values)
──────────────────────────────────────────────────────────────────────────────
                        nominal   eval-proven   HARD (this run)
  obs_noise_std           0.01       0.06         0.10  (1.67× eval)
  obs_bias_std            0.005      0.07         0.12  (1.71× eval)
  motor_scale_std         0.03       0.20         0.25  (1.25× eval)
  drag_lin_max            0.05       0.10         0.15  (1.50× eval)
  drag_quad_max           0.0        0.0          0.05  NEW
  action_noise_std        0.0        0.0          0.02  NEW
  frame_skip_jitter       0          2            2     (same as eval)

──────────────────────────────────────────────────────────────────────────────
CURRICULUM
──────────────────────────────────────────────────────────────────────────────
   0 –  2M : anchor at nominal SHARED DR     (let policy settle from warm-start)
             NOTE: in the original run a curriculum bug set DR to zero here
             instead of nominal. Fixed in callbacks.py on 2026-11-19.
   2M–  6M : ramp to eval-proven DR
   6M– 10M : ramp further to HARD DR         (beyond eval for sim2real margin)
  10M– 40M : 30M consolidation at HARD DR

  LR:         5e-5 → 2e-5
  clip:       0.10 → 0.05
  target_kl:  0.015
  ENT:        0.005 → 0.001
"""
import argparse
import copy
import multiprocessing
import os
import pickle

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from common.paths import SCENE_XML, model_dir, log_dir
from training.callbacks import (
    ZERO_DR, DebugCallback, EntropyCoefficientSchedule, MultiStageDRCallback,
    SaveVecNormOnBest, SyncedEvalCallback, linear_schedule, make_env, refuse_to_overwrite,
)

TARGET_Z = 1.0
MAX_STEPS = 1000
HOVER_REQ = 300

# The original run evaluated with eval_freq=200_000 and 16 envs. SB3 counts
# eval_freq per env, so that meant one evaluation every 3.2M total steps.
# Writing it as total steps and dividing by n_envs keeps the same behaviour
# with 16 envs and stays sensible for short test runs with fewer envs.
EVAL_EVERY_STEPS = 3_200_000

# ── DR curriculum: (start_step, end_step, {param: target}) ──────────────────
DR_STAGES = [
    # Phase 1: ramp to eval-proven values (known good)
    (2_000_000, 6_000_000, {
        "obs_noise_std":   0.06,
        "obs_bias_std":    0.07,
        "motor_scale_std": 0.20,
        "drag_lin_max":    0.10,
    }),
    # Frame skip jitter ramps in parallel
    (2_000_000, 4_000_000, {
        "frame_skip_jitter": 2,
    }),
    # Phase 2: push BEYOND eval for sim2real headroom
    (6_000_000, 10_000_000, {
        "obs_noise_std":    0.10,
        "obs_bias_std":     0.12,
        "motor_scale_std":  0.25,
        "drag_lin_max":     0.15,
        "drag_quad_max":    0.05,
        "action_noise_std": 0.02,
    }),
]

# ── Env params shared by training and debug episodes ────────────────────────
SHARED = dict(
    hover_required_steps=HOVER_REQ,
    auto_landing=False,
    init_tilt_max_deg=25.0,
    tilt_min_z=0.20,
    # Sensor DR: nominal from step 0, ramped harder by the curriculum
    obs_noise_std=0.01,
    obs_bias_std=0.005,
    action_noise_std=0.0,
    motor_scale_std=0.03,
    # Torque: permanently excluded (physically untrainable in this sim)
    torque_bias_std=0.0,
    torque_gust_std=0.0,
    torque_gust_tau=1.5,
    # Drag: nominal from step 0, ramped harder by the curriculum
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
    """Command-line options. The defaults reproduce the original 40M-step run."""
    p = argparse.ArgumentParser(description="Warm-start PPO from an existing model with hard DR.")
    p.add_argument("--run-name", default="Velocity_WarmStart_Rebuild",
                   help="output folder name under models/ and logs/")
    p.add_argument("--source", default="Velocity_Final", help="model folder to warm-start from")
    p.add_argument("--total-steps", type=int, default=40_000_000)
    p.add_argument("--n-envs", type=int, default=16, help="parallel simulator processes")
    p.add_argument("--overwrite", action="store_true", help="allow replacing an existing best_model.zip")
    return p.parse_args()


def main():
    args = parse_args()
    src_model_path = os.path.join(model_dir(args.source), "best_model.zip")
    src_norm_path = os.path.join(model_dir(args.source), "vecnormalize_best.pkl")
    models_dir = model_dir(args.run_name)
    logs_dir = log_dir(args.run_name)
    refuse_to_overwrite(models_dir, args.overwrite)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    if not os.path.exists(src_model_path):
        raise FileNotFoundError(f"Source model not found: {src_model_path}")

    # ── Vectorised envs ─────────────────────────────────────────────────────
    # "spawn" starts each env in a fresh Python process, which is the only
    # option on Windows and avoids sharing MuJoCo state between processes.
    train_venv = VecNormalize(
        SubprocVecEnv([make_env(SCENE_XML, TARGET_Z, MAX_STEPS, TRAIN) for _ in range(args.n_envs)],
                      start_method="spawn"),
        norm_obs=True, norm_reward=False, clip_obs=10.0)
    train_venv.training = True

    eval_venv = VecNormalize(
        SubprocVecEnv([make_env(SCENE_XML, TARGET_Z, MAX_STEPS, EVAL) for _ in range(args.n_envs)],
                      start_method="spawn"),
        norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

    # ── Load the source normaliser so observations are scaled as before ─────
    if os.path.exists(src_norm_path):
        with open(src_norm_path, "rb") as f:
            src_norm = pickle.load(f)
        train_venv.obs_rms = src_norm.obs_rms
        print(f"[WarmStart] Loaded obs_rms from {src_norm_path}")
    eval_venv.obs_rms = copy.deepcopy(train_venv.obs_rms)

    ent_start, ent_end = 0.005, 0.001

    # ── Load model and override its hyperparameters for this phase ──────────
    model = PPO.load(
        src_model_path,
        env=train_venv,
        learning_rate=linear_schedule(5e-5, 2e-5),
        clip_range=linear_schedule(0.10, 0.05),
        ent_coef=ent_start,
        device="auto",
        verbose=1,
    )
    model.tensorboard_log = logs_dir
    model.target_kl = 0.015
    print(f"[WarmStart] Loaded model from {src_model_path}")

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
    print(f"\n{sep}\n  Warm-start from {args.source} -> {args.run_name}  ({args.total_steps:,} steps, {args.n_envs} envs)")
    print("  LR 5e-5->2e-5 | clip 0.10->0.05 | kl=0.015 | ENT 0.005->0.001 | torque bias excluded")
    print(f"{sep}\n")

    model.learn(
        total_timesteps=args.total_steps,
        progress_bar=True,
        reset_num_timesteps=True,
        callback=CallbackList([eval_cb, ent_cb, dr_cb, dbg_cb]),
    )

    model.save(os.path.join(models_dir, "final_model.zip"))
    train_venv.save(os.path.join(models_dir, "vecnormalize_final.pkl"))
    print(f"\nDone -> {models_dir}")
    print("  best_model.zip + vecnormalize_best.pkl  : best clean-eval checkpoint, use these for evaluation")
    print("  final_model.zip + vecnormalize_final.pkl: weights at the very end of training")
    train_venv.close()
    eval_venv.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed on Windows when child processes are spawned
    main()
