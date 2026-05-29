"""
EvaluateV2Systematic.py
=======================
Systematic evaluation of the Velocity_TiltRobustness_v2 best_model.

Runs three suites WITHOUT the MuJoCo viewer (headless) for speed, then
optionally opens the viewer for a single visual demonstration episode.

Suites
------
  CLEAN-GROUND  upright spawn, z = 0.01 m  (ground takeoff)
  CLEAN-AIR     upright spawn, z = 0.30 / 0.60 / 1.00 / 1.50 m
  TILT-5        5° lean,  z = 0.30 / 0.60 / 1.00 / 1.30 m

Each suite runs N_REPS deterministic episodes per height and records:
  - Outcome (SUCCESS / timeout / crash reason)
  - Final z (m), peak tilt (°), lateral radius (m), return R
  - Whether auto-landing succeeded

A summary table is printed to stdout so you can copy it into the report.

Usage
-----
    python EvaluateV2Systematic.py              # headless suites + demo
    python EvaluateV2Systematic.py --no-demo    # headless suites only
"""

import os
import sys
import time
import copy
import argparse
import numpy as np

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from CrazyFlieEnvVelocity2 import CrazyFlieEnvVelocity


# ── paths ─────────────────────────────────────────────────────────────────────

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
XML_PATH     = os.path.join(PROJECT_ROOT, "Assets", "bitcraze_crazyflie_2", "scene.xml")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models", "Velocity_TiltRobustness_v2")
MODEL_PATH   = os.path.join(MODELS_DIR, "best_model")   # SB3 appends .zip
NORM_PATH    = os.path.join(MODELS_DIR, "vecnormalize_best.pkl")

TARGET_Z  = 1.0
MAX_STEPS = 1000
N_REPS    = 3     # episodes per (suite, height) — increase for cleaner stats

# ── shared env kwargs (no noise, no DR) ───────────────────────────────────────

BASE_KWARGS = dict(
    target_z             = TARGET_Z,
    max_steps            = MAX_STEPS,
    n_stack              = 4,
    hover_required_steps = 300,
    auto_landing         = True,
    safety_radius        = 4.0,
    obs_noise_std        = 0.0,
    obs_bias_std         = 0.0,
    action_noise_std     = 0.0,
    motor_scale_std      = 0.0,
    torque_bias_std      = 0.0,
    torque_gust_std      = 0.0,
    drag_lin_max         = 0.0,
    drag_quad_max        = 0.0,
    frame_skip_jitter    = 0,
    start_xy_range       = 0.0,
)

# ── evaluation suites ─────────────────────────────────────────────────────────

SUITES = [
    {
        "label":   "CLEAN-GROUND",
        "tilt":    0.0,
        "heights": [0.01],
        "reps":    5,        # more reps for the most important case
        "desc":    "Upright spawn from ground — the primary hardware-relevant test",
    },
    {
        "label":   "CLEAN-AIR",
        "tilt":    0.0,
        "heights": [0.30, 0.60, 1.00, 1.50],
        "reps":    N_REPS,
        "desc":    "Upright spawn at various altitudes",
    },
    {
        "label":   "TILT-5",
        "tilt":    5.0,
        "heights": [0.30, 0.60, 1.00, 1.30],
        "reps":    N_REPS,
        "desc":    "5° lean at spawn (training distribution)",
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def make_loader(xml_path, target_z, max_steps):
    """Minimal DummyVecEnv needed to load VecNormalize from disk."""
    def _thunk():
        env = CrazyFlieEnvVelocity(
            xml_path=xml_path,
            **{**BASE_KWARGS, "target_z": target_z, "max_steps": max_steps,
               "start_z_min": 0.01, "start_z_max": 0.01,
               "init_tilt_max_deg": 0.0},
        )
        return Monitor(env)
    return DummyVecEnv([_thunk])


def run_episode(env, model, vecnorm, spawn_z, tilt_deg, seed=None):
    """
    Run one deterministic episode and return a result dict.
    env must have init_tilt_max_deg and tilt_min_z already set.
    We override start_z_min/max via reset options is not available in Gym,
    so instead we re-set them on the env object before reset.
    """
    env.start_z_min = spawn_z
    env.start_z_max = spawn_z
    env.init_tilt_max_deg = tilt_deg
    # tilt_min_z is already 0.30 from construction

    obs_raw, _ = env.reset(seed=seed)
    ep_ret     = 0.0
    terminated = truncated = False
    max_tilt   = 0.0
    max_radius = 0.0
    last_info  = {}
    steps      = 0

    while not (terminated or truncated):
        obs_n         = vecnorm.normalize_obs(obs_raw[None, :])
        action, _     = model.predict(obs_n, deterministic=True)
        action        = np.asarray(action, dtype=np.float32)
        if action.ndim == 2:
            action = action[0]

        obs_raw, reward, terminated, truncated, info = env.step(action)
        ep_ret    += float(reward)
        steps     += 1
        last_info  = info
        max_tilt   = max(max_tilt,   float(info.get("tilt_deg", 0.0)))
        max_radius = max(max_radius, float(info.get("radius",   0.0)))

    # determine outcome label
    if last_info.get("success"):
        outcome = "SUCCESS"
    elif last_info.get("crash"):
        outcome = f"crash:{last_info.get('reason', '?')}"
    elif last_info.get("ceiling"):
        outcome = "ceiling"
    elif last_info.get("timeout"):
        # timed out but check hover quality
        outcome = "timeout"
    elif last_info.get("phase") == "landing":
        # landed via auto-landing
        landed = last_info.get("landing_landed", False)
        outcome = "landed" if landed else "landing_timeout"
    else:
        outcome = "unknown"

    # final z from the raw obs (spawn-relative; convert to absolute)
    final_z_rel = float(obs_raw[2])
    final_z_abs = final_z_rel  # z_abs is stored directly in obs index 2

    return {
        "outcome":    outcome,
        "steps":      steps,
        "return":     ep_ret,
        "final_z":    final_z_abs,
        "max_tilt":   max_tilt,
        "max_radius": max_radius,
        "hover_count": int(last_info.get("hover_steps", 0)),
        "landed":     last_info.get("landing_landed", False),
        "info":       last_info,
    }


def fmt_outcome(r):
    """Short display string for a result dict."""
    success_like = r["outcome"] in ("SUCCESS", "landed")
    star = "\u2713" if success_like else " "
    return (f"{star} {r['outcome']:<22s}  "
            f"z={r['final_z']:+.3f}m  tilt={r['max_tilt']:5.1f}\u00b0  "
            f"r={r['max_radius']:.2f}m  R={r['return']:+.3f}  "
            f"steps={r['steps']:4d}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-demo", action="store_true",
                        help="Skip the visual demonstration episode")
    parser.add_argument("--reps", type=int, default=N_REPS,
                        help="Episodes per (suite, height) override")
    args = parser.parse_args()

    # ── load model + normaliser ───────────────────────────────────────────────
    print(f"\nLoading model:    {MODEL_PATH}.zip")
    print(f"Loading vecnorm:  {NORM_PATH}")
    model   = PPO.load(MODEL_PATH)
    vecnorm = VecNormalize.load(NORM_PATH, make_loader(XML_PATH, TARGET_Z, MAX_STEPS))
    vecnorm.training    = False
    vecnorm.norm_reward = False

    # ── build a single reusable env (headless) ────────────────────────────────
    eval_env = CrazyFlieEnvVelocity(
        xml_path=XML_PATH,
        **{**BASE_KWARGS,
           "start_z_min": 0.01, "start_z_max": 0.01,
           "init_tilt_max_deg": 0.0,
           "tilt_min_z": 0.30},
    )

    # ── run all suites ────────────────────────────────────────────────────────
    all_results = {}   # suite_label -> list of (height, rep, result_dict)

    for suite in SUITES:
        label     = suite["label"]
        tilt_deg  = suite["tilt"]
        heights   = suite["heights"]
        n         = args.reps if suite["label"] not in ("CLEAN-GROUND",) else suite["reps"]
        all_results[label] = []

        SEP = "=" * 70
        print(f"\n{SEP}")
        print(f"  Suite: {label}   tilt={tilt_deg:.0f}\u00b0   reps={n}")
        print(f"  {suite['desc']}")
        print(SEP)

        for z in heights:
            results_this_height = []
            for rep in range(n):
                seed = 1000 * int(z * 100) + rep
                res  = run_episode(eval_env, model, vecnorm,
                                   spawn_z=z, tilt_deg=tilt_deg, seed=seed)
                results_this_height.append(res)
                all_results[label].append((z, rep, res))

            # aggregate for this height
            outcomes      = [r["outcome"] for r in results_this_height]
            returns       = [r["return"]  for r in results_this_height]
            tilts         = [r["max_tilt"] for r in results_this_height]
            n_success     = sum(1 for o in outcomes if o in ("SUCCESS", "landed"))

            print(f"\n  spawn_z = {z:.2f} m  →  {n_success}/{n} success-like")
            for i, res in enumerate(results_this_height):
                print(f"    rep {i+1}: {fmt_outcome(res)}")
            print(f"    mean R = {np.mean(returns):+.3f}   "
                  f"max tilt = {max(tilts):.1f}\u00b0")
    
    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  SUMMARY TABLE ")
    print(f"{'='*70}")
    print(f"  {'Suite':<14} {'z(m)':<6} {'Success':<10} {'Mean R':<10} "
          f"{'Max Tilt':<10} {'Mean z_final'}")
    print(f"  {'-'*64}")

    s_systemic = np.zeros(shape=(4, 25), dtype=np.float32)
    H_0 = np.abs
  
    
    for suite in SUITES:
        label    = suite["label"]
        heights  = suite["heights"]
        entries  = all_results[label]

        for z in heights:
            recs = [r for (h, _, r) in entries if abs(h - z) < 1e-6]
            if not recs:
                continue
            n_suc  = sum(1 for r in recs if r["outcome"] in ("SUCCESS", "landed"))
            n_tot  = len(recs)
            mean_R = np.mean([r["return"]   for r in recs])
            max_t  = max   ([r["max_tilt"]  for r in recs])
            mean_z = np.mean([r["final_z"]  for r in recs])
            suc_str = f"{n_suc}/{n_tot}"
            print(f"  {label:<14} {z:<6.2f} {suc_str:<10} {mean_R:<+10.3f} "
                  f"{max_t:<10.1f} {mean_z:+.3f}")

    print(f"{'='*70}\n")

    eval_env.close()

    # ── visual demo episode ───────────────────────────────────────────────────
    if args.no_demo:
        print("Skipping visual demo (--no-demo).")
        return

    print("\nOpening MuJoCo viewer for visual demo...")
    print("  Suite: CLEAN-GROUND  (ground spawn, no tilt, auto_landing=True)")
    print("  Press Ctrl-C or close the viewer window to exit.\n")

    demo_env = CrazyFlieEnvVelocity(
        xml_path=XML_PATH,
        **{**BASE_KWARGS,
           "start_z_min": 0.01, "start_z_max": 0.01,
           "init_tilt_max_deg": 0.0,
           "tilt_min_z": 0.30},
    )
    obs_raw, _ = demo_env.reset()
    dt_step    = demo_env.model.opt.timestep * demo_env.frame_skip

    with mujoco.viewer.launch_passive(demo_env.model, demo_env.data) as v:
        terminated = truncated = False
        ep_ret = 0.0
        t0 = last_print = time.time()

        while not (terminated or truncated):
            obs_n     = vecnorm.normalize_obs(obs_raw[None, :])
            action, _ = model.predict(obs_n, deterministic=True)
            action    = np.asarray(action, dtype=np.float32)
            if action.ndim == 2:
                action = action[0]

            obs_raw, reward, terminated, truncated, info = demo_env.step(action)
            ep_ret += float(reward)

            # phase from env directly (never shows "?")
            phase  = demo_env.phase
            z      = float(obs_raw[2])
            vz     = float(obs_raw[9])
            v.sync()

            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                tilt   = float(info.get("tilt_deg",   0.0))
                hover  = int  (info.get("hover_steps", 0))
                radius = float(info.get("radius",      0.0))
                print(f"  t={int(now-t0):3d}s | phase={phase:6s} | "
                      f"z={z:+.3f}m  vz={vz:+.4f}m/s | "
                      f"tilt={tilt:5.1f}\u00b0  r={radius:.2f}m | "
                      f"hover={hover:4d}/300  R={ep_ret:.3f}")
            time.sleep(max(0.0, dt_step - (time.time() - now)))

    # outcome
    if info.get("success"):
        outcome = "SUCCESS"
    elif info.get("phase") == "landing":
        landed  = info.get("landing_landed", False)
        outcome = "LANDED" if landed else f"landing ({info.get('landing_mode','?')})"
    elif info.get("crash"):
        outcome = f"CRASH: {info.get('reason', '?')}"
    else:
        outcome = f"timeout  hover_count={info.get('hover_steps',0)}"

    print(f"\n  Demo done: {outcome}  return={ep_ret:.3f}")
    demo_env.close()


if __name__ == "__main__":
    main()