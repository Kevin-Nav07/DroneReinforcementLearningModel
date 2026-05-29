"""
TorqueBias Diagnostic — determines what torque_bias values, if any, are stable.

Tests a policy (or random actions) across a sweep of torque_bias_std values
to find the threshold at which the drone becomes uncontrollable.

Usage:
  python TestTorqueBias.py --model models/Velocity_Final/best_model.zip \
                           --norm   models/Velocity_Final/vecnormalize_best.pkl
"""

import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from CrazyFlieEnvVelocity2 import CrazyFlieEnvVelocity

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def make_env(xml_path, torque_bias, torque_gust, n_ep_steps=500):
    def _f():
        return Monitor(CrazyFlieEnvVelocity(
            xml_path           = xml_path,
            target_z           = 1.0,
            max_steps          = n_ep_steps,
            n_stack            = 4,
            hover_required_steps = 300,
            auto_landing       = False,
            safety_radius      = 2.0,
            init_tilt_max_deg  = 0.0,   # always upright — isolate torque effect
            tilt_min_z         = 0.20,
            start_z_min        = 0.50,  # mid-air start — skip ground effects
            start_z_max        = 0.50,
            start_xy_range     = 0.0,
            # All sensor DR off — isolate torque
            obs_noise_std      = 0.0,
            obs_bias_std       = 0.0,
            motor_scale_std    = 0.0,
            torque_bias_std    = float(torque_bias),
            torque_gust_std    = float(torque_gust),
            torque_gust_tau    = 1.5,
            drag_lin_max       = 0.0,
            frame_skip_jitter  = 0,
        ))
    return _f


def run_episodes(model, vecnorm_src, xml_path, torque_bias, torque_gust,
                 n_episodes=20, ep_steps=500, seed=42):
    venv = DummyVecEnv([make_env(xml_path, torque_bias, torque_gust, ep_steps)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False,
                        training=False, clip_obs=10.0)
    if vecnorm_src is not None:
        venv.obs_rms = vecnorm_src.obs_rms

    results = []
    rng = np.random.default_rng(seed)

    for _ in range(n_episodes):
        obs = venv.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = [False]
        ep_len, tilt_max, reason = 0, 0.0, "timeout"
        while not done[0]:
            if model is not None:
                act, _ = model.predict(obs, deterministic=True)
            else:
                act = rng.uniform(-1, 1, size=(1, 4)).astype(np.float32)
            obs, _, done, infos = venv.step(act)
            ep_len += 1
            tilt_max = max(tilt_max, float(infos[0].get("tilt_deg", 0)))
            if "reason" in infos[0]:
                reason = infos[0]["reason"]
        results.append((ep_len, tilt_max, reason))

    venv.close()

    flips   = sum(1 for _, _, r in results if "flip" in r or "nan" in r)
    crashes = sum(1 for _, _, r in results if r not in ("timeout",))
    avg_len = np.mean([l for l, _, _ in results])
    avg_tilt = np.mean([t for _, t, _ in results])
    return flips, crashes, avg_len, avg_tilt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Path to best_model.zip (omit for random-action baseline)")
    parser.add_argument("--norm", default=None,
                        help="Path to vecnormalize_best.pkl")
    parser.add_argument("--xml", default=None,
                        help="Path to scene.xml (default: auto-detect)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--ep_steps", type=int, default=500)
    args = parser.parse_args()

    xml_path = args.xml or os.path.join(
        PROJECT_ROOT, "Assets", "bitcraze_crazyflie_2", "scene.xml"
    )

    # Load model and normalization if provided
    model     = PPO.load(args.model) if args.model else None
    vecnorm   = None
    if args.norm and os.path.exists(args.norm):
        import pickle
        loader = DummyVecEnv([make_env(xml_path, 0.0, 0.0)])
        vecnorm = VecNormalize.load(args.norm, loader)
        vecnorm.training = False

    print(f"\n{'='*70}")
    print(f"  Torque Bias Diagnostic")
    print(f"  Model: {args.model or 'RANDOM ACTIONS'}")
    print(f"  Episodes per value: {args.episodes}  |  Steps per episode: {args.ep_steps}")
    print(f"  Init: upright, z=0.50m, no sensor DR  — isolating torque effect only")
    print(f"{'='*70}")
    print(f"\n  {'torque_bias':>12} {'gust':>8} | {'flips':>6} {'crashes':>8} {'avg_len':>8} {'avg_tilt':>9}  verdict")
    print(f"  {'-'*12} {'-'*8}-{'-'*6}-{'-'*8}-{'-'*8}-{'-'*9}----------")

    # Sweep: bias only, then bias + gust
    sweep = [
        # (torque_bias, torque_gust)
        (0.000000, 0.000),   # baseline
        (0.000001, 0.000),   # 1 µN·m
        (0.000005, 0.000),   # 5 µN·m
        (0.000010, 0.000),   # 10 µN·m — equals max actuator
        (0.000050, 0.000),   # 50 µN·m
        (0.000100, 0.000),   # 0.1 mN·m
        (0.000200, 0.000),   # 0.2 mN·m
        (0.000500, 0.000),   # 0.5 mN·m
        (0.001000, 0.000),   # 1 mN·m
        (0.002000, 0.000),   # 2 mN·m (safe-ish from physics calc)
        # with gust added
        (0.000100, 0.000050),
        (0.000100, 0.000100),
        (0.000200, 0.000100),
    ]

    safe_threshold = None
    for bias, gust in sweep:
        flips, crashes, avg_len, avg_tilt = run_episodes(
            model, vecnorm, xml_path, bias, gust,
            n_episodes=args.episodes, ep_steps=args.ep_steps
        )
        flip_pct  = flips / args.episodes * 100
        crash_pct = crashes / args.episodes * 100
        if flips == 0 and crashes <= 2:
            verdict = "✓ STABLE"
            if safe_threshold is None:
                safe_threshold = bias
        elif flips <= 2:
            verdict = "~ marginal"
        else:
            verdict = "✗ UNSTABLE"
            safe_threshold = None  # reset if we find instability

        print(f"  {bias:>12.6f} {gust:>8.6f} | {flips:>6} {crashes:>8} "
              f"{avg_len:>8.0f} {avg_tilt:>8.1f}°  {verdict}")

    print(f"\n{'='*70}")
    if safe_threshold is not None:
        print(f"  SAFE maximum torque_bias for training: ~{safe_threshold:.6f} N·m")
    else:
        print("  No stable torque_bias value found — exclude from training entirely.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()