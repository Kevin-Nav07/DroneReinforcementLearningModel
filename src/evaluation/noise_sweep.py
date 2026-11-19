"""
noise_sweep.py

Measures how robust trained hover policies are to each type of noise the sim
environment can inject, and to starting mid-air or tilted (the spawn tests
merged in from the old EvaluationSystemic.py). For every condition it runs N headless episodes
per model, using the same random seeds for every model so they face identical
noise, and prints a comparison table. Results are also saved to a CSV file in
logs/NoiseSweep/ so they can be used in the report.

Design choices:
  - One noise type is changed at a time, so a failure can be blamed on that
    noise type. Two combined conditions (ALL hard, ALL 2x hard) are added at the end.
  - Levels go past the values used in training (2x hard) to show safety margin.
  - Every episode starts on the ground (z = 0.01 m), like a real takeoff,
    except the spawn_air and spawn_tilt rows, which change only the start.
  - Metrics come from the simulator's true state, not the noisy observation,
    so we measure where the drone really is, not where it thinks it is.
  - Three outcome metrics are reported, because the env's own "success" rule
    (300 consecutive steps with |vz| < 0.05 m/s) is broken by tiny twitches
    under sensor noise even when the drone is hovering well:
        survive  = did not crash, flip or leave the safety radius
        success  = the env's strict 300-step hover rule
        in_band  = share of the last 5 s within 10 cm of target height
                   and 30 cm of the takeoff point

Usage (run from src/ with the venv active):
    python -m evaluation.noise_sweep                    # full sweep, 20 episodes
    python -m evaluation.noise_sweep --episodes 5       # quick check
    python -m evaluation.noise_sweep --watch HardDR_Long ALL 2x   # view one condition
"""
import os
import csv
import time
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from common.paths import SCENE_XML as XML_PATH, LOGS_DIR, model_dir
from envs.sim_env import CrazyFlieEnvVelocity


# ---------------------------------------------------------------------------
# Paths and models
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(LOGS_DIR, "NoiseSweep")

# Short display name -> model folder. Each folder must hold best_model.zip
# and the matching vecnormalize_best.pkl (the normaliser saved with it).
MODELS = {
    "HardDR_Long": "Velocity_WarmStart_HardDR_Long",  # trained 40M steps with hard noise
    "Final":       "Velocity_Final",                  # same network before noise training
}

TARGET_Z = 1.0
MAX_STEPS = 1000        # episode limit (20 s of sim time at the default 0.02 s per step)
HOVER_REQUIRED = 300    # consecutive stable steps needed for the env's strict success (6 s)
BASE_SEED = 12345       # fixed so every run of the sweep is reproducible
TAIL_STEPS = 250        # last 5 s of an episode, used for the steady-state metrics
BAND_Z = 0.10           # in-band height tolerance (m)
BAND_R = 0.30           # in-band sideways tolerance (m), same as the env's hover rule

# Settings shared by every condition: no noise, upright ground takeoff.
BASE_KWARGS = dict(
    target_z=TARGET_Z,
    max_steps=MAX_STEPS,
    n_stack=4,
    hover_required_steps=HOVER_REQUIRED,
    auto_landing=False,         # episode ends on success instead of starting a landing
    safety_radius=2.0,
    start_z_min=0.01,
    start_z_max=0.01,
    start_xy_range=0.0,
    init_tilt_max_deg=0.0,
    obs_noise_std=0.0,
    obs_bias_std=0.0,
    action_noise_std=0.0,
    motor_scale_std=0.0,
    frame_skip_jitter=0,
    torque_bias_std=0.0,
    torque_gust_std=0.0,
    drag_lin_max=0.0,
    drag_quad_max=0.0,
)

# The hard domain randomisation values the HardDR_Long model was trained on.
HARD = dict(obs_noise_std=0.10, obs_bias_std=0.12, motor_scale_std=0.25,
            action_noise_std=0.02, drag_lin_max=0.15, drag_quad_max=0.05,
            frame_skip_jitter=2)

# Each condition is (noise type, level label, settings that override BASE_KWARGS).
CONDITIONS = [
    ("CLEAN", "-", {}),
    # Sensor noise: jitter, slow drift, rare spikes, small gain errors
    ("obs_noise", "0.03", dict(obs_noise_std=0.03)),
    ("obs_noise", "0.06", dict(obs_noise_std=0.06)),
    ("obs_noise", "0.10 hard", dict(obs_noise_std=0.10)),
    ("obs_noise", "0.20 2x", dict(obs_noise_std=0.20)),
    # Constant sensor offset for the whole flight
    ("obs_bias", "0.03", dict(obs_bias_std=0.03)),
    ("obs_bias", "0.07", dict(obs_bias_std=0.07)),
    ("obs_bias", "0.12 hard", dict(obs_bias_std=0.12)),
    ("obs_bias", "0.24 2x", dict(obs_bias_std=0.24)),
    # Thrust multiplier per flight (battery level, motor wear)
    ("motor_scale", "0.05", dict(motor_scale_std=0.05)),
    ("motor_scale", "0.15", dict(motor_scale_std=0.15)),
    ("motor_scale", "0.25 hard", dict(motor_scale_std=0.25)),
    ("motor_scale", "0.50 2x", dict(motor_scale_std=0.50)),
    # Noise added to the policy's commands
    ("action_noise", "0.02 hard", dict(action_noise_std=0.02)),
    ("action_noise", "0.05", dict(action_noise_std=0.05)),
    ("action_noise", "0.10", dict(action_noise_std=0.10)),
    # Air resistance (linear and quadratic in speed)
    ("drag", "0.05", dict(drag_lin_max=0.05)),
    ("drag", "0.15+q0.05 hard", dict(drag_lin_max=0.15, drag_quad_max=0.05)),
    ("drag", "0.30+q0.10 2x", dict(drag_lin_max=0.30, drag_quad_max=0.10)),
    # Control timing wobble: physics substeps per action vary by +/- jitter around 10
    ("timing_jitter", "1", dict(frame_skip_jitter=1)),
    ("timing_jitter", "2 hard", dict(frame_skip_jitter=2)),
    ("timing_jitter", "4 2x", dict(frame_skip_jitter=4)),
    # Constant body torque. Excluded from training, so failure is expected here.
    ("torque_bias", "1e-6 untrained", dict(torque_bias_std=1e-6)),
    ("torque_bias", "5e-6 untrained", dict(torque_bias_std=5e-6)),
    # Everything at once
    ("ALL", "hard", dict(HARD)),
    ("ALL", "2x hard", {k: v * 2 for k, v in HARD.items()}),
    # Start mid-air instead of on the ground (no noise). Was EvaluationSystemic's CLEAN-AIR suite.
    ("spawn_air", "0.30 m", dict(start_z_min=0.30, start_z_max=0.30)),
    ("spawn_air", "0.60 m", dict(start_z_min=0.60, start_z_max=0.60)),
    ("spawn_air", "1.50 m", dict(start_z_min=1.50, start_z_max=1.50)),
    # Start with a random lean of up to N degrees (no noise). Training used up to 25 deg.
    ("spawn_tilt", "5 deg @0.6m", dict(start_z_min=0.60, start_z_max=0.60, init_tilt_max_deg=5.0, tilt_min_z=0.20)),
    ("spawn_tilt", "25 deg @1.0m trained", dict(start_z_min=1.00, start_z_max=1.00, init_tilt_max_deg=25.0, tilt_min_z=0.20)),
    ("spawn_tilt", "40 deg @1.0m beyond", dict(start_z_min=1.00, start_z_max=1.00, init_tilt_max_deg=40.0, tilt_min_z=0.20)),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(folder):
    """
    Load a PPO model together with the observation normaliser it was trained with.
    The normaliser needs a throwaway env with the same observation shape to load into.
    """
    folder_path = model_dir(folder)
    model = PPO.load(os.path.join(folder_path, "best_model"), device="cpu")  # small MLP runs faster on CPU
    dummy = DummyVecEnv([lambda: Monitor(CrazyFlieEnvVelocity(xml_path=XML_PATH, **BASE_KWARGS))])
    vecnorm = VecNormalize.load(os.path.join(folder_path, "vecnormalize_best.pkl"), dummy)
    vecnorm.training = False     # freeze mean/std so noisy test data cannot change them
    vecnorm.norm_reward = False  # rewards are only reported, never normalised
    return model, vecnorm


def make_env(overrides):
    """Build an env with the base settings plus this condition's noise settings."""
    return CrazyFlieEnvVelocity(xml_path=XML_PATH, **{**BASE_KWARGS, **overrides})


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------
def run_episode(env, model, vecnorm, seed, viewer=None):
    """
    Fly one deterministic episode and return a dict of metrics measured from
    the simulator's true state.
    """
    obs, _ = env.reset(seed=seed)   # same seed gives every model the same noise draws
    dt = env.model.opt.timestep * env.frame_skip  # seconds per policy step for this episode
    z_err_hist, r_hist = [], []
    max_tilt = 0.0
    info, steps = {}, 0
    done = False

    while not done:
        obs_n = vecnorm.normalize_obs(obs[None, :])               # same scaling the network saw in training
        action, _ = model.predict(obs_n, deterministic=True)      # no exploration noise during evaluation
        obs, _, terminated, truncated, info = env.step(action[0])
        done = terminated or truncated
        steps += 1

        tilt_rad, r = env._tilt_and_radius()                      # true tilt and sideways drift from takeoff point
        z_err_hist.append(abs(env.get_altitude() - TARGET_Z))
        r_hist.append(r)
        max_tilt = max(max_tilt, float(np.rad2deg(tilt_rad)))

        if viewer is not None:
            viewer.sync()
            time.sleep(dt)  # play back at real speed

    success = bool(info.get("success", False))
    if success:
        outcome = "success"
    elif info.get("timeout"):
        outcome = "timeout"
    else:
        outcome = info.get("reason", "unknown")

    z_tail = np.array(z_err_hist[-TAIL_STEPS:])
    r_tail = np.array(r_hist[-TAIL_STEPS:])
    survived = outcome in ("success", "timeout")                   # anything else is a crash or fly-away
    in_band = float(np.mean((z_tail < BAND_Z) & (r_tail < BAND_R))) if survived else 0.0
    return {
        "outcome": outcome,
        "survived": survived,
        "success": success,
        "in_band": in_band,
        "z_err": float(np.mean(z_tail)),
        "drift": float(np.mean(r_tail)),
        "max_tilt": max_tilt,
    }


# ---------------------------------------------------------------------------
# Parallel workers
# ---------------------------------------------------------------------------
_WORKER_MODELS = {}  # each worker process loads every model once and keeps it here


def _init_worker():
    """Runs once in each worker process: load all models so jobs can reuse them."""
    for name, folder in MODELS.items():
        _WORKER_MODELS[name] = load_model(folder)


def _run_job(job):
    """Run all episodes of one (condition, model) pair inside a worker process."""
    cond_idx, model_name, n_episodes = job
    _, _, overrides = CONDITIONS[cond_idx]
    model, vecnorm = _WORKER_MODELS[model_name]
    env = make_env(overrides)
    results = [run_episode(env, model, vecnorm, BASE_SEED + i) for i in range(n_episodes)]
    env.close()
    return cond_idx, model_name, results


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def summarise(results):
    """Turn a list of episode dicts into one table row."""
    fails = Counter(r["outcome"] for r in results if not r["survived"])
    return {
        "survive_pct": 100.0 * np.mean([r["survived"] for r in results]),
        "success_pct": 100.0 * np.mean([r["success"] for r in results]),
        "in_band_pct": 100.0 * np.mean([r["in_band"] for r in results]),
        "z_err_m": float(np.mean([r["z_err"] for r in results])),
        "drift_m": float(np.mean([r["drift"] for r in results])),
        "max_tilt_deg": float(np.max([r["max_tilt"] for r in results])),
        "main_crash": f"{fails.most_common(1)[0][0]} x{fails.most_common(1)[0][1]}" if fails else "-",
    }


def run_sweep(n_episodes, n_workers):
    """Run every condition for every model in parallel, then print and save the table."""
    jobs = [(i, name, n_episodes) for i in range(len(CONDITIONS)) for name in MODELS]
    t_start = time.time()
    print(f"Running {len(jobs)} jobs x {n_episodes} episodes on {n_workers} worker processes...", flush=True)

    collected = {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
        for cond_idx, model_name, results in pool.map(_run_job, jobs):  # map keeps the job order
            collected[(cond_idx, model_name)] = summarise(results)

    header = (f"{'noise type':<14}{'level':<21}{'model':<13}{'survive':>8}{'success':>8}"
              f"{'in_band':>8}{'z_err':>8}{'drift':>8}{'tilt':>7}  main crash")
    print(f"\nEpisodes per row: {n_episodes}   (ground takeoff unless spawn_*, target z = {TARGET_Z} m, 20 s max)")
    print("z_err / drift = mean over last 5 s (m), tilt = worst over all episodes (deg)\n")
    print(header)
    print("-" * len(header))

    rows = []
    for i, (noise_type, level, _) in enumerate(CONDITIONS):
        for name in MODELS:
            s = collected[(i, name)]
            rows.append({"noise_type": noise_type, "level": level, "model": name,
                         "episodes": n_episodes, **s})
            print(f"{noise_type:<14}{level:<21}{name:<13}{s['survive_pct']:>7.0f}%"
                  f"{s['success_pct']:>7.0f}%{s['in_band_pct']:>7.0f}%{s['z_err_m']:>8.3f}"
                  f"{s['drift_m']:>8.3f}{s['max_tilt_deg']:>7.1f}  {s['main_crash']}")
        print()

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"noise_sweep_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done in {time.time() - t_start:.0f} s. Saved: {csv_path}")


def watch(model_name, noise_type, level_prefix=""):
    """Replay one episode of one condition in the MuJoCo viewer."""
    import mujoco.viewer  # imported here so headless runs do not need a display
    matches = [c for c in CONDITIONS if c[0] == noise_type and c[1].startswith(level_prefix)]
    if not matches:
        raise SystemExit(f"No condition matches {noise_type} {level_prefix}")
    _, level, overrides = matches[-1]   # pick the strongest matching level
    model, vecnorm = load_model(MODELS[model_name])
    env = make_env(overrides)
    env.reset(seed=BASE_SEED)
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        r = run_episode(env, model, vecnorm, BASE_SEED, viewer=v)
    print(f"{model_name} | {noise_type} {level}: {r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="episodes per condition per model")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="parallel processes (default: all cores but two)")
    parser.add_argument("--watch", nargs="+", metavar="ARG",
                        help="view one episode: MODEL NOISE_TYPE [LEVEL], e.g. --watch HardDR_Long ALL 2x")
    args = parser.parse_args()

    if args.watch:
        watch(args.watch[0], args.watch[1], args.watch[2] if len(args.watch) > 2 else "")
    else:
        run_sweep(args.episodes, args.workers)
