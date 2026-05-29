"""
EvaluationVelocityReal.py — Pure policy evaluation on the real Crazyflie.

This file intentionally does NOT hardcode flight assistance:
- no scripted takeoff
- no vertical-only mode
- no action clipping in this evaluation file
- no policy handoff logic
- no manual command override

It loads the PPO policy, normalizes observations with the saved VecNormalize
statistics, sends the raw policy action directly to CrazyFlieRealEnvVelocity,
and logs enough information to judge what the policy is doing.

Safety behavior retained:
- safe-stop on Ctrl+C / exceptions / invalid observations
- env.close() on exit
- STM32 power cycle on every exit path
"""

import os
import sys
import time
import logging
import traceback
from typing import Any, Dict, Tuple

import numpy as np
import cflib.crtp
from cflib.utils.power_switch import PowerSwitch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from CrazyFlieEnvVelocity2 import CrazyFlieEnvVelocity

# Use the environment file you currently have in the folder. If you saved the
# uploaded original as CrazyFlieEnvReal2.py, this import will use it. If you
# renamed it back to CrazyFlieEnvReal.py, the fallback will use that.

from CrazyFlieEnvReal2 import CrazyFlieRealEnvVelocity



# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_real")
logging.getLogger("cflib").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware / model config
# ─────────────────────────────────────────────────────────────────────────────
URI = "radio://0/80/2M/E7E7E7E703"

here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(here, "..", ".."))

XML_PATH = os.path.join(PROJECT_ROOT, "Assets", "bitcraze_crazyflie_2", "scene.xml")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "Velocity_WarmStart_HardDR_Long")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model")
NORM_PATH = os.path.join(MODEL_DIR, "vecnormalize_best.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation config
# ─────────────────────────────────────────────────────────────────────────────
N_EPISODES = 1
TARGET_Z = 1.0
MAX_STEPS = 1500
LOG_PERIOD_MS = 20

# Keep the original real-env behavior. The environment default is 2.0 m.
SAFETY_RADIUS = 2.0

# Logging frequency
ACTION_LOG_EVERY_N_STEPS = 10
HUD_LOG_EVERY_N_STEPS = 10
SUMMARY_LOG_EVERY_N_STEPS = 50

# Hard emergency wall in case auto-landing or termination does not finish.
# This is not flight assistance; it is just a safety timeout for the script.
HARD_TOTAL_STEP_LIMIT = 2500


# ─────────────────────────────────────────────────────────────────────────────
# VecNormalize loader stub
# ─────────────────────────────────────────────────────────────────────────────
def make_norm_loader(xml_path: str, target_z: float, max_steps: int):
    """
    Build a dummy sim env so VecNormalize.load can attach to an env with the
    right observation/action spaces. This env is not stepped for real flight.
    """

    def _thunk():
        env = CrazyFlieEnvVelocity(
            xml_path=xml_path,
            target_z=target_z,
            max_steps=max_steps,
            n_stack=4,
            hover_required_steps=300,
            auto_landing=False,
            obs_noise_std=0.0,
            obs_bias_std=0.0,
            action_noise_std=0.0,
            motor_scale_std=0.0,
            torque_bias_std=0.0,
            torque_gust_std=0.0,
            drag_lin_max=0.0,
            drag_quad_max=0.0,
            frame_skip_jitter=0,
        )
        return Monitor(env)

    return DummyVecEnv([_thunk])


# ─────────────────────────────────────────────────────────────────────────────
# Safety helpers
# ─────────────────────────────────────────────────────────────────────────────
def safe_env_stop(env: Any) -> None:
    if env is None:
        return

    try:
        env.emergency_stop()
    except Exception:
        pass

    try:
        env._send_safe_stop()
    except Exception:
        pass


def force_power_cycle(uri: str) -> None:
    logger.info("Forcing STM32 power cycle on the Crazyflie...")
    try:
        PowerSwitch(uri).stm_power_cycle()
        time.sleep(2.0)
        logger.info("STM power cycle complete.")
    except Exception as e:
        logger.warning("STM power cycle failed: %s", e)
        logger.warning("MANUAL ACTION REQUIRED: unplug/replug the Crazyflie battery.")


# ─────────────────────────────────────────────────────────────────────────────
# Observation / diagnostics helpers
# ─────────────────────────────────────────────────────────────────────────────
def obs_is_valid(obs: np.ndarray) -> bool:
    if obs is None:
        return False
    obs = np.asarray(obs)
    return obs.shape == (52,) and np.all(np.isfinite(obs))


def latest_single_obs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    return obs[-13:]


def get_z_from_obs(obs: np.ndarray) -> float:
    return float(latest_single_obs(obs)[2])


def get_vz_from_obs(obs: np.ndarray) -> float:
    return float(latest_single_obs(obs)[9])


def get_xy_from_obs(obs: np.ndarray) -> Tuple[float, float]:
    s = latest_single_obs(obs)
    return float(s[0]), float(s[1])


def action_to_command_units(env: Any, action: np.ndarray) -> Dict[str, float]:
    """
    Convert normalized action to approximate command units using the env's
    configured limits. This is logging-only. It does not modify the action.
    """
    a = np.asarray(action, dtype=np.float32).reshape(4)

    max_roll = float(getattr(env, "max_roll_deg", float("nan")))
    max_pitch = float(getattr(env, "max_pitch_deg", float("nan")))
    max_yawrate = float(getattr(env, "max_yawrate_deg", float("nan")))
    max_vz = float(getattr(env, "max_vz_cmd", float("nan")))

    return {
        "roll_deg": float(a[0]) * max_roll,
        "pitch_deg": float(a[1]) * max_pitch,
        "yawrate_deg_s": float(a[2]) * max_yawrate,
        "vz_cmd": float(a[3]) * max_vz,
    }


def normalized_obs_stats(vecnorm: VecNormalize, obs: np.ndarray) -> Dict[str, float]:
    """Return simple stats for the normalized observation used by the policy."""
    obs_norm = vecnorm.normalize_obs(obs[None, :])
    flat = np.asarray(obs_norm, dtype=np.float32).reshape(-1)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "z_latest_norm": float(flat[-13 + 2]),
    }


def log_hud(
    step_idx: int,
    phase: str,
    obs: np.ndarray,
    info: Dict[str, Any],
    reward: float,
    ep_reward: float,
) -> None:
    z = float(info.get("z", get_z_from_obs(obs) if obs_is_valid(obs) else float("nan")))
    vz = float(info.get("vz", get_vz_from_obs(obs) if obs_is_valid(obs) else float("nan")))
    tilt = float(info.get("tilt_deg", float("nan")))
    radius = float(info.get("radius", float("nan")))
    hover = int(info.get("hover_steps", 0))

    x_rel, y_rel = get_xy_from_obs(obs) if obs_is_valid(obs) else (float("nan"), float("nan"))

    logger.info(
        "step=%4d | phase=%s | z=%.3f m vz=%+.3f | x_rel=%+.2f y_rel=%+.2f "
        "r=%.2f m | tilt=%.1f° | hover=%d | reward=%+.4f | R_total=%+.3f",
        step_idx,
        phase,
        z,
        vz,
        x_rel,
        y_rel,
        radius,
        tilt,
        hover,
        reward,
        ep_reward,
    )


def log_action(
    env: Any,
    step_idx: int,
    raw_action: np.ndarray,
    obs_norm_stats: Dict[str, float],
) -> None:
    cmd = action_to_command_units(env, raw_action)
    saturation_count = int(np.sum(np.abs(raw_action) > 0.95))

    logger.info(
        "step=%4d | raw_policy_action=%s | sat_axes=%d/4",
        step_idx,
        np.round(raw_action, 3).tolist(),
        saturation_count,
    )
    logger.info(
        "step=%4d | approx_cmd roll=%+.1f° pitch=%+.1f° yawrate=%+.1f°/s vz_cmd=%+.2f",
        step_idx,
        cmd["roll_deg"],
        cmd["pitch_deg"],
        cmd["yawrate_deg_s"],
        cmd["vz_cmd"],
    )
    logger.info(
        "step=%4d | norm_obs min=%+.2f max=%+.2f mean=%+.2f std=%.2f z_latest_norm=%+.2f",
        step_idx,
        obs_norm_stats["min"],
        obs_norm_stats["max"],
        obs_norm_stats["mean"],
        obs_norm_stats["std"],
        obs_norm_stats["z_latest_norm"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner: pure policy
# ─────────────────────────────────────────────────────────────────────────────
def run_episode(env: Any, model: PPO, vecnorm: VecNormalize, episode_idx: int, total_episodes: int):
    logger.info("=" * 78)
    logger.info("Episode %d / %d — PURE POLICY FROM RESET", episode_idx + 1, total_episodes)
    logger.info("=" * 78)

    obs, info = env.reset()

    if not obs_is_valid(obs):
        raise RuntimeError(f"Invalid obs immediately after reset: shape={getattr(obs, 'shape', None)}")

    spawn_xy = info.get("spawn_xy", np.zeros(2))
    spawn_z = float(info.get("spawn_z", 0.0))
    target_z_abs = float(info.get("target_z_abs", TARGET_Z))

    logger.info(
        "Spawn pose: x=%.3f y=%.3f z=%.3f",
        float(spawn_xy[0]),
        float(spawn_xy[1]),
        spawn_z,
    )
    logger.info("Target z absolute: %.3f m", target_z_abs)
    logger.info("Initial obs z=%.3f vz=%+.3f", get_z_from_obs(obs), get_vz_from_obs(obs))

    done = False
    truncated = False
    ep_reward = 0.0
    step_idx = 0
    last_info = info
    t_start = time.time()

    # Diagnostics counters
    saturation_steps = 0
    total_action_abs = np.zeros(4, dtype=np.float64)
    max_radius = 0.0
    max_tilt = 0.0
    max_z = get_z_from_obs(obs)
    min_z = get_z_from_obs(obs)

    while not (done or truncated):
        if not obs_is_valid(obs):
            raise RuntimeError(f"Invalid obs before policy action at step {step_idx}")

        obs_norm = vecnorm.normalize_obs(obs[None, :])
        raw_action, _ = model.predict(obs_norm, deterministic=True)
        raw_action = np.asarray(raw_action, dtype=np.float32)

        if raw_action.ndim == 2:
            raw_action = raw_action[0]

        raw_action = raw_action.reshape(4)

        if not np.all(np.isfinite(raw_action)):
            raise RuntimeError(f"Policy returned non-finite action at step {step_idx}: {raw_action}")

        total_action_abs += np.abs(raw_action)
        if np.any(np.abs(raw_action) > 0.95):
            saturation_steps += 1

        if step_idx % ACTION_LOG_EVERY_N_STEPS == 0:
            stats = normalized_obs_stats(vecnorm, obs)
            log_action(env, step_idx, raw_action, stats)

        # IMPORTANT: raw action is sent directly. No clamp, no scripted assist.
        obs, reward, done, truncated, info = env.step(raw_action)

        ep_reward += float(reward)
        last_info = info
        step_idx += 1

        z = float(info.get("z", get_z_from_obs(obs) if obs_is_valid(obs) else float("nan")))
        radius = float(info.get("radius", float("nan")))
        tilt = float(info.get("tilt_deg", float("nan")))

        if np.isfinite(z):
            max_z = max(max_z, z)
            min_z = min(min_z, z)
        if np.isfinite(radius):
            max_radius = max(max_radius, radius)
        if np.isfinite(tilt):
            max_tilt = max(max_tilt, tilt)

        phase = str(info.get("phase", "policy"))

        if step_idx % HUD_LOG_EVERY_N_STEPS == 0:
            log_hud(step_idx, phase, obs, info, float(reward), ep_reward)

        if step_idx % SUMMARY_LOG_EVERY_N_STEPS == 0:
            mean_abs_action = total_action_abs / max(1, step_idx)
            logger.info(
                "summary@%d | mean_abs_action=%s | saturation_steps=%d/%d | "
                "z_range=[%.3f, %.3f] | max_r=%.2f | max_tilt=%.1f°",
                step_idx,
                np.round(mean_abs_action, 3).tolist(),
                saturation_steps,
                step_idx,
                min_z,
                max_z,
                max_radius,
                max_tilt,
            )

        for key in ("crash", "ceiling", "success", "error", "timeout"):
            if info.get(key):
                logger.warning(
                    "env reported %s=%s | reason=%s | full info=%s",
                    key,
                    info[key],
                    info.get("reason", "—"),
                    info,
                )
                break

        if step_idx >= HARD_TOTAL_STEP_LIMIT:
            logger.warning("Hard total step limit reached: %d", HARD_TOTAL_STEP_LIMIT)
            safe_env_stop(env)
            truncated = True

    wall_time = time.time() - t_start
    mean_abs_action = total_action_abs / max(1, step_idx)

    logger.info("Episode %d done", episode_idx + 1)
    logger.info("steps=%d | total_reward=%+.3f | wall_time=%.1fs", step_idx, ep_reward, wall_time)
    logger.info(
        "episode assessment | mean_abs_action=%s | saturation_steps=%d/%d | "
        "z_range=[%.3f, %.3f] | max_radius=%.2f | max_tilt=%.1f°",
        np.round(mean_abs_action, 3).tolist(),
        saturation_steps,
        step_idx,
        min_z,
        max_z,
        max_radius,
        max_tilt,
    )
    logger.info("Last info: %s", last_info)

    return step_idx, ep_reward, last_info


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 78)
    logger.info("CrazyFlie REAL drone evaluation — PURE POLICY / NO ASSIST")
    logger.info("=" * 78)
    logger.info("URI:        %s", URI)
    logger.info("Model:      %s", MODEL_PATH)
    logger.info("VecNorm:    %s", NORM_PATH)
    logger.info("Episodes:   %d", N_EPISODES)
    logger.info("Target z:   %.2f m", TARGET_Z)
    logger.info("Max steps:  %d", MAX_STEPS)
    logger.info("Safety radius passed to env: %.2f m", SAFETY_RADIUS)
    logger.info("=" * 78)

    if not os.path.exists(MODEL_PATH + ".zip"):
        logger.error("Model not found at %s.zip", MODEL_PATH)
        sys.exit(1)

    if not os.path.exists(NORM_PATH):
        logger.error("VecNormalize stats not found at %s", NORM_PATH)
        sys.exit(1)

    if not os.path.exists(XML_PATH):
        logger.warning("Sim XML not found at %s — VecNormalize loader may fail", XML_PATH)

    env = None

    try:
        logger.info("Initializing cflib drivers...")
        cflib.crtp.init_drivers(enable_debug_driver=False)

        logger.info("Loading PPO model from %s.zip", MODEL_PATH)
        model = PPO.load(MODEL_PATH, device="cpu")
        logger.info(
            "Policy loaded | obs_space=%s | act_space=%s",
            model.observation_space,
            model.action_space,
        )

        logger.info("Loading VecNormalize stats from %s", NORM_PATH)
        norm_loader = make_norm_loader(XML_PATH, target_z=TARGET_Z, max_steps=MAX_STEPS)
        vecnorm: VecNormalize = VecNormalize.load(NORM_PATH, norm_loader)
        vecnorm.training = False
        vecnorm.norm_reward = False

        logger.info(
            "VecNormalize ready | obs_rms.mean[:3]=%s | obs_rms.var[:3]=%s",
            np.round(vecnorm.obs_rms.mean[:3], 4),
            np.round(vecnorm.obs_rms.var[:3], 4),
        )

        logger.info("Connecting to Crazyflie via CrazyFlieRealEnvVelocity...")
        env = CrazyFlieRealEnvVelocity(
            uri=URI,
            target_z=TARGET_Z,
            max_steps=MAX_STEPS,
            n_stack=4,
            hover_required_steps=300,
            auto_landing=True,
            safety_radius=SAFETY_RADIUS,
            log_period_ms=LOG_PERIOD_MS,
            debug=True,
        )

        logger.info(
            "Real env ready | max_roll=%.1f° max_pitch=%.1f° max_yawrate=%.1f°/s max_vz=%.2f",
            float(getattr(env, "max_roll_deg", float("nan"))),
            float(getattr(env, "max_pitch_deg", float("nan"))),
            float(getattr(env, "max_yawrate_deg", float("nan"))),
            float(getattr(env, "max_vz_cmd", float("nan"))),
        )
        logger.info(
            "Pure policy mode: raw model actions are sent directly to env.step(action)."
        )

        for ep in range(N_EPISODES):
            run_episode(env, model, vecnorm, ep, N_EPISODES)

        logger.info("All episodes completed.")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Emergency stopping.")
        safe_env_stop(env)

    except Exception as e:
        logger.error("Unexpected error in main(): %s", e)
        logger.error("Traceback:\n%s", traceback.format_exc())
        safe_env_stop(env)

    finally:
        try:
            if env is not None:
                logger.info("Closing CrazyFlie env...")
                env.close()
        except Exception:
            logger.exception("env.close() raised")

        force_power_cycle(URI)

        try:
            cflib.crtp.close_all()
        except Exception:
            pass

        logger.info("=" * 78)
        logger.info("Cleanup complete.")
        logger.info("=" * 78)


if __name__ == "__main__":
    main()
