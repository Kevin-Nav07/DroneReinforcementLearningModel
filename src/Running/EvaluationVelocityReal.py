"""
EvaluationVelocityReal.py — Direct policy evaluation on real Crazyflie.

This version:
- Runs the PPO policy from reset immediately.
- Does NOT perform scripted takeoff.
- Does NOT clamp actions in the evaluation file.
- Sends raw policy actions directly to CrazyFlieRealEnvVelocity.
- Relies on CrazyFlieEnvReal.py to filter/safely reinterpret raw actions.
- Always safe-stops and power-cycles on exit.
"""

import os
import sys
import time
import logging
import traceback

import numpy as np
import cflib.crtp
from cflib.utils.power_switch import PowerSwitch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from CrazyFlieEnvVelocity2 import CrazyFlieEnvVelocity
from CrazyFlieEnvReal import CrazyFlieRealEnvVelocity


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
# Eval config
# ─────────────────────────────────────────────────────────────────────────────
N_EPISODES = 1
TARGET_Z = 1.0
MAX_STEPS = 500
LOG_PERIOD_MS = 20

ACTION_LOG_EVERY_N_STEPS = 10
HUD_LOG_EVERY_N_STEPS = 10

# Tighter than the previous 4m. The last flyaway reached >2m radius.
REAL_SAFETY_RADIUS = 2.0

# Hard wall so the script cannot run forever if landing state has an issue.
HARD_TOTAL_STEP_LIMIT = 1500


def make_norm_loader(xml_path: str, target_z: float, max_steps: int):
    """Build a dummy sim env so VecNormalize.load has the correct observation shape."""
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


def force_power_cycle(uri: str) -> None:
    logger.info("Forcing STM32 power cycle on the Crazyflie...")
    try:
        PowerSwitch(uri).stm_power_cycle()
        time.sleep(2.0)
        logger.info("STM power cycle complete.")
    except Exception as e:
        logger.warning("STM power cycle failed: %s", e)
        logger.warning("MANUAL ACTION REQUIRED: unplug/replug the Crazyflie battery.")


def safe_env_stop(env) -> None:
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


def format_action(value):
    if value is None:
        return None
    return np.round(np.asarray(value, dtype=np.float32), 3).tolist()


def log_hud(step_idx: int, phase: str, obs: np.ndarray, info: dict, reward: float, ep_reward: float) -> None:
    z = info.get("z", get_z_from_obs(obs) if obs_is_valid(obs) else float("nan"))
    vz = info.get("vz", get_vz_from_obs(obs) if obs_is_valid(obs) else float("nan"))
    tilt = info.get("tilt_deg", float("nan"))
    radius = info.get("radius", float("nan"))
    hover = info.get("hover_steps", 0)
    authority = info.get("policy_authority", float("nan"))

    raw_action = format_action(info.get("raw_action", None))
    safe_action = format_action(info.get("safe_action", None))

    logger.info(
        "step=%4d | phase=%s | z=%.3f m vz=%+.3f | tilt=%.1f° r=%.2f m | "
        "hover=%d | authority=%.2f | reward=%+.4f | R_total=%+.3f",
        step_idx,
        phase,
        z,
        vz,
        tilt,
        radius,
        hover,
        authority,
        reward,
        ep_reward,
    )

    if raw_action is not None or safe_action is not None:
        logger.info("step=%4d | env raw_action=%s | env safe_action=%s", step_idx, raw_action, safe_action)


def run_episode(env, model, vecnorm, episode_idx: int, total_episodes: int):
    logger.info("=" * 78)
    logger.info("Episode %d / %d — DIRECT POLICY FROM RESET", episode_idx + 1, total_episodes)
    logger.info("=" * 78)

    obs, info = env.reset()

    if not obs_is_valid(obs):
        raise RuntimeError(f"Invalid obs immediately after reset: shape={getattr(obs, 'shape', None)}")

    spawn_xy = info.get("spawn_xy", np.zeros(2))
    spawn_z = info.get("spawn_z", 0.0)
    target_z_abs = info.get("target_z_abs", TARGET_Z)

    logger.info("Spawn pose: x=%.3f y=%.3f z=%.3f", float(spawn_xy[0]), float(spawn_xy[1]), float(spawn_z))
    logger.info("Target z absolute: %.3f m", float(target_z_abs))
    logger.info("Initial obs z=%.3f vz=%+.3f", get_z_from_obs(obs), get_vz_from_obs(obs))

    done = False
    truncated = False
    ep_reward = 0.0
    step_idx = 0
    last_info = info
    t_start = time.time()

    while not (done or truncated):
        if not obs_is_valid(obs):
            raise RuntimeError(f"Invalid obs before policy action at step {step_idx}")

        obs_norm = vecnorm.normalize_obs(obs[None, :])
        raw_action, _ = model.predict(obs_norm, deterministic=True)
        raw_action = np.asarray(raw_action, dtype=np.float32)

        if raw_action.ndim == 2:
            raw_action = raw_action[0]

        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(4)

        if not np.all(np.isfinite(raw_action)):
            raise RuntimeError(f"Policy returned non-finite action at step {step_idx}: {raw_action}")

        if step_idx % ACTION_LOG_EVERY_N_STEPS == 0:
            logger.info("step=%4d | raw_policy_action=%s", step_idx, np.round(raw_action, 3).tolist())

        obs, reward, done, truncated, info = env.step(raw_action)

        ep_reward += float(reward)
        last_info = info
        step_idx += 1

        phase = info.get("phase", "direct_policy")

        if step_idx % HUD_LOG_EVERY_N_STEPS == 0:
            log_hud(step_idx, phase, obs, info, float(reward), ep_reward)

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
            logger.warning("Hard total step limit reached: %d. Emergency stopping.", HARD_TOTAL_STEP_LIMIT)
            safe_env_stop(env)
            truncated = True

    logger.info(
        "Episode %d done | steps=%d | total_reward=%+.3f | wall_time=%.1fs",
        episode_idx + 1,
        step_idx,
        ep_reward,
        time.time() - t_start,
    )
    logger.info("Last info: %s", last_info)

    return step_idx, ep_reward, last_info


def main():
    logger.info("=" * 78)
    logger.info("CrazyFlie REAL drone evaluation — DIRECT POLICY ONLY")
    logger.info("=" * 78)
    logger.info("URI:        %s", URI)
    logger.info("Model:      %s", MODEL_PATH)
    logger.info("VecNorm:    %s", NORM_PATH)
    logger.info("Episodes:   %d", N_EPISODES)
    logger.info("Target z:   %.2f m", TARGET_Z)
    logger.info("Max steps:  %d", MAX_STEPS)
    logger.info("Safety r:   %.2f m", REAL_SAFETY_RADIUS)
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
        logger.info("Policy loaded | obs_space=%s | act_space=%s", model.observation_space, model.action_space)

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
            safety_radius=REAL_SAFETY_RADIUS,
            log_period_ms=LOG_PERIOD_MS,
            debug=True,
        )

        logger.info("Real env ready. Starting direct policy evaluation.")

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
