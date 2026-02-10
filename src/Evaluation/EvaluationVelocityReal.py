

import os
import time
import logging

import numpy as np
import cflib.crtp
from cflib.utils.power_switch import PowerSwitch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from CrazyFlieEnvVelocity import CrazyFlieEnvVelocity
from CreazyFlieRealEnvVelocity import CrazyFlieRealEnvVelocity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URI = "radio://0/80/2M/E7E7E7E701"##URI for our drone, will vary on drones

here = os.path.dirname(__file__)

XML_PATH = os.path.abspath(
    os.path.join(here, "..", "Assets", "bitcraze_crazyflie_2", "scene.xml")
)


MODEL_DIR  = os.path.abspath(os.path.join(here, "..", "models", "VerticalComplex_DR_Curriculum"))
MODEL_DIR2  = os.path.abspath(os.path.join(here, "..", "models", "VerticalComplex_DR_Curriculum", "stage4_very_high_noise"))
MODEL_NAME = "best_model.zip"
NORM_NAME  = "vecnormalize_curriculum.pkl"

MODEL_PATH = os.path.join(MODEL_DIR2, MODEL_NAME)
NORM_PATH  = os.path.join(MODEL_DIR, NORM_NAME)

N_EPISODES = 1
TARGET_Z   = 1.0
MAX_STEPS  = 1500


def make_norm_loader(xml_path: str, target_z: float, max_steps: int):
    def _thunk():
        env = CrazyFlieEnvVelocity(
            xml_path=xml_path,
            target_z=target_z,
            max_steps=max_steps,
            n_stack=4,
            hover_required_steps=600,
            auto_landing=False,   
        )
        return Monitor(env)
    return DummyVecEnv([_thunk])


def main():
    cflib.crtp.init_drivers(enable_debug_driver=False)

    logger.info("Loading PPO model from %s", MODEL_PATH)
    model = PPO.load(MODEL_PATH, device="cpu")
    logger.info("Loading VecNormalize stats from %s", NORM_PATH)
    norm_loader = make_norm_loader(XML_PATH, target_z=TARGET_Z, max_steps=MAX_STEPS)
    vecnorm: VecNormalize = VecNormalize.load(NORM_PATH, norm_loader)
    vecnorm.training = False       # freeze stats
    vecnorm.norm_reward = False    # don't normalize rewards at eval

    env = None

    try:
        logger.info("Creating CrazyFlieRealEnvVelocity...")
        env = CrazyFlieRealEnvVelocity(
            uri=URI,
            target_z=TARGET_Z,
            max_steps=MAX_STEPS,
            n_stack=4,
            control_dt=0.02,      # match sim step (~50 Hz)
            safety_radius=100.0,
            auto_landing=False,   # landing still manual / external
        )

        logger.info("Starting Stage-2 RL evaluation on REAL Crazyflie (velocity env).")
        logger.info("Episodes: %d | target_z = %.2f m", N_EPISODES, TARGET_Z)

        for ep in range(N_EPISODES):
            logger.info("=== Episode %d / %d ===", ep + 1, N_EPISODES)

            obs, info = env.reset()
            done = False
            truncated = False
            ep_reward = 0.0
            step_idx = 0

            while not (done or truncated):
                obs_norm = vecnorm.normalize_obs(obs[None, :])  # shape (1, obs_dim)

                ##Deterministic policy for evaluation
                action, _ = model.predict(obs_norm, deterministic=True)

                # SB3 sometimes returns shape (1,4); flatten to (4,)
                action = np.asarray(action, dtype=np.float32)
                if action.ndim == 2:
                    action_to_env = action[0]
                else:
                    action_to_env = action
                
                if step_idx % 10 == 0:
                    logger.info(
                        "DEBUG policy action (raw): %s",
                        np.round(action_to_env, 3), 
                    )
                obs, reward, done, truncated, info = env.step(action_to_env)
                ep_reward += float(reward)
                step_idx += 1

                ##light hud
                if step_idx % 10 == 0:
                    z = info.get("z", float("nan"))
                    tilt = info.get("tilt_deg", float("nan"))
                    radius = info.get("radius", float("nan"))

                    logger.info(
                        "step=%4d | z=%.3f m | tilt=%.1f deg | r=%.2f m | r_step=%.3f",
                        step_idx,
                        z,
                        tilt,
                        radius,
                        reward,
                    )

                ##Early Safeyt exit
                if info.get("crash") or info.get("ceiling"):
                    logger.warning("Episode ended due to safety condition: %s", info)
                    break

            logger.info(
                "Episode %d done after %d steps | total reward = %.3f | last info = %s",
                ep + 1,
                step_idx,
                ep_reward,
                info,
            )

            # Brief pause between episodes (you can move / reset drone)
            time.sleep(1.0)

        logger.info("All Stage-2 velocity-env episodes completed.")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). Stopping motors and cleaning up...")
        try:
            if env is not None:
                env.emergency_stop()
        except Exception:
            pass

    except Exception as e:
        logger.error("Unexpected error in Stage-2 velocity evaluation: %s", e)
        try:
            if env is not None:
                env.emergency_stop()
        except Exception:
            pass

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:##VERy important to power cycle the stm to avoid weird states with the real crazy flie
            logger.info("Forcing STM power cycle...")
            PowerSwitch(URI).stm_power_cycle()
            time.sleep(1.0)
        except Exception as e:
            logger.warning("STM power cycle failed: %s", e)

        try:
            cflib.crtp.close_all()
        except Exception:
            pass

        logger.info("Stage-2 velocity evaluation cleanup complete.")


if __name__ == "__main__":
    main()
