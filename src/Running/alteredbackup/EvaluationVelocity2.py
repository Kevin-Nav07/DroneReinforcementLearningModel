"""CrazyFlie Evaluation v5 — auto_landing=True"""
import os, time
import numpy as np
import mujoco
from mujoco import viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from CrazyFlieEnvVelocity2 import CrazyFlieEnvVelocity


def make_loader(xml_path, target_z, max_steps):
    def _thunk():
        env = CrazyFlieEnvVelocity(
            xml_path=xml_path, target_z=target_z, max_steps=max_steps, n_stack=4,
            hover_required_steps=100, auto_landing=True, safety_radius=4.0,
            obs_noise_std=0.0, obs_bias_std=0.0, action_noise_std=0.0,
            motor_scale_std=0.0, torque_bias_std=0.0, torque_gust_std=0.0,
            drag_lin_max=0.0, drag_quad_max=0.0, frame_skip_jitter=0,
            start_z_min=0.01, start_z_max=0.01, random_start=False)
        return Monitor(env)
    return DummyVecEnv([_thunk])


if __name__ == "__main__":
    here         = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(here, "..", ".."))

    xml_path   = os.path.join(PROJECT_ROOT, "Assets", "bitcraze_crazyflie_2", "scene.xml")
    models_dir = os.path.join(PROJECT_ROOT, "models", "Velocity_Test_v5")

    # SB3 appends .zip — do NOT include extension
    model_path = os.path.join(models_dir, "best_model")
    norm_path  = os.path.join(models_dir, "vecnormalize_best.pkl")

    TARGET_Z, MAX_STEPS = 1.0, 1500

    model   = PPO.load(model_path)
    vecnorm = VecNormalize.load(norm_path, make_loader(xml_path, TARGET_Z, MAX_STEPS))
    vecnorm.training = False
    vecnorm.norm_reward = False

    env = CrazyFlieEnvVelocity(
        xml_path=xml_path, target_z=TARGET_Z, max_steps=MAX_STEPS, n_stack=4,
        hover_required_steps=100, auto_landing=True, safety_radius=4.0,
        obs_noise_std=0.0, obs_bias_std=0.0, action_noise_std=0.0,
        motor_scale_std=0.0, torque_bias_std=0.0, torque_gust_std=0.0,
        drag_lin_max=0.0, drag_quad_max=0.0, frame_skip_jitter=0,
        start_z_min=0.01, start_z_max=0.01, random_start=False)

    obs_raw, _ = env.reset()
    dt_step = env.model.opt.timestep * env.frame_skip

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        terminated = truncated = False
        ep_ret = 0.0
        t0 = last_print = time.time()

        while not (terminated or truncated):
            obs_n = vecnorm.normalize_obs(obs_raw[None, :])
            action, _ = model.predict(obs_n, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            if action.ndim == 2: action = action[0]

            obs_raw, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            z = float(obs_raw[2]); vz = float(obs_raw[9])
            v.sync()

            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                print(f"t={int(now-t0):3d}s | z={z:+.3f} vz={vz:+.3f} | "
                      f"phase={info.get('phase','?')} hover={info.get('hover_steps',0)} | "
                      f"tilt={info.get('tilt_deg',0):.1f}° r={info.get('radius',0):.2f} | R={ep_ret:.2f}")
            time.sleep(max(0.0, dt_step - (time.time() - now)))

    print(f"\nEpisode done. Return={ep_ret:.2f}  info={info}")