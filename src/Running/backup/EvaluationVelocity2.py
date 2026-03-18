import os
import time
import numpy as np
import mujoco
from mujoco import viewer

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from CrazyFlieEnvVelocity2 import CrazyFlieEnvVelocity


def _make_norm_loader(xml_path: str, target_z: float = 1000, max_steps: int=300):
    """
    Factory for a single-env DummyVecEnv used only to load VecNormalize
    statistics from disk (no real training happens here).
    """
    def _thunk():
        env = CrazyFlieEnvVelocity(
            xml_path=xml_path,
            target_z=target_z,
            max_steps=max_steps,
            n_stack=4,
            hover_required_steps=300,

        obs_noise_std=0.0,
        obs_bias_std=0.00,


        action_noise_std=0.0,

        motor_scale_std=0.0,


        frame_skip=10,
        frame_skip_jitter=0,


        start_xy_range=0,
        start_z_min=0.01,
        start_z_max=0.01,

        torque_bias_std=0.000,
        torque_gust_std=0.0,
        torque_gust_tau=0,   
        
         drag_lin_min=0.00,
         drag_lin_max=0.0,
         drag_quad_min=0.0,
         drag_quad_max=0.0,
            random_start=False,
            auto_landing=False,  
        )
        return Monitor(env)

    return DummyVecEnv([_thunk])


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    xml_path = os.path.abspath(
        os.path.join(here, "..", "..", "Assets", "bitcraze_crazyflie_2", "scene.xml")
    )


    models_dir = os.path.abspath(os.path.join(here, "..", "..", "models", "Velocity_Test"))
    model_path = os.path.join(models_dir, "best_model.zip")
    norm_path  = os.path.join(models_dir, "vecnormalize_best.pkl")

    TARGET_Z = 1.0
    MAX_STEPS = 1000

    model = PPO.load(model_path)

    norm_loader = _make_norm_loader(xml_path, TARGET_Z, MAX_STEPS)
    vecnorm: VecNormalize = VecNormalize.load(norm_path, norm_loader)
    vecnorm.training = False      
    vecnorm.norm_reward = False  

    env = CrazyFlieEnvVelocity(
        xml_path=xml_path,
        target_z=TARGET_Z,
        max_steps=MAX_STEPS,
        n_stack=4,
        hover_required_steps=300,

        obs_noise_std=0.0,
        obs_bias_std=0.00,


        action_noise_std=0.0,


        motor_scale_std=0.0,

 
        frame_skip=10,
        frame_skip_jitter=0,

        start_xy_range=1,
        start_z_min=0.01,
        start_z_max=0.01,
        random_start=True,


        torque_bias_std=0.000,
        torque_gust_std=0.0,
        torque_gust_tau=0,   
         drag_lin_min=0.00,
         drag_lin_max=0.0,
         drag_quad_min=0.0,
         drag_quad_max=0.0,
         auto_landing=True
    )


    obs_raw, _ = env.reset()

    dt_sim = env.model.opt.timestep
    dt_step = dt_sim * env.frame_skip

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        terminated = False
        truncated = False
        ep_return = 0.0

        t0 = time.time()
        last_print = t0

        while not (terminated or truncated):
            obs_norm = vecnorm.normalize_obs(obs_raw[None, :])

            action, _ = model.predict(obs_norm, deterministic=True)


            action = np.asarray(action, dtype=np.float32)
            if action.ndim == 2:
                a = action[0]
            else:
                a = action


            obs_raw, reward, terminated, truncated, info = env.step(a)
            ep_return += float(reward)

  
            x, y, z = float(obs_raw[0]), float(obs_raw[1]), float(obs_raw[2])
            vx, vy, vz = float(obs_raw[7]), float(obs_raw[8]), float(obs_raw[9])

            # Sync the viewer
            v.sync()

            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                R = float(ep_return)
                print(
                    f"t={int(now - t0):2d}s | "
                    f"z={z:+.3f} m  vz={vz:+.3f} m/s  "
                    f"(x,y,z)=({x:+.3f},{y:+.3f},{z:+.3f})  "
                    f"Î”u={env.last_du:.3f}  "
                    f"R={R:.1f}  "
                    f"info={info}"
                )


            time.sleep(dt_step)

    print(
        f"\nEpisode finished. Return={ep_return:.2f}, "
        f"terminated={terminated}, truncated={truncated}, info={info}"
    )