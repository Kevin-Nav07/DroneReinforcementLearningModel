import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

##IMPORTANT: import env from src/Environments
from CrazyFlieEnvVelocity2 import CrazyFlieEnvVelocity


class SaveVecNormalizeOnBestCallback(BaseCallback):
    def __init__(self, vec_env, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.save_path = save_path

    def _on_step(self) -> bool:
        ##save current normalization stats (obs_rms, etc.)
        self.vec_env.save(self.save_path)
        if self.verbose:
            print(f"[SaveVecNormalizeOnBest] Saved VecNormalize to: {self.save_path}")
        return True


class DebugBehaviorCallback(BaseCallback):
    def __init__(self, eval_env, every_n_steps=100_000, n_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.every_n_steps = every_n_steps
        self.n_episodes = n_episodes

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every_n_steps != 0:
            return True

        print(f"\n[DebugBehaviorCallback] Step {self.num_timesteps}, running {self.n_episodes} debug episodes...\n")
        for i in range(self.n_episodes):
            obs = self.eval_env.reset()
            ##VecEnv reset returns obs, but keep guard just in case
            if isinstance(obs, tuple):
                obs = obs[0]

            ep_len = 0
            ep_rew = 0.0
            z_min, z_max = 1e9, -1e9
            max_tilt = 0.0
            reason = "timeout"

            done = [False]
            last_info0 = {}

            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, infos = self.eval_env.step(action)

                ep_len += 1

                r0 = float(reward[0])
                info0 = infos[0]
                last_info0 = info0

                ep_rew += r0
                z = float(self.eval_env.get_attr("get_altitude")[0]())
                tilt = float(info0.get("tilt_deg", 0.0))

                z_min = min(z_min, z)
                z_max = max(z_max, z)
                max_tilt = max(max_tilt, tilt)

                if "reason" in info0:
                    reason = info0["reason"]

            print(
                f"[DBG ep {i}] len={ep_len:4d} R={ep_rew:8.1f} "
                f"z_min={z_min:6.3f} z_max={z_max:6.3f} "
                f"max_tilt={max_tilt:6.2f} deg  reason={reason}  info={last_info0}"
            )

        print("=" * 48 + "\n")
        return True


def make_env(xml_path: str, target_z: float, max_steps: int, dr_params: dict, rank: int = 0):
    """
    Factory that creates one CrazyFlieEnvVelocity wrapped in a Monitor.
    Used by DummyVecEnv to build N env instances.
    """
    def _f():
        env = CrazyFlieEnvVelocity(
            xml_path=xml_path,
            target_z=target_z,
            max_steps=max_steps,
            n_stack=4,
            hover_required_steps=300,
            auto_landing=False,
            **dr_params,
        )
        env = Monitor(env)
        env.reset(seed=rank)  ##seed once so RNG stream is consistent
        return env

    return _f


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    ##src/Training -> src -> project root
    PROJECT_ROOT = os.path.abspath(os.path.join(here, "..", ".."))

    ##Domain randomization parameters (Stage-1: mild/none)
    DR_PARAMS = dict(
        start_z_min=0.0,
        start_z_max=0.0,
        random_start=True,

        obs_noise_std=0.0,
        obs_bias_std=0.0,
        action_noise_std=0.0,
        motor_scale_std=0.0,

        start_xy_range=1.0,
        torque_bias_std=0.0,
        torque_gust_std=0.0,

        drag_lin_max=0.0,
        drag_quad_max=0.0,

        frame_skip_jitter=0,
    )

    ##Paths
    xml_path = os.path.join(PROJECT_ROOT, "Assets", "bitcraze_crazyflie_2", "scene.xml")
    models_dir = os.path.join(PROJECT_ROOT, "models", "Velocity_Test")
    logs_dir = os.path.join(PROJECT_ROOT, "logs", "Velocity_Test")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    ##Env constants
    TARGET_Z = 1.0
    MAX_STEPS = 1000
    N_ENVS = 16

    ##Build training env
    env_fns = [make_env(xml_path, TARGET_Z, MAX_STEPS, DR_PARAMS, rank=i) for i in range(N_ENVS)]
    train_venv = DummyVecEnv(env_fns=env_fns)

    ##Normalize OBS only (reward is already clipped/normalized in env)
    train_venv = VecNormalize(
        train_venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    train_venv.training = True

    ##Build eval env (separate, no training updates)
    eval_env_fns = [make_env(xml_path, TARGET_Z, MAX_STEPS, DR_PARAMS, rank=1000 + i) for i in range(N_ENVS)]
    eval_venv = DummyVecEnv(env_fns=eval_env_fns)
    eval_venv = VecNormalize(
        eval_venv,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )

    ##Share obs normalization stats so evaluation uses the same scaling
    eval_venv.obs_rms = train_venv.obs_rms

    ##Create PPO model
    model = PPO(
        "MlpPolicy",
        env=train_venv,
        learning_rate=1e-4,
        clip_range=0.15,
        target_kl=0.02,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log=logs_dir,
        verbose=1,
    )

    ##Save VecNormalize whenever we get a new best model
    best_vecnorm_path = os.path.join(models_dir, "vecnormalize_best.pkl")
    save_vecnorm_cb = SaveVecNormalizeOnBestCallback(train_venv, best_vecnorm_path, verbose=1)

    ##Eval callback
    eval_callback = EvalCallback(
        eval_env=eval_venv,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=100_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_on_new_best=save_vecnorm_cb,
    )

    ##Debug env (single env is enough)
    debug_env_fns = [make_env(xml_path, TARGET_Z, MAX_STEPS, DR_PARAMS, rank=9999)]
    debug_venv = DummyVecEnv(env_fns=debug_env_fns)
    debug_venv = VecNormalize(
        debug_venv,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )
    debug_venv.obs_rms = train_venv.obs_rms

    debug_cb = DebugBehaviorCallback(debug_venv, every_n_steps=200_000, n_episodes=5)
    callback = CallbackList([eval_callback, debug_cb])

    model.learn(
        total_timesteps=2_000_000,
        progress_bar=True,
        reset_num_timesteps=True,
        callback=callback,
    )

    ##Save final model + final VecNormalize stats
    stage1_model_path = os.path.join(models_dir, "stage1.zip")
    stage1_vecnorm_path = os.path.join(models_dir, "vecnormalize_final.pkl")

    model.save(stage1_model_path)
    train_venv.save(stage1_vecnorm_path)

    print(f"Stage-1 model saved to: {stage1_model_path}")
    print(f"Stage-1 VecNormalize stats saved to: {stage1_vecnorm_path}")
    print(f"Best-model VecNormalize stats saved to: {best_vecnorm_path}")