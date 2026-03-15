import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from CrazyFlieEnvVelocity2 import CrazyFlieEnvVelocity


class SaveVecNormalizeOnBestCallback(BaseCallback):
    def __init__(self, vec_env, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.save_path = save_path

    def _on_step(self) -> bool:
        self.vec_env.save(self.save_path)
        if self.verbose:
            print(f"[SaveVecNorm] Saved -> {self.save_path}")
        return True


class DebugBehaviorCallback(BaseCallback):
    def __init__(self, eval_env, every_n_steps=200_000, n_episodes=5, verbose=0):
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
                ep_rew += float(reward[0])
                info0 = infos[0]
                last_info0 = info0
                z = float(self.eval_env.get_attr("get_altitude")[0]())
                tilt = float(info0.get("tilt_deg", 0.0))
                z_min = min(z_min, z)
                z_max = max(z_max, z)
                max_tilt = max(max_tilt, tilt)
                if "reason" in info0:
                    reason = info0["reason"]

            print(
                f"[DBG ep {i}] len={ep_len:4d} R={ep_rew:8.2f} "
                f"z_min={z_min:6.3f} z_max={z_max:6.3f} "
                f"max_tilt={max_tilt:6.2f}deg  hover={info0.get('hover_steps',0)}  reason={reason}"
            )

        print("=" * 52 + "\n")
        return True


def make_env(xml_path, target_z, max_steps, dr_params, rank=0):
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
        env.reset(seed=rank)
        return env
    return _f


if __name__ == "__main__":
    here         = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(here, "..", ".."))

    xml_path   = os.path.join(PROJECT_ROOT, "Assets", "bitcraze_crazyflie_2", "scene.xml")
    models_dir = os.path.join(PROJECT_ROOT, "models", "Velocity_Test_v3")
    logs_dir   = os.path.join(PROJECT_ROOT, "logs",   "Velocity_Test_v3")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir,   exist_ok=True)

    TARGET_Z   = 1.0   # ABSOLUTE target height (env now uses target_z_abs = target_z)
    MAX_STEPS  = 1000
    N_ENVS     = 16

    PHASE1_STEPS = 1_000_000
    TOTAL_STEPS  = 2_500_000

    # ================================================================
    # PHASE 1: Hover near target
    # spawn z in [0.60, 1.40] with ABSOLUTE target=1.0m
    # dz at spawn: -0.40 to +0.40m  (easy corrections)
    # No lateral randomization — focus purely on vertical hover
    # ================================================================
    DR_PHASE1 = dict(
        start_z_min     = 0.60,
        start_z_max     = 1.40,
        random_start    = True,
        start_xy_range  = 0.0,   # No lateral: learn vertical hover first

        obs_noise_std   = 0.0,
        obs_bias_std    = 0.0,
        action_noise_std= 0.0,
        motor_scale_std = 0.0,

        torque_bias_std = 0.0,
        torque_gust_std = 0.0,
        drag_lin_max    = 0.0,
        drag_quad_max   = 0.0,
        frame_skip_jitter = 0,
    )

    # ================================================================
    # PHASE 2: Full range including ground starts
    # Policy already knows how to hover; now learns takeoff
    # ================================================================
    DR_PHASE2 = dict(
        start_z_min     = 0.01,
        start_z_max     = 1.80,
        random_start    = True,
        start_xy_range  = 1.0,   # Lateral randomization added in Phase 2

        obs_noise_std   = 0.0,
        obs_bias_std    = 0.0,
        action_noise_std= 0.0,
        motor_scale_std = 0.0,

        torque_bias_std = 0.0,
        torque_gust_std = 0.0,
        drag_lin_max    = 0.0,
        drag_quad_max   = 0.0,
        frame_skip_jitter = 0,
    )

    # Eval/debug always use ground start — honest measure of full-task performance
    DR_EVAL = dict(
        start_z_min     = 0.01,
        start_z_max     = 0.01,
        random_start    = True,
        start_xy_range  = 0.5,

        obs_noise_std   = 0.0,
        obs_bias_std    = 0.0,
        action_noise_std= 0.0,
        motor_scale_std = 0.0,

        torque_bias_std = 0.0,
        torque_gust_std = 0.0,
        drag_lin_max    = 0.0,
        drag_quad_max   = 0.0,
        frame_skip_jitter = 0,
    )

    # ---- Phase 1 training envs ----
    train_fns  = [make_env(xml_path, TARGET_Z, MAX_STEPS, DR_PHASE1, rank=i) for i in range(N_ENVS)]
    train_venv = DummyVecEnv(env_fns=train_fns)
    train_venv = VecNormalize(train_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    train_venv.training = True

    # ---- Eval env (ground start) ----
    eval_fns  = [make_env(xml_path, TARGET_Z, MAX_STEPS, DR_EVAL, rank=1000 + i) for i in range(N_ENVS)]
    eval_venv = DummyVecEnv(env_fns=eval_fns)
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)
    eval_venv.obs_rms = train_venv.obs_rms

    # ---- Debug env (ground start) ----
    debug_fns  = [make_env(xml_path, TARGET_Z, MAX_STEPS, DR_EVAL, rank=9999)]
    debug_venv = DummyVecEnv(env_fns=debug_fns)
    debug_venv = VecNormalize(debug_venv, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)
    debug_venv.obs_rms = train_venv.obs_rms

    # ---- PPO ----
    model = PPO(
        "MlpPolicy",
        env             = train_venv,
        learning_rate   = 1e-4,
        clip_range      = 0.15,
        target_kl       = 0.02,
        n_steps         = 2048,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        ent_coef        = 0.001,
        vf_coef         = 0.5,
        policy_kwargs   = dict(net_arch=[64, 64]),
        tensorboard_log = logs_dir,
        verbose         = 1,
        device          = "auto",
    )

    best_norm_path = os.path.join(models_dir, "vecnormalize_best.pkl")
    save_norm_cb   = SaveVecNormalizeOnBestCallback(train_venv, best_norm_path, verbose=1)

    eval_cb = EvalCallback(
        eval_env             = eval_venv,
        best_model_save_path = models_dir,
        log_path             = logs_dir,
        eval_freq            = 100_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        render               = False,
        callback_on_new_best = save_norm_cb,
    )

    debug_cb = DebugBehaviorCallback(debug_venv, every_n_steps=200_000, n_episodes=5)
    callback = CallbackList([eval_cb, debug_cb])

    # ================================================================
    # PHASE 1 — hover learning (spawn near absolute target)
    # ================================================================
    sep = "=" * 62
    print(f"\n{sep}")
    print("  PHASE 1: Hover learning  (spawn z=[0.60, 1.40]m, target=1.0m abs)")
    print(f"  Steps: 0 -> {PHASE1_STEPS:,}")
    print(f"  Key: target_z_abs = {TARGET_Z}m ABSOLUTE (fixed), not spawn-relative")
    print(f"{sep}\n")

    model.learn(
        total_timesteps     = PHASE1_STEPS,
        progress_bar        = True,
        reset_num_timesteps = True,
        callback            = callback,
    )

    model.save(os.path.join(models_dir, "phase1.zip"))
    print("Phase 1 model saved.")

    # ================================================================
    # PHASE 2 — full range: ground starts + lateral drift
    # ================================================================
    print(f"\n{sep}")
    print("  PHASE 2: Full range  (spawn z=[0.01, 1.80]m, xy_range=1.0m)")
    print(f"  Steps: {PHASE1_STEPS:,} -> {TOTAL_STEPS:,}")
    print(f"{sep}\n")

    phase2_fns = [make_env(xml_path, TARGET_Z, MAX_STEPS, DR_PHASE2, rank=i) for i in range(N_ENVS)]
    train_venv_p2 = DummyVecEnv(env_fns=phase2_fns)
    train_venv_p2 = VecNormalize(train_venv_p2, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # Carry over normalization stats from Phase 1
    train_venv_p2.obs_rms   = train_venv.obs_rms
    train_venv_p2.ret_rms   = train_venv.ret_rms
    train_venv_p2.training  = True

    eval_venv.obs_rms       = train_venv_p2.obs_rms
    debug_venv.obs_rms      = train_venv_p2.obs_rms
    save_norm_cb.vec_env    = train_venv_p2

    model.set_env(train_venv_p2)

    model.learn(
        total_timesteps     = TOTAL_STEPS - PHASE1_STEPS,
        progress_bar        = True,
        reset_num_timesteps = False,
        callback            = callback,
    )

    final_path     = os.path.join(models_dir, "final_model.zip")
    final_norm_path = os.path.join(models_dir, "vecnormalize_final.pkl")
    model.save(final_path)
    train_venv_p2.save(final_norm_path)

    print(f"\n{sep}")
    print(f"  Training complete")
    print(f"  Final model -> {final_path}")
    print(f"  Final norm  -> {final_norm_path}")
    print(f"  Best norm   -> {best_norm_path}")
    print(f"{sep}\n")