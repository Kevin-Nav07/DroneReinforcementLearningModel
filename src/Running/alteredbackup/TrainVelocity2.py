"""
CrazyFlie PPO Hover — Velocity_AbsTarget_v4  (overnight run)

Built on v7 (z_obs = absolute) with the following improvements:
  ENV
    ff_k        0.8  → 1.5   stronger climb incentive from ground
    z_scale     1.0  → 0.60  steeper gradient near target
    vz_scale    0.30 → 0.45  avoids reward clipping at ground
    tilt_scale  20°  → 12°   stronger upright pressure
    w_vz        1.0  → 0.8   slightly reduced to balance scales
    att_kd      0.8  → 0.3   fixes derivative saturation at low angular rates
    w_smooth    0.02 → 0.05  more jerk penalty = better sim2real
    dense clip  -3   → -2    symmetric, standard for PPO
    stable tilt 15°  → 10°   tighter hover quality requirement
    progress reward            +0.3/T per step that reduced |dz|

  TRAINER
    TOTAL_STEPS         2M   → 10M   overnight
    HOVER_REQ           100  → 300   6s stable hover
    learning_rate       3e-4 → 1e-4→2e-5 linear decay
    clip_range          0.2  → 0.20→0.05 linear decay
    target_kl           0.05 → 0.02
    net_arch      [128,128]  → [64,64]
    eval_freq          100k  → 250k  (20 episodes each)
    debug callback           every 500k
    SyncedEvalCallback       deepcopies obs_rms before every eval round
    Spawn                    fully randomised [0.01, 1.80]m from step 0
    obs_rms sharing          deepcopy (not a reference)
"""
import copy
import os
import multiprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from CrazyFlieEnvVelocity2 import CrazyFlieEnvVelocity


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

class SaveVecNormOnBest(BaseCallback):
    def __init__(self, vec_env, path, verbose=0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.path    = path

    def _on_step(self):
        self.vec_env.save(self.path)
        if self.verbose:
            print(f"[SaveNorm] → {self.path}")
        return True


class SyncedEvalCallback(EvalCallback):
    """
    EvalCallback that deep-copies obs_rms from the training VecNormalize
    before every evaluation round, so eval normalisation stays in sync
    with training statistics over the full 10M-step run.
    """
    def __init__(self, train_venv: VecNormalize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_venv = train_venv

    def _on_step(self) -> bool:
        if self.eval_env is not None and hasattr(self.train_venv, "obs_rms"):
            self.eval_env.obs_rms = copy.deepcopy(self.train_venv.obs_rms)
        return super()._on_step()


class DebugCallback(BaseCallback):
    """
    Runs one episode per spawn height every N steps.
    All heights should eventually show SUCCESS once full-range training works.
    """
    def __init__(self, xml_path, target_z, max_steps, base_params,
                 every_n=500_000, n_ep=5):
        super().__init__()
        self.xml_path    = xml_path
        self.target_z    = target_z
        self.max_steps   = max_steps
        self.base_params = base_params
        self.every_n     = every_n
        self.spawn_heights = [0.01, 0.25, 0.50, 0.75, 1.00][:n_ep]

    def _on_step(self):
        if self.num_timesteps % self.every_n != 0:
            return True

        obs_rms = copy.deepcopy(getattr(self.model.env, "obs_rms", None))
        print(f"\n[DBG] Step {self.num_timesteps:,}")
        print(f"  {'spawn':>6}  {'len':>5}  {'R':>7}  {'z_max':>6}  {'tilt':>6}  result")

        for spawn_z in self.spawn_heights:
            p = dict(**self.base_params, start_z_min=spawn_z, start_z_max=spawn_z)
            env  = CrazyFlieEnvVelocity(self.xml_path, self.target_z, self.max_steps,
                                         n_stack=4, **p)
            venv = VecNormalize(DummyVecEnv([lambda e=Monitor(env): e]),
                                norm_obs=True, norm_reward=False,
                                training=False, clip_obs=10.0)
            if obs_rms is not None:
                venv.obs_rms = obs_rms

            obs = venv.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            ep_r, ep_len, z_max, max_tilt = 0.0, 0, -1e9, 0.0
            reason, done, last_info = "timeout", [False], {}

            while not done[0]:
                act, _ = self.model.predict(obs, deterministic=True)
                obs, rew, done, infos = venv.step(act)
                ep_r      += float(rew[0])
                ep_len    += 1
                z          = float(venv.get_attr("get_altitude")[0]())
                z_max      = max(z_max, z)
                max_tilt   = max(max_tilt, float(infos[0].get("tilt_deg", 0)))
                last_info  = infos[0]
                if "reason" in infos[0]:
                    reason = infos[0]["reason"]

            venv.close()
            result = "SUCCESS!" if last_info.get("success") else reason
            print(f"  {spawn_z:>6.2f}  {ep_len:>5d}  {ep_r:>7.2f}  {z_max:>6.2f}  "
                  f"{max_tilt:>5.1f}°  {result}")
        print()
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Env factory
# ──────────────────────────────────────────────────────────────────────────────

def make_env(xml_path, target_z, max_steps, params, rank=0):
    def _f():
        env = CrazyFlieEnvVelocity(xml_path, target_z, max_steps, n_stack=4, **params)
        return Monitor(env)
    return _f


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.freeze_support()

    here         = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(here, "..", ".."))
    xml_path     = os.path.join(PROJECT_ROOT, "Assets", "bitcraze_crazyflie_2", "scene.xml")
    models_dir   = os.path.join(PROJECT_ROOT, "models", "Velocity_AbsTarget_v4")
    logs_dir     = os.path.join(PROJECT_ROOT, "logs",   "Velocity_AbsTarget_v4")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir,   exist_ok=True)

    TARGET_Z    = 1.0
    MAX_STEPS   = 1000
    TOTAL_STEPS = 6_000_000
    N_ENVS      = 16
    HOVER_REQ   = 300

    # All DR off — clean baseline
    SHARED = dict(
        hover_required_steps=HOVER_REQ,
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

    # Full random spawn from step 0 — policy sees all heights every episode
    TRAIN = dict(**SHARED, start_z_min=0.01, start_z_max=1.80,
                 start_xy_range=0.30, safety_radius=4.0)

    # Eval always from ground — hardest test
    EVAL  = dict(**SHARED, start_z_min=0.01, start_z_max=0.01,
                 start_xy_range=0.0, safety_radius=4.0)

    # ── Vectorised envs ───────────────────────────────────────────────────────
    train_venv = VecNormalize(
        SubprocVecEnv([make_env(xml_path, TARGET_Z, MAX_STEPS, TRAIN, i)
                       for i in range(N_ENVS)], start_method="spawn"),
        norm_obs=True, norm_reward=False, clip_obs=10.0)
    train_venv.training = True

    eval_venv = VecNormalize(
        SubprocVecEnv([make_env(xml_path, TARGET_Z, MAX_STEPS, EVAL, 1000+i)
                       for i in range(N_ENVS)], start_method="spawn"),
        norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)
    eval_venv.obs_rms = copy.deepcopy(train_venv.obs_rms)

    # ── Schedules ─────────────────────────────────────────────────────────────
    def lr_schedule(progress_remaining: float) -> float:
        return 2e-5 + progress_remaining * (1e-4 - 2e-5)

    def clip_schedule(progress_remaining: float) -> float:
        return 0.05 + progress_remaining * (0.20 - 0.05)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        env             = train_venv,
        learning_rate   = lr_schedule,
        clip_range      = clip_schedule,
        target_kl       = 0.02,
        n_steps         = 2048,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        ent_coef        = 0.005,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = dict(net_arch=[64, 64]),
        tensorboard_log = logs_dir,
        verbose         = 1,
        device          = "auto",
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_norm    = os.path.join(models_dir, "vecnormalize_best.pkl")
    save_norm_cb = SaveVecNormOnBest(train_venv, best_norm, verbose=1)

    eval_cb = SyncedEvalCallback(
        train_venv=train_venv,
        eval_env=eval_venv,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=250_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        callback_on_new_best=save_norm_cb,
    )

    dbg_cb = DebugCallback(
        xml_path, TARGET_Z, MAX_STEPS, SHARED,
        every_n=500_000,
    )

    # ── Banner ────────────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("  CrazyFlie PPO Hover — Velocity_AbsTarget_v4 (overnight)")
    print(f"  z_obs=absolute | ff_k=1.5 | z_scale=0.60 | tilt_scale=12° | att_kd=0.3")
    print(f"  {TOTAL_STEPS:,} steps | LR=1e-4→2e-5 | clip=0.20→0.05 | kl=0.02 | net=[64,64]")
    print(f"  Spawn fully randomised: z=[0.01, 1.80]m  xy=±0.30m — no curriculum")
    print(f"  hover_req={HOVER_REQ} steps | eval every 250k (20 eps) | debug every 500k")
    print(f"{sep}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = TOTAL_STEPS,
        progress_bar        = True,
        reset_num_timesteps = True,
        callback            = CallbackList([eval_cb, dbg_cb]),
    )

    model.save(os.path.join(models_dir, "final_model.zip"))
    train_venv.save(os.path.join(models_dir, "vecnormalize_final.pkl"))
    print(f"Done → {models_dir}")
    train_venv.close()
    eval_venv.close()