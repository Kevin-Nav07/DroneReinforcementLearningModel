"""
CrazyFlie PPO Hover — TrainWarmStartLong.py  (40M steps — extended consolidation)

Same as TrainWarmStart.py but trains for 40M steps instead of 20M, with the
extra 20M used as additional consolidation at HARD DR.

WHY THIS INSTEAD OF TrainWarmStartContinue.py:
  Loading a fully-trained PPO model with new hyperparameters caused immediate
  policy collapse (KL early-stopping at step 0, std dropping from 8.6 to 4.0,
  ep_rew_mean -2.4 from step 1). The optimizer state reset combined with new
  LR/clip/ent values tore the policy apart before any learning could happen.

  The cleanest fix is to extend a single training run rather than chain runs —
  the optimizer state stays consistent throughout, the DR curriculum properly
  ramps from the warm-start, and the value function has time to recalibrate.

Warm-starts from Velocity_Final/best_model.zip and pushes domain
randomization BEYOND the eval-proven values for maximum sim2real robustness.

──────────────────────────────────────────────────────────────────────────────
LESSONS APPLIED FROM PREVIOUS RUNS
──────────────────────────────────────────────────────────────────────────────
  v1 warm-start (failed): LR=5e-6 was too low. clip_fraction <0.01 in the
    final 3M steps means essentially zero learning. Fixed: LR=5e-5→1e-5.

  torque_bias permanently excluded: TestTorqueBias.py confirmed that even
    10 µN·m flips the drone — actuator gear=0.00001 makes any external
    body torque uncompensatable. The real drone handles rotor imbalance
    through its firmware PID, not the RL policy.

  tilt_min_z=0.20: below this the 25° init tilt is physically unrecoverable.
    Eval spawns below 0.20m with tilt are impossible and not a model failure.

  max() in MultiStageDRCallback: later stages don't zero out earlier ones
    when their ramp starts at scale=0.

  EVAL always at zero DR: prevents noise variance from corrupting
    best_model selection.

──────────────────────────────────────────────────────────────────────────────
DR ESCALATION (beyond eval-proven values)
──────────────────────────────────────────────────────────────────────────────
                        nominal   eval-proven   HARD (this run)
  obs_noise_std           0.01       0.06         0.10  (1.67× eval)
  obs_bias_std            0.005      0.07         0.12  (1.71× eval)
  motor_scale_std         0.03       0.20         0.25  (1.25× eval)
  drag_lin_max            0.05       0.10         0.15  (1.50× eval)
  drag_quad_max           0.0        0.0          0.05  NEW
  action_noise_std        0.0        0.0          0.02  NEW
  frame_skip_jitter       0          2            2     (same as eval)

──────────────────────────────────────────────────────────────────────────────
CURRICULUM
──────────────────────────────────────────────────────────────────────────────
   0 –  2M : anchor at nominal SHARED DR     (let policy settle from warm-start)
   2M–  6M : ramp to eval-proven DR          (confirmed working in earlier eval)
   6M– 10M : ramp further to HARD DR         (beyond eval for sim2real margin)
  10M– 40M : 30M consolidation at HARD DR    (extended fine-tuning)

  LR:         5e-5 → 1e-5  (10× higher than v1 warm-start — actually learns)
  clip:       0.10 → 0.05  (larger early updates, conservative late)
  target_kl:  0.015
  ENT:        0.005 → 0.001
"""
import copy
import os
import multiprocessing
import pickle

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from CrazyFlieEnvVelocity2 import CrazyFlieEnvVelocity


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

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


class EntropyCoefficientSchedule(BaseCallback):
    def __init__(self, total_timesteps, ent_start=0.005, ent_end=0.001):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.ent_start = ent_start
        self.ent_end   = ent_end

    def _on_step(self):
        progress = self.num_timesteps / max(1, self.total_timesteps)
        self.model.ent_coef = float(
            self.ent_start + (self.ent_end - self.ent_start) * progress
        )
        return True


class SyncedEvalCallback(EvalCallback):
    def __init__(self, train_venv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_venv = train_venv

    def _on_step(self):
        if self.eval_env is not None and hasattr(self.train_venv, "obs_rms"):
            self.eval_env.obs_rms = copy.deepcopy(self.train_venv.obs_rms)
        return super()._on_step()


class MultiStageDRCallback(BaseCallback):
    """
    Linear-ramp DR curriculum with max() semantics so later stages
    never zero out earlier settled params.
    """
    def __init__(self, train_venv, stages, verbose=1):
        super().__init__(verbose)
        self.train_venv   = train_venv
        self.stages       = stages
        self._cache       = {}
        self._last_bucket = -1

    def _current_values(self):
        t = self.num_timesteps
        vals = {}
        for start, end, targets in self.stages:
            if t < start:
                scale = 0.0
            elif t >= end:
                scale = 1.0
            else:
                scale = (t - start) / max(1, end - start)
            for param, target in targets.items():
                vals[param] = max(vals.get(param, 0.0), float(target * scale))
        return vals

    def _on_step(self):
        vals = self._current_values()
        for param, val in vals.items():
            if abs(val - self._cache.get(param, -1e9)) > 1e-9:
                self.train_venv.set_attr(param, val)
                self._cache[param] = val
        t = self.num_timesteps
        bucket = t // 500_000
        if self.verbose and bucket != self._last_bucket:
            self._last_bucket = bucket
            display = {k: round(v, 6) for k, v in vals.items()}
            print(f"[MultiStageDR] step={t:,}  {display}")
        return True


class DebugCallback(BaseCallback):
    """
    CLEAN + TILT25 suites every N steps. Always zero DR for comparability.
    Tilt heights start at 0.20m to match tilt_min_z — no impossible spawns.
    """
    def __init__(self, xml_path, target_z, max_steps, base_params,
                 every_n=500_000, n_ep=5):
        super().__init__()
        self.xml_path    = xml_path
        self.target_z    = target_z
        self.max_steps   = max_steps
        self.base_params = {k: v for k, v in base_params.items()
                            if k != "init_tilt_max_deg"}
        self.every_n       = every_n
        self.clean_heights = [0.01, 0.25, 0.50, 0.75, 1.00][:n_ep]
        self.tilt_heights  = [0.20, 0.50, 0.75, 1.00, 1.50][:n_ep]

    def _run_suite(self, obs_rms, tilt_deg, label, heights):
        print(f"  [{label}]  init_tilt_max_deg={tilt_deg:.1f}°")
        print(f"  {'spawn':>6}  {'len':>5}  {'R':>7}  {'z_max':>6}  {'tilt':>6}  result")
        for spawn_z in heights:
            p = {
                **self.base_params,
                "start_z_min":        spawn_z,
                "start_z_max":        spawn_z,
                "init_tilt_max_deg":  tilt_deg,
                "obs_noise_std":      0.0,
                "obs_bias_std":       0.0,
                "motor_scale_std":    0.0,
                "action_noise_std":   0.0,
                "torque_bias_std":    0.0,
                "torque_gust_std":    0.0,
                "drag_lin_max":       0.0,
                "drag_quad_max":      0.0,
                "frame_skip_jitter":  0,
            }
            env  = CrazyFlieEnvVelocity(self.xml_path, self.target_z,
                                         self.max_steps, n_stack=4, **p)
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
                ep_r    += float(rew[0])
                ep_len  += 1
                z        = float(venv.get_attr("get_altitude")[0]())
                z_max    = max(z_max, z)
                max_tilt = max(max_tilt, float(infos[0].get("tilt_deg", 0)))
                last_info = infos[0]
                if "reason" in infos[0]:
                    reason = infos[0]["reason"]
            venv.close()
            result = "SUCCESS!" if last_info.get("success") else reason
            print(f"  {spawn_z:>6.2f}  {ep_len:>5d}  {ep_r:>7.2f}  {z_max:>6.2f}  "
                  f"{max_tilt:>5.1f}°  {result}")

    def _on_step(self):
        if self.num_timesteps % self.every_n != 0:
            return True
        obs_rms = copy.deepcopy(getattr(self.model.env, "obs_rms", None))
        print(f"\n[DBG] Step {self.num_timesteps:,}")
        self._run_suite(obs_rms, tilt_deg=0.0,  label="CLEAN",
                        heights=self.clean_heights)
        print()
        self._run_suite(obs_rms, tilt_deg=25.0, label="TILT ",
                        heights=self.tilt_heights)
        print()
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Env factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(xml_path, target_z, max_steps, params, rank=0):
    def _f():
        env = CrazyFlieEnvVelocity(xml_path, target_z, max_steps,
                                    n_stack=4, **params)
        return Monitor(env)
    return _f


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.freeze_support()

    here         = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(here, "..", ".."))
    xml_path   = os.path.join(PROJECT_ROOT, "Assets", "bitcraze_crazyflie_2", "scene.xml")

    # Source: Velocity_Final best_model
    src_models_dir = os.path.join(PROJECT_ROOT, "models", "Velocity_Final")
    src_model_path = os.path.join(src_models_dir, "best_model.zip")
    src_norm_path  = os.path.join(src_models_dir, "vecnormalize_best.pkl")

    # Output
    models_dir = os.path.join(PROJECT_ROOT, "models", "Velocity_WarmStart_HardDR_Long")
    logs_dir   = os.path.join(PROJECT_ROOT, "logs",   "Velocity_WarmStart_HardDR_Long")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir,   exist_ok=True)

    TARGET_Z    = 1.0
    MAX_STEPS   = 1000
    TOTAL_STEPS = 40_000_000
    N_ENVS      = 16
    HOVER_REQ   = 300

    # ── DR curriculum ─────────────────────────────────────────────────────────
    #
    #  0  –  2M : anchor at SHARED (nominal — let policy settle from warm-start)
    #  2M –  6M : ramp to eval-proven DR (obs=0.06, bias=0.07, motor=0.20, drag=0.10)
    #  2M –  4M : ramp frame_skip_jitter 0 → 2
    #  6M – 10M : ramp further to HARD DR (beyond eval for sim2real margin)
    #             obs=0.10, bias=0.12, motor=0.25, drag=0.15, drag_quad=0.05,
    #             action_noise=0.02
    # 10M – 40M : 30M consolidation at HARD DR (extended)
    DR_STAGES = [
        # Phase 1 — ramp to eval-proven values (known good)
        (2_000_000, 6_000_000, {
            "obs_noise_std":   0.06,
            "obs_bias_std":    0.07,
            "motor_scale_std": 0.20,
            "drag_lin_max":    0.10,
        }),
        # Frame skip jitter ramps in parallel
        (2_000_000, 4_000_000, {
            "frame_skip_jitter": 3,
        }),
        # Phase 2 — push BEYOND eval for sim2real headroom
        (6_000_000, 10_000_000, {
            "obs_noise_std":    0.10,    # 1.67× eval
            "obs_bias_std":     0.12,    # 1.71× eval
            "motor_scale_std":  0.25,    # 1.25× eval
            "drag_lin_max":     0.15,    # 1.50× eval
            "drag_quad_max":    0.05,    # NEW — quadratic drag
            "action_noise_std": 0.02,    # NEW — actuator command noise
        }),
    ]

    # ── Shared env params ─────────────────────────────────────────────────────
    SHARED = dict(
        hover_required_steps = HOVER_REQ,
        auto_landing         = False,
        init_tilt_max_deg    = 25.0,
        tilt_min_z           = 0.20,
        # Sensor DR — nominal from step 0, ramped harder by curriculum
        obs_noise_std        = 0.01,
        obs_bias_std         = 0.005,
        action_noise_std     = 0.0,
        motor_scale_std      = 0.03,
        # Torque — PERMANENTLY EXCLUDED (physically untrainable in this sim)
        torque_bias_std      = 0.0,
        torque_gust_std      = 0.0,
        torque_gust_tau      = 1.5,
        # Drag — nominal from step 0, ramped harder by curriculum
        drag_lin_min         = 0.0,
        drag_lin_max         = 0.05,
        drag_quad_min        = 0.0,
        drag_quad_max        = 0.0,
        # Frame skip jitter — ramped 0→2 by curriculum
        frame_skip_jitter    = 0,
    )

    TRAIN = dict(**SHARED,
                 start_z_min=0.01, start_z_max=1.80,
                 start_xy_range=1.0, safety_radius=2.0)

    # Eval always clean — stable best_model selection
    EVAL = {
        **SHARED,
        "init_tilt_max_deg":  0.0,
        "obs_noise_std":      0.0,
        "obs_bias_std":       0.0,
        "motor_scale_std":    0.0,
        "action_noise_std":   0.0,
        "torque_bias_std":    0.0,
        "drag_lin_max":       0.0,
        "drag_quad_max":      0.0,
        "frame_skip_jitter":  0,
        "start_z_min":        0.01,
        "start_z_max":        0.01,
        "start_xy_range":     0.0,
        "safety_radius":      2.0,
    }

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

    # ── Load Velocity_Final ───────────────────────────────────────────────────
    if not os.path.exists(src_model_path):
        raise FileNotFoundError(
            f"Source model not found: {src_model_path}\n"
            "Run TrainFromScratch.py first, or point to an existing checkpoint."
        )
    if os.path.exists(src_norm_path):
        with open(src_norm_path, "rb") as f:
            src_norm = pickle.load(f)
        train_venv.obs_rms = src_norm.obs_rms
        eval_venv.obs_rms  = copy.deepcopy(src_norm.obs_rms)
        print(f"[WarmStart] Loaded obs_rms from {src_norm_path}")

    # ── Schedules ─────────────────────────────────────────────────────────────
    def lr_schedule(progress_remaining: float) -> float:
        return 1e-5 + progress_remaining * (5e-5 - 1e-5)

    def clip_schedule(progress_remaining: float) -> float:
        return 0.05 + progress_remaining * (0.10 - 0.05)

    ENT_START = 0.005
    ENT_END   = 0.001

    # ── Load model ────────────────────────────────────────────────────────────
    model = PPO.load(
        src_model_path,
        env           = train_venv,
        learning_rate = lr_schedule,
        clip_range    = clip_schedule,
        ent_coef      = ENT_START,
        device        = "auto",
        verbose       = 1,
    )
    model.tensorboard_log = logs_dir
    model.target_kl       = 0.015
    print(f"[WarmStart] Loaded model from {src_model_path}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_norm_path = os.path.join(models_dir, "vecnormalize_best.pkl")
    save_norm_cb   = SaveVecNormOnBest(train_venv, best_norm_path, verbose=1)

    eval_cb = SyncedEvalCallback(
        train_venv           = train_venv,
        eval_env             = eval_venv,
        best_model_save_path = models_dir,
        log_path             = logs_dir,
        eval_freq            = 200_000,
        n_eval_episodes      = 20,
        deterministic        = True,
        render               = False,
        callback_on_new_best = save_norm_cb,
    )
    ent_cb = EntropyCoefficientSchedule(TOTAL_STEPS, ENT_START, ENT_END)
    dr_cb  = MultiStageDRCallback(train_venv, DR_STAGES, verbose=1)
    dbg_cb = DebugCallback(xml_path, TARGET_Z, MAX_STEPS, SHARED,
                           every_n=500_000)

    # ── Banner ────────────────────────────────────────────────────────────────
    sep = "=" * 78
    print(f"\n{sep}")
    print("  CrazyFlie PPO — TrainWarmStartLong.py  (HARD DR, 40M)")
    print(f"  Source: Velocity_Final/best_model.zip")
    print(f"  Curriculum:")
    print(f"     0 –  2M : anchor at nominal DR  (let warm-start settle)")
    print(f"     2M–  6M : ramp to eval-proven DR  (obs=0.06 bias=0.07 motor=0.20 drag=0.10)")
    print(f"     6M– 10M : ramp to HARD DR  (obs=0.10 bias=0.12 motor=0.25 drag=0.15")
    print(f"                                 +drag_quad=0.05 +action_noise=0.02)")
    print(f"    10M– 40M : 30M consolidation at HARD DR (extended)")
    print(f"  LR 5e-5→1e-5 | clip 0.10→0.05 | kl=0.015 | ENT 0.005→0.001")
    print(f"  Torque bias EXCLUDED (TestTorqueBias.py confirmed physically untrainable)")
    print(f"{sep}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps     = TOTAL_STEPS,
        progress_bar        = True,
        reset_num_timesteps = True,
        callback            = CallbackList([eval_cb, ent_cb, dr_cb, dbg_cb]),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(os.path.join(models_dir, "warmstart_hard_final.zip"))
    train_venv.save(os.path.join(models_dir, "vecnormalize_final.pkl"))
    print(f"\nDone → {models_dir}")
    print("  warmstart_hard_final.zip  — deploy this")
    print("  vecnormalize_best.pkl     — use for evaluation")
    train_venv.close()
    eval_venv.close()