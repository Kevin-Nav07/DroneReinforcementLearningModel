"""
Callbacks and helpers shared by train_from_scratch.py and train_warmstart.py.

These were copy-pasted into every training script before the cleanup. They now
live here so a fix to one (for example the DR curriculum) applies to both.

A Stable-Baselines3 callback is an object whose _on_step() method SB3 calls
after every environment step during model.learn(). Returning False stops
training; returning True continues.
"""
import copy
import os

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.sim_env import CrazyFlieEnvVelocity

# Every domain randomisation (DR) setting in the sim env. Used to switch them
# all off for clean evaluation and debug episodes.
ZERO_DR = dict(
    obs_noise_std=0.0,
    obs_bias_std=0.0,
    motor_scale_std=0.0,
    action_noise_std=0.0,
    torque_bias_std=0.0,
    torque_gust_std=0.0,
    drag_lin_max=0.0,
    drag_quad_max=0.0,
    frame_skip_jitter=0,
)


def make_env(xml_path, target_z, max_steps, params):
    """
    Return a function that builds one Monitor-wrapped env. SubprocVecEnv needs a
    function (not an env) because each env is built inside its own process.
    """
    def _f():
        env = CrazyFlieEnvVelocity(xml_path, target_z, max_steps, n_stack=4, **params)
        return Monitor(env)  # Monitor records episode reward and length for the logs
    return _f


def linear_schedule(start, end):
    """
    Build an SB3 schedule that moves linearly from start to end over training.
    SB3 calls it with progress_remaining, which goes from 1.0 (start) to 0.0 (end).
    """
    def _schedule(progress_remaining):
        return end + progress_remaining * (start - end)
    return _schedule


def refuse_to_overwrite(models_dir, overwrite):
    """
    Stop before training if the output folder already holds a trained model.
    Two old scripts wrote to the same folder, so one could silently replace the
    other's best model. This makes that impossible without --overwrite.
    """
    if os.path.exists(os.path.join(models_dir, "best_model.zip")) and not overwrite:
        raise SystemExit(
            f"{models_dir} already contains best_model.zip.\n"
            "Pick a new --run-name, or pass --overwrite if you really want to replace it."
        )


class SaveVecNormOnBest(BaseCallback):
    """
    Saves the observation normaliser whenever EvalCallback finds a new best model,
    so best_model.zip always has a matching vecnormalize_best.pkl.
    """
    def __init__(self, vec_env, path, verbose=0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.path = path

    def _on_step(self):
        self.vec_env.save(self.path)
        if self.verbose:
            print(f"[SaveNorm] -> {self.path}")
        return True


class EntropyCoefficientSchedule(BaseCallback):
    """
    SB3 only accepts schedules for learning_rate and clip_range, not ent_coef.
    This callback decays ent_coef linearly from ent_start to ent_end by setting
    model.ent_coef directly every step. Higher entropy early means more
    exploration; lower entropy late lets the policy settle.
    """
    def __init__(self, total_timesteps, ent_start, ent_end):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.ent_start = ent_start
        self.ent_end = ent_end

    def _on_step(self):
        progress = self.num_timesteps / max(1, self.total_timesteps)  # 0.0 at start, 1.0 at end
        self.model.ent_coef = float(self.ent_start + (self.ent_end - self.ent_start) * progress)
        return True


class SyncedEvalCallback(EvalCallback):
    """
    EvalCallback that copies the training normaliser's running mean and variance
    into the eval env before each evaluation. Without this the eval env would
    normalise observations with stale statistics and score the policy unfairly.
    """
    def __init__(self, train_venv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_venv = train_venv

    def _on_step(self):
        if self.eval_env is not None and hasattr(self.train_venv, "obs_rms"):
            self.eval_env.obs_rms = copy.deepcopy(self.train_venv.obs_rms)  # a copy, so eval cannot alter training stats
        return super()._on_step()


class MultiStageDRCallback(BaseCallback):
    """
    Domain randomisation curriculum. Each stage is (start_step, end_step, targets):
    between start and end every listed parameter ramps linearly from 0 to its
    target. When stages overlap, the larger value wins (max), so a later stage
    never resets an earlier one back to zero.

    base_values are the env's own starting values (the trainers' SHARED dict).
    A parameter never goes below its base value. Before this was added, a stage
    that had not started yet computed target x 0 = 0 and pushed that to the envs,
    silently switching off the "nominal" DR the trainers asked for at step 0.
    The original HardDR_Long run was affected for its first 2M steps.
    """
    def __init__(self, train_venv, stages, base_values=None, verbose=1):
        super().__init__(verbose)
        self.train_venv = train_venv
        self.stages = stages
        self.base_values = dict(base_values or {})
        self._cache = {}          # last value sent to the envs, to avoid resending every step
        self._last_bucket = -1    # used to print progress every 500k steps

    def _current_values(self):
        """Work out every DR parameter's value at the current timestep."""
        t = self.num_timesteps
        # Start each curriculum parameter at its base value (0 if it has none)
        vals = {param: float(self.base_values.get(param, 0.0))
                for _, _, targets in self.stages for param in targets}
        for start, end, targets in self.stages:
            if t < start:
                scale = 0.0
            elif t >= end:
                scale = 1.0
            else:
                scale = (t - start) / max(1, end - start)
            for param, target in targets.items():
                vals[param] = max(vals[param], float(target * scale))
        return vals

    def _on_step(self):
        vals = self._current_values()
        for param, val in vals.items():
            if abs(val - self._cache.get(param, -1e9)) > 1e-9:
                self.train_venv.set_attr(param, val)  # sets the attribute inside every subprocess env
                self._cache[param] = val
        bucket = self.num_timesteps // 500_000
        if self.verbose and bucket != self._last_bucket:
            self._last_bucket = bucket
            print(f"[MultiStageDR] step={self.num_timesteps:,}  {({k: round(v, 6) for k, v in vals.items()})}")
        return True


class DebugCallback(BaseCallback):
    """
    Every every_n steps, flies a few deterministic episodes with all DR off and
    prints one line per episode: a CLEAN suite (upright spawns at several
    heights) and a TILT suite (up to 25 deg lean at spawn). This shows progress
    in the console between the less frequent EvalCallback runs.
    """
    def __init__(self, xml_path, target_z, max_steps, base_params, every_n=500_000, n_ep=5):
        super().__init__()
        self.xml_path = xml_path
        self.target_z = target_z
        self.max_steps = max_steps
        # Removed so each suite can set its own tilt without a duplicate-key error
        self.base_params = {k: v for k, v in base_params.items() if k != "init_tilt_max_deg"}
        self.every_n = every_n
        self.clean_heights = [0.01, 0.25, 0.50, 0.75, 1.00][:n_ep]
        self.tilt_heights = [0.20, 0.50, 0.75, 1.00, 1.50][:n_ep]  # starts at tilt_min_z: lower tilted spawns are unrecoverable

    def _run_suite(self, obs_rms, tilt_deg, label, heights):
        """Fly one episode per spawn height and print the result."""
        print(f"  [{label}]  init_tilt_max_deg={tilt_deg:.1f}")
        print(f"  {'spawn':>6}  {'len':>5}  {'R':>7}  {'z_max':>6}  {'tilt':>6}  result")
        for spawn_z in heights:
            p = {**self.base_params, **ZERO_DR,
                 "start_z_min": spawn_z, "start_z_max": spawn_z, "init_tilt_max_deg": tilt_deg}
            env = CrazyFlieEnvVelocity(self.xml_path, self.target_z, self.max_steps, n_stack=4, **p)
            venv = VecNormalize(DummyVecEnv([lambda e=Monitor(env): e]),
                                norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)
            if obs_rms is not None:
                venv.obs_rms = obs_rms
            obs = venv.reset()
            ep_r, ep_len, z_max, max_tilt = 0.0, 0, -1e9, 0.0
            reason, done, last_info = "timeout", [False], {}
            while not done[0]:
                act, _ = self.model.predict(obs, deterministic=True)
                obs, rew, done, infos = venv.step(act)
                ep_r += float(rew[0])
                ep_len += 1
                z_max = max(z_max, float(venv.get_attr("get_altitude")[0]()))
                max_tilt = max(max_tilt, float(infos[0].get("tilt_deg", 0)))
                last_info = infos[0]
                reason = infos[0].get("reason", reason)
            venv.close()
            result = "SUCCESS!" if last_info.get("success") else reason
            print(f"  {spawn_z:>6.2f}  {ep_len:>5d}  {ep_r:>7.2f}  {z_max:>6.2f}  {max_tilt:>5.1f}  {result}")

    def _on_step(self):
        if self.num_timesteps % self.every_n != 0:
            return True
        obs_rms = copy.deepcopy(getattr(self.model.env, "obs_rms", None))
        print(f"\n[DBG] Step {self.num_timesteps:,}")
        self._run_suite(obs_rms, tilt_deg=0.0, label="CLEAN", heights=self.clean_heights)
        print()
        self._run_suite(obs_rms, tilt_deg=25.0, label="TILT ", heights=self.tilt_heights)
        print()
        return True
