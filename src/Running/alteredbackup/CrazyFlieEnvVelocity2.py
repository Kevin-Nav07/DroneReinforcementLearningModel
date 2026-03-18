import os
from typing import Optional, Tuple, Dict, Any
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco as mj

class CrazyFlieEnvVelocity(gym.Env):
    """
    MuJoCo RL environment for the CrazyFlie 2.1 drone — sim-to-real hover transfer.

    ACTION SPACE  — 4D continuous [-1, 1]:
        [0] roll_cmd     → desired roll  angle  (±max_roll_deg)
        [1] pitch_cmd    → desired pitch angle  (±max_pitch_deg)
        [2] yawrate_cmd  → desired yaw rate     (±max_yawrate_deg/s)
        [3] vz_cmd       → desired vertical vel (±max_vz_cmd m/s)

    OBSERVATION — 13D × n_stack frames:
        [0:3]   x_rel, y_rel (spawn-relative), z_abs (ABSOLUTE altitude)
        [3:7]   quaternion qw, qx, qy, qz
        [7:10]  linear velocity  vx, vy, vz  (world frame)
        [10:13] angular velocity wx, wy, wz  (body frame)

    CHANGES vs v4:
        max_roll/pitch  3°  → 6°    doubles lateral correction force during takeoff
        ff_k            1.5 → 1.2   gentler climb reduces lateral excitation
        max_vz_ff       0.6 → 0.5   matches reduced ff_k
        w_z             2.0 → 1.5   allows gradient to flow at ground (see below)
        dense clip      -2  → -5    ROOT CAUSE FIX: at ground w/ old params, EVERY
                                    state clipped to -2.0 regardless of tilt/drift.
                                    Policy had ZERO gradient to learn upright takeoff.
                                    With w_z=1.5 + clip=-5: good ground behavior
                                    gives -3.1 (unclipped); bad gives -5+ (clipped).
        w_r             0.3 → 0.5   stronger lateral penalty (now gradient flows)
        r_scale         0.5 → 0.3   penalty starts earlier (at 0.3m not 0.5m)
        w_vxy           0.2 → 0.4   stronger lateral speed penalty
        vxy_scale       0.3 → 0.2   tighter lateral speed tolerance
        w_tilt          0.8 → 1.2   stronger upright pressure (compensates larger max_roll)
        tilt_scale      12° → 10°   10° is the upright reference (was 12°)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str,
        # ── Task ──────────────────────────────────────────────────────────────
        target_z: float = 1.0,
        max_steps: int = 1000,
        n_stack: int = 4,
        hover_band: float = 0.10,
        hover_required_steps: int = 300,
        hard_ceiling_margin: float = 2.0,
        # ── Spawn ─────────────────────────────────────────────────────────────
        start_z_min: float = 0.85,
        start_z_max: float = 1.15,
        start_xy_range: float = 0.05,
        # ── Safety ────────────────────────────────────────────────────────────
        safety_radius: float = 4.0,
        # ── Thrust smoothing ──────────────────────────────────────────────────
        thrust_lowpass_alpha: float = 0.25,
        thrust_slew_per_step: float = 0.08,
        # ── Auto-landing ──────────────────────────────────────────────────────
        auto_landing: bool = False,
        # ── Physics ───────────────────────────────────────────────────────────
        frame_skip: int = 10,
        # ── Domain Randomisation ──────────────────────────────────────────────
        obs_noise_std: float = 0.0,
        obs_bias_std: float = 0.0,
        action_noise_std: float = 0.0,
        motor_scale_std: float = 0.0,
        frame_skip_jitter: int = 0,
        torque_bias_std: float = 0.0,
        torque_gust_std: float = 0.0,
        torque_gust_tau: float = 1.5,
        drag_lin_min: float = 0.0,
        drag_lin_max: float = 0.0,
        drag_quad_min: float = 0.0,
        drag_quad_max: float = 0.0,
    ):
        super().__init__()
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data  = mj.MjData(self.model)

        # ── Task ──────────────────────────────────────────────────────────────
        self.target_z            = float(target_z)
        self.max_steps           = int(max_steps)
        self.band                = float(hover_band)
        self.hover_required      = int(hover_required_steps)
        self.hard_ceiling_margin = float(hard_ceiling_margin)

        # ── Spawn ─────────────────────────────────────────────────────────────
        self.start_z_min    = float(start_z_min)
        self.start_z_max    = float(start_z_max)
        self.start_xy_range = float(start_xy_range)
        self.spawn_xy = np.zeros(2, dtype=np.float64)
        self.spawn_z  = 0.0

        # ── Safety ────────────────────────────────────────────────────────────
        self.safety_radius      = float(safety_radius)
        self.ground_z_threshold = 0.05
        self.max_ground_steps   = 100

        # ── Spaces ────────────────────────────────────────────────────────────
        self.n_stack        = int(n_stack)
        self.obs_dim_single = 13
        hi = np.full(self.obs_dim_single * self.n_stack, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)
        self.obs_stack = deque(maxlen=self.n_stack)
        self.action_space = spaces.Box(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, +1.0, dtype=np.float32),
            dtype=np.float32,
        )

        # ── Thrust physics ────────────────────────────────────────────────────
        self.tmin = 0.0
        self.tmax = 0.4
        self.HOVER_THRUST = float(np.clip(0.34335, self.tmin, self.tmax))
        self.alpha  = float(thrust_lowpass_alpha)
        self.max_du = float(thrust_slew_per_step)
        self.u_cmd  = self.HOVER_THRUST
        self.last_du = 0.0
        self.last_moments = np.zeros(3, dtype=np.float32)
        self.last_dm = 0.0

        # ── Commander limits ──────────────────────────────────────────────────
        # max_roll/pitch 3°→6°: PD torque at max command doubles (0.314→0.628),
        # and lateral correction force doubles (0.018N→0.036N = 1.03 m/s²).
        # This is the physical fix enabling attitude correction during takeoff.
        self.max_roll_deg    = 6.0
        self.max_pitch_deg   = 6.0
        self.max_yawrate_deg = 45.0
        self.max_vz_cmd      = 0.6
        self.max_roll_rad  = np.deg2rad(self.max_roll_deg)
        self.max_pitch_rad = np.deg2rad(self.max_pitch_deg)
        self.max_yawrate   = np.deg2rad(self.max_yawrate_deg)

        # ── Attitude PD controller ────────────────────────────────────────────
        self.att_kp = 6.0
        self.att_kd = 0.3
        self.yaw_kp = 1.0
        self.yaw_kd = 0.05
        self.vz_kp  = 0.5

        # ── Domain Randomisation ──────────────────────────────────────────────
        self.obs_noise_std    = float(obs_noise_std)
        self.obs_bias_std     = float(obs_bias_std)
        self.action_noise_std = float(action_noise_std)
        self.motor_scale_std  = float(motor_scale_std)
        self.frame_skip_base  = int(frame_skip)
        self.frame_skip_jitter = int(frame_skip_jitter)
        self.frame_skip       = self.frame_skip_base
        self.torque_bias_std = float(torque_bias_std)
        self.torque_gust_std = float(torque_gust_std)
        self.torque_gust_tau = float(max(1e-3, torque_gust_tau))
        self.drag_lin_min  = float(drag_lin_min)
        self.drag_lin_max  = float(drag_lin_max)
        self.drag_quad_min = float(drag_quad_min)
        self.drag_quad_max = float(drag_quad_max)
        self.obs_bias    = np.zeros(self.obs_dim_single, dtype=np.float32)
        self.motor_scale = 1.0
        self.pos_gain    = np.ones(3, dtype=np.float32)
        self.vel_gain    = np.ones(3, dtype=np.float32)
        self.bias_drift  = np.zeros(self.obs_dim_single, dtype=np.float32)
        self.torque_bias = np.zeros(3, dtype=np.float32)
        self.torque_ou   = np.zeros(3, dtype=np.float32)
        self.drag_lin    = np.zeros(3, dtype=np.float32)
        self.drag_quad   = np.zeros(3, dtype=np.float32)

        # ── Reward ────────────────────────────────────────────────────────────
        # ff_k 1.5→1.2, max_vz_ff 0.6→0.5: slightly gentler climb rate from
        # ground reduces lateral excitation during takeoff.
        self.ff_k      = 1.2
        self.max_vz_ff = 0.50

        # w_z 2.0→1.5: CRITICAL for ground learning.
        # Old: z_cost at ground (dz=1m) = 2.0×(1/0.6)² = 5.56 → ALWAYS clips dense
        # New: z_cost at ground = 1.5×(1/0.6)² = 4.17 → dense=-3.1 (unclipped!)
        # This allows gradient to flow from tilt/lateral terms even at ground level.
        # Near hover (dz=0.1m): z_cost = 1.5×(0.1/0.6)² = 0.042 → still clear signal
        self.z_scale    = 0.60
        self.vz_scale   = 0.45
        # Tighter lateral: r_scale 0.5→0.3 means full penalty at 0.3m (was 0.5m)
        self.r_scale    = 0.30
        self.vxy_scale  = 0.20
        # tilt_scale 12°→10°: tighter upright reference; at 5° tilt, cost triples
        self.tilt_scale  = np.deg2rad(10.0)
        self.omega_scale = np.deg2rad(100.0)
        self.du_scale    = 0.02
        self.dm_scale    = 0.40

        # Reward weights
        self.w_z        = 1.5   # reduced (was 2.0) to unlock gradient at ground
        self.w_vz       = 0.8
        self.w_r        = 0.5   # stronger lateral penalty (was 0.3)
        self.w_vxy      = 0.4   # stronger lateral speed penalty (was 0.2)
        self.w_tilt     = 1.2   # stronger upright pressure (was 0.8)
        self.w_omega    = 0.02
        self.w_smooth_u = 0.05
        self.w_smooth_m = 0.05

        # Progress reward tracking
        self.prev_dz = 0.0

        # ── Auto-landing ──────────────────────────────────────────────────────
        self.auto_landing = bool(auto_landing)
        self._init_landing_params()

        # ── Episode state ─────────────────────────────────────────────────────
        self.step_idx     = 0
        self.hover_count  = 0
        self.ground_steps = 0
        self.phase        = "HOVER"
        self.target_z_abs = self.target_z
        self.hard_ceiling = self.target_z + self.hard_ceiling_margin

    # ──────────────────────────────────────────────────────────────────────────
    def _init_landing_params(self):
        self.landing_max_radius      = 0.8
        self.landing_safe_radius     = 0.5   # was 0.4 — more lenient landing zone
        self.landing_tilt_abort_deg  = 25.0
        self.landing_tilt_ok_deg     = 10.0
        self.landing_beta_ramp_steps = 200
        self.landing_max_steps       = 800
        self.landing_vz_fast = -0.30
        self.landing_vz_med  = -0.20
        self.landing_vz_mid  = -0.15
        self.landing_vz_slow = -0.10
        self.landing_k_vz    = 0.4
        self.landing_step_idx    = 0
        self.landing_beta        = 0.0
        self.landing_mode        = "DESCEND"
        self.landing_catch_steps = 0
        self.pre_landing_reason: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────────────
    def _sample_episode_randomization(self):
        rng = getattr(self, "np_random", np.random)
        if self.obs_bias_std > 0.0:
            bias = rng.normal(0.0, self.obs_bias_std, size=self.obs_dim_single).astype(np.float32)
            bias[3:7] = 0.0
            self.obs_bias = bias
        else:
            self.obs_bias[:] = 0.0
        self.pos_gain[:] = 1.0
        self.vel_gain[:] = 1.0
        if self.obs_bias_std > 0.0 or self.obs_noise_std > 0.0:
            self.pos_gain[0:2] = 1.0 + rng.normal(0.0, 0.02, size=2).astype(np.float32)
            self.pos_gain[2]   = 1.0 + float(rng.normal(0.0, 0.03))
            self.vel_gain[:]   = 1.0 + rng.normal(0.0, 0.05, size=3).astype(np.float32)
        self.motor_scale = float(1.0 + rng.normal(0.0, self.motor_scale_std)) \
                           if self.motor_scale_std > 0.0 else 1.0
        if self.frame_skip_jitter > 0:
            jitter = int(rng.integers(-self.frame_skip_jitter, self.frame_skip_jitter + 1))
            self.frame_skip = max(1, self.frame_skip_base + jitter)
        else:
            self.frame_skip = self.frame_skip_base
        self.bias_drift[:] = 0.0
        self.torque_bias = rng.normal(0.0, self.torque_bias_std, size=3).astype(np.float32) \
                           if self.torque_bias_std > 0.0 else np.zeros(3, dtype=np.float32)
        self.torque_ou[:] = 0.0
        self.drag_lin = rng.uniform(self.drag_lin_min, self.drag_lin_max, size=3).astype(np.float32) \
                        if self.drag_lin_max > 0.0 else np.zeros(3, dtype=np.float32)
        self.drag_quad = rng.uniform(self.drag_quad_min, self.drag_quad_max, size=3).astype(np.float32) \
                         if self.drag_quad_max > 0.0 else np.zeros(3, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def _get_single_obs(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[0:3],
            self.data.qpos[3:7],
            self.data.qvel[0:3],
            self.data.qvel[3:6],
        ]).astype(np.float32)

    def _apply_obs_noise(self, raw: np.ndarray) -> np.ndarray:
        rng = getattr(self, "np_random", np.random)
        s = raw.astype(np.float32).copy()
        pos_meas  = self.pos_gain * s[0:3] + self.obs_bias[0:3]
        vel_meas  = self.vel_gain * s[7:10] + self.obs_bias[7:10]
        angv_meas = s[10:13] + self.obs_bias[10:13]
        if self.obs_noise_std > 0.0:
            self.bias_drift += rng.normal(0.0, self.obs_noise_std * 0.01,
                                           size=self.obs_dim_single).astype(np.float32)
            pos_meas  += self.bias_drift[0:3]
            vel_meas  += self.bias_drift[7:10]
            angv_meas += self.bias_drift[10:13]
            pos_meas[0:2] += rng.normal(0.0, self.obs_noise_std * 0.5, size=2).astype(np.float32)
            pos_meas[2]   += float(rng.normal(0.0, self.obs_noise_std * 1.0))
            vel_meas      += rng.normal(0.0, self.obs_noise_std * 0.7, size=3).astype(np.float32)
            angv_meas     += rng.normal(0.0, self.obs_noise_std * 0.4, size=3).astype(np.float32)
            if rng.random() < 1e-3:
                pos_meas[0:3] += rng.normal(0.0, self.obs_noise_std * 10.0, size=3).astype(np.float32)
        else:
            self.bias_drift[:] = 0.0
        quat = s[3:7].copy()
        q_norm = float(np.linalg.norm(quat))
        if q_norm > 1e-6:
            quat /= q_norm
        obs_pos = np.array([
            pos_meas[0] - float(self.spawn_xy[0]),
            pos_meas[1] - float(self.spawn_xy[1]),
            pos_meas[2],   # z absolute
        ], dtype=np.float32)
        return np.concatenate([obs_pos, quat, vel_meas, angv_meas]).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)
        rng = getattr(self, "np_random", np.random)

        x0 = float(rng.uniform(-self.start_xy_range, self.start_xy_range))
        y0 = float(rng.uniform(-self.start_xy_range, self.start_xy_range))
        z0 = float(np.clip(rng.uniform(self.start_z_min, self.start_z_max), 0.01, 10.0))
        self.spawn_xy[:] = [x0, y0]
        self.spawn_z = z0

        self.target_z_abs = self.target_z
        self.hard_ceiling = self.target_z_abs + self.hard_ceiling_margin
        if z0 > self.hard_ceiling - 0.05:
            z0 = self.hard_ceiling - 0.05
            self.spawn_z = z0

        self.data.qpos[:] = np.array([x0, y0, z0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.u_cmd           = self.HOVER_THRUST
        self.last_du         = 0.0
        self.last_moments[:] = 0.0
        self.last_dm         = 0.0
        self.hover_count     = 0
        self.step_idx        = 0
        self.ground_steps    = 0
        self.prev_dz         = 0.0
        self.phase           = "HOVER"
        self.landing_step_idx    = 0
        self.landing_beta        = 0.0
        self.landing_mode        = "DESCEND"
        self.landing_catch_steps = 0
        self.pre_landing_reason  = None

        self._end_noise_free_landing()
        self._sample_episode_randomization()

        self.obs_stack.clear()
        first = self._apply_obs_noise(self._get_single_obs())
        for _ in range(self.n_stack):
            self.obs_stack.append(first.copy())
        return np.concatenate(list(self.obs_stack), axis=0).astype(np.float32), {}

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_disturbances(self, dt: float):
        if int(self.model.nv) < 6:
            return
        self.data.qfrc_applied[:] = 0.0
        rng = getattr(self, "np_random", np.random)
        if self.torque_gust_std > 0.0:
            decay = np.exp(-dt / self.torque_gust_tau)
            noise = rng.normal(0.0, self.torque_gust_std, size=3).astype(np.float32)
            self.torque_ou = decay * self.torque_ou + np.sqrt(max(1e-9, 1.0 - decay**2)) * noise
        else:
            self.torque_ou[:] = 0.0
        v    = self.data.qvel[0:3].astype(np.float32)
        drag = -self.drag_lin * v - self.drag_quad * np.abs(v) * v
        self.data.qfrc_applied[0:3] += drag.astype(np.float64)
        self.data.qfrc_applied[3:6] += (self.torque_bias + self.torque_ou).astype(np.float64)

    def _apply_thrust(self, u_scalar: float, m_vec: np.ndarray):
        du       = np.clip(u_scalar - self.u_cmd, -self.max_du, self.max_du)
        u_slewed = self.u_cmd + du
        new_u    = (1.0 - self.alpha) * self.u_cmd + self.alpha * u_slewed
        self.last_du  = float(abs(new_u - self.u_cmd))
        self.u_cmd    = float(new_u)
        self.data.ctrl[0] = self.u_cmd
        m = np.clip(np.asarray(m_vec, dtype=np.float32), -1.0, 1.0)
        if int(self.model.nu) >= 4:
            self.data.ctrl[1:4] = m.astype(np.float64)
            if int(self.model.nu) > 4:
                self.data.ctrl[4:] = 0.0
        self.last_dm      = float(np.linalg.norm(m - self.last_moments, ord=2))
        self.last_moments = m.copy()

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _quat_to_euler(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
        roll  = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
        sinp  = float(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0))
        pitch = np.sign(sinp) * np.pi / 2.0 if abs(sinp) >= 1.0 else float(np.arcsin(sinp))
        yaw   = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
        return float(roll), float(pitch), float(yaw)

    def _decode_commander(self, a_norm: np.ndarray) -> Tuple[float, float, float, float]:
        a = np.asarray(a_norm, dtype=np.float32).reshape(4)
        return (
            float(a[0]) * self.max_roll_rad,
            float(a[1]) * self.max_pitch_rad,
            float(a[2]) * self.max_yawrate,
            float(a[3]) * self.max_vz_cmd,
        )

    def _attitude_pd(
        self, roll_cmd: float, pitch_cmd: float, yawrate_cmd: float, state: np.ndarray
    ) -> np.ndarray:
        qw, qx, qy, qz = state[3], state[4], state[5], state[6]
        wx, wy, wz      = float(state[10]), float(state[11]), float(state[12])
        roll, pitch, _  = self._quat_to_euler(qw, qx, qy, qz)
        tau_roll  = self.att_kp * (roll_cmd  - roll)  - self.att_kd * wx
        tau_pitch = self.att_kp * (pitch_cmd - pitch) - self.att_kd * wy
        tau_yaw   = self.yaw_kp * (yawrate_cmd - wz)  - self.yaw_kd * wz
        return np.clip(np.array([tau_roll, tau_pitch, tau_yaw], dtype=np.float32), -1.0, 1.0)

    def _vertical_pd(self, vz_cmd: float, state: np.ndarray) -> float:
        return float(np.clip(
            self.HOVER_THRUST - self.vz_kp * (float(state[9]) - vz_cmd),
            self.tmin, self.tmax
        ))

    def _commander_to_actuators(
        self, a_norm: np.ndarray, state: np.ndarray
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd = self._decode_commander(a_norm)
        m_vec = self._attitude_pd(roll_cmd, pitch_cmd, yawrate_cmd, state)
        u_req = self._vertical_pd(vz_cmd, state)
        return u_req, m_vec, {"vz_cmd": float(vz_cmd)}

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 4:
            raise ValueError(f"Action must be shape (4,), got {a.shape}")

        if self.auto_landing and self.phase == "LANDING":
            return self._step_landing(action)

        if self.action_noise_std > 0.0:
            rng = getattr(self, "np_random", np.random)
            a = a + rng.normal(0.0, self.action_noise_std, size=4).astype(np.float32)
        a = np.clip(a, -1.0, 1.0)

        state_now = self._get_single_obs()
        u_req, m_req, _ = self._commander_to_actuators(a, state_now)
        u_req = float(np.clip(u_req * self.motor_scale, self.tmin, self.tmax))

        self._apply_thrust(u_req, m_req)
        dt = float(self.model.opt.timestep)
        for _ in range(self.frame_skip):
            self._apply_disturbances(dt)
            mj.mj_step(self.model, self.data)
        self.step_idx += 1

        single_clean = self._get_single_obs()
        single_noisy = self._apply_obs_noise(single_clean)
        self.obs_stack.append(single_noisy)
        obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)

        x2, y2, z2    = float(single_clean[0]), float(single_clean[1]), float(single_clean[2])
        qx2, qy2      = float(single_clean[4]), float(single_clean[5])
        vx2, vy2, vz2 = float(single_clean[7]), float(single_clean[8]), float(single_clean[9])
        wx2, wy2, wz2 = float(single_clean[10]), float(single_clean[11]), float(single_clean[12])

        r_rad2      = float(np.sqrt((x2 - self.spawn_xy[0])**2 + (y2 - self.spawn_xy[1])**2))
        tilt_sin2   = float(np.clip(np.sqrt(qx2**2 + qy2**2), 0.0, 1.0))
        tilt_angle2 = float(2.0 * np.arcsin(tilt_sin2))
        tilt_deg2   = float(np.rad2deg(tilt_angle2))

        on_ground = z2 < self.ground_z_threshold
        self.ground_steps = self.ground_steps + 1 if on_ground else 0

        # ── Reward ────────────────────────────────────────────────────────────
        dz     = z2 - self.target_z_abs
        vxy2   = vx2**2 + vy2**2
        omega2 = wx2**2 + wy2**2 + wz2**2

        vz_desired = float(np.clip(-self.ff_k * dz, -self.max_vz_ff, self.max_vz_ff))

        cost = (
            self.w_z        * (dz / self.z_scale) ** 2
            + self.w_vz     * ((vz2 - vz_desired) / self.vz_scale) ** 2
            + self.w_r      * (r_rad2 / self.r_scale) ** 2
            + self.w_vxy    * (vxy2 / self.vxy_scale**2)
            + self.w_tilt   * (tilt_angle2 / self.tilt_scale) ** 2
            + self.w_omega  * (omega2 / self.omega_scale**2)
            + self.w_smooth_u * (self.last_du / self.du_scale) ** 2
            + self.w_smooth_m * (self.last_dm / self.dm_scale) ** 2
        )

        # Extended clip: -5.0 allows gradient at ground level.
        # Old -2.0 clip meant EVERY ground state gave exactly -2.0 reward regardless
        # of tilt or lateral drift — policy had zero gradient to learn upright takeoff.
        dense  = float(np.clip(1.0 - cost, -5.0, 2.0))
        reward = dense / max(1, self.max_steps)

        # Altitude progress bonus
        curr_dz_abs = abs(dz)
        if curr_dz_abs < abs(self.prev_dz):
            reward += 0.3 / max(1, self.max_steps)
        self.prev_dz = dz

        if on_ground:
            reward -= 0.1 / max(1, self.max_steps)

        # ── Hover tracking ────────────────────────────────────────────────────
        stable = (
            abs(dz)            <= self.band
            and abs(vz2)       <  0.05
            and tilt_angle2    <  np.deg2rad(10.0)
            and r_rad2         <  0.30
            and np.sqrt(vxy2)  <  0.15
            and not on_ground
        )
        if stable:
            self.hover_count += 1
            reward += 0.2 / max(1, self.max_steps)
        else:
            self.hover_count = 0

        # ── Success ───────────────────────────────────────────────────────────
        if self.hover_count >= self.hover_required:
            reward += 1.0
            info = {"success": True, "hover_steps": self.hover_count,
                    "tilt_deg": tilt_deg2, "radius": r_rad2, "vz": vz2, "z": z2,
                    "att_scale": 1.0}
            if self.auto_landing:
                self._start_landing_phase("success")
                info["phase"] = "landing_start"
                return obs, float(np.clip(reward, -5.0, 2.0)), False, False, info
            return obs, float(np.clip(reward, -5.0, 2.0)), True, False, info

        # ── Terminations ──────────────────────────────────────────────────────
        if self.ground_steps >= self.max_ground_steps:
            return obs, -1.0, True, False, {"crash": True, "reason": "stalled_on_ground"}
        if z2 < 0.01 or not np.all(np.isfinite(obs)):
            return obs, -1.0, True, False, {"crash": True, "reason": "nan_or_below_ground"}
        if tilt_angle2 > np.deg2rad(70.0):
            return obs, -1.0, True, False, {"crash": True, "reason": "flipped"}
        if z2 > self.hard_ceiling:
            return obs, -1.0, True, False, {"ceiling": True, "reason": "hard_ceiling"}
        if r_rad2 > self.safety_radius:
            return obs, -1.0, True, False, {"crash": True, "reason": "out_of_bounds"}
        if self.step_idx >= self.max_steps:
            info = {"hover_steps": self.hover_count, "timeout": True, "att_scale": 1.0}
            if self.auto_landing:
                self._start_landing_phase("timeout")
                info["phase"] = "landing_start"
                return obs, float(np.clip(reward, -5.0, 2.0)), False, False, info
            return obs, float(np.clip(reward, -5.0, 2.0)), False, True, info

        return obs, float(np.clip(reward, -5.0, 2.0)), False, False, {
            "hover_steps": self.hover_count,
            "tilt_deg":    tilt_deg2,
            "radius":      r_rad2,
            "vz":          vz2,
            "z":           z2,
            "att_scale":   1.0,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def get_altitude(self) -> float:
        return float(self.data.qpos[2])

    def cut_motors(self) -> None:
        self.u_cmd = 0.0
        self.data.ctrl[:] = 0.0

    def _tilt_and_radius(self) -> Tuple[float, float]:
        qx, qy = float(self.data.qpos[4]), float(self.data.qpos[5])
        tilt = float(2.0 * np.arcsin(float(np.clip(np.sqrt(qx**2 + qy**2), 0.0, 1.0))))
        r    = float(np.sqrt(
            (self.data.qpos[0] - self.spawn_xy[0])**2 +
            (self.data.qpos[1] - self.spawn_xy[1])**2
        ))
        return tilt, r

    # ──────────────────────────────────────────────────────────────────────────
    def _start_landing_phase(self, reason: str):
        self.phase               = "LANDING"
        self.landing_step_idx    = 0
        self.landing_beta        = 0.0
        self.landing_mode        = "DESCEND"
        self.landing_catch_steps = 0
        self.pre_landing_reason  = reason
        self._begin_noise_free_landing()

    def _begin_noise_free_landing(self):
        self._landing_noise_backup = {
            "action_noise_std": self.action_noise_std,
            "motor_scale":      self.motor_scale,
        }
        self.action_noise_std = 0.0
        self.motor_scale      = 1.0

    def _end_noise_free_landing(self):
        if hasattr(self, "_landing_noise_backup"):
            self.action_noise_std = self._landing_noise_backup["action_noise_std"]
            self.motor_scale      = self._landing_noise_backup["motor_scale"]
            del self._landing_noise_backup

    def _step_landing(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        a_pol = np.clip(np.asarray(action, dtype=np.float32).reshape(4), -1.0, 1.0)
        state = self._get_single_obs()
        z, vz = float(state[2]), float(state[9])
        vx, vy = float(state[7]), float(state[8])

        roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd_pol = self._decode_commander(a_pol)
        m_pol = self._attitude_pd(roll_cmd, pitch_cmd, yawrate_cmd, state)
        u_pol = float(np.clip(
            self.HOVER_THRUST - self.vz_kp * (vz - vz_cmd_pol), self.tmin, self.tmax
        ))

        tilt, r  = self._tilt_and_radius()
        tilt_deg = float(np.rad2deg(tilt))

        if tilt_deg > self.landing_tilt_abort_deg or r > self.landing_max_radius:
            self.landing_mode        = "CATCH"
            self.landing_catch_steps = 0

        if self.landing_mode == "CATCH":
            v_des = 0.0
            self.landing_beta = max(0.0, self.landing_beta - 0.05)
            stabilised = (tilt_deg < self.landing_tilt_ok_deg and r < self.landing_safe_radius
                          and abs(vx) < 0.2 and abs(vy) < 0.2)
            self.landing_catch_steps = self.landing_catch_steps + 1 if stabilised else 0
            if self.landing_catch_steps > 50:
                self.landing_mode = "DESCEND"
        else:
            h = max(0.0, z - 0.03)
            if   h > 0.8: v_des = self.landing_vz_fast
            elif h > 0.4: v_des = self.landing_vz_med
            elif h > 0.2: v_des = self.landing_vz_mid
            else:         v_des = self.landing_vz_slow
            self.landing_beta = min(1.0, self.landing_step_idx / max(1, self.landing_beta_ramp_steps))

        beta   = float(self.landing_beta)
        u_land = float(np.clip(self.HOVER_THRUST - self.landing_k_vz * (vz - v_des), 0.12, self.tmax))
        u      = float(np.clip((1.0 - beta) * u_pol + beta * u_land, 0.12, self.tmax))

        self._apply_thrust(u, m_pol)
        dt = float(self.model.opt.timestep)
        for _ in range(self.frame_skip):
            self._apply_disturbances(dt)
            mj.mj_step(self.model, self.data)

        self.step_idx         += 1
        self.landing_step_idx += 1

        next_state = self._get_single_obs()
        self.obs_stack.append(self._apply_obs_noise(next_state))
        obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)

        z2, vz2   = float(next_state[2]), float(next_state[9])
        tilt2, r2 = self._tilt_and_radius()
        tilt_deg2 = float(np.rad2deg(tilt2))

        # Core landing success criteria
        close_to_ground = (z2 <= 0.03)
        slow_descent    = (abs(vz2) < 0.15)
        upright         = (tilt_deg2 < 12.0)
        inside_zone     = (r2 < self.landing_safe_radius)
        landed = close_to_ground and slow_descent and upright and inside_zone

        # Emergency ground termination: prevents bounce-flip loop.
        # If we've been in landing for >80 steps and are basically on the ground,
        # terminate even if tilt or radius checks fail — the drone is on the ground.
        emergency_ground = (z2 <= 0.04 and self.landing_step_idx > 80)

        timeout = self.landing_step_idx >= self.landing_max_steps
        terminated = False
        if landed or emergency_ground or timeout:
            self._end_noise_free_landing()
            self.phase = "HOVER"
            terminated = True

        return obs, 0.0, terminated, False, {
            "phase":              "landing",
            "landing_mode":       self.landing_mode,
            "landing_beta":       beta,
            "landing_landed":     landed or emergency_ground,
            "landing_timeout":    timeout,
            "tilt_deg":           tilt_deg2,
            "radius":             r2,
            "vz":                 float(vz2),
            "pre_landing_reason": self.pre_landing_reason,
            "att_scale":          1.0,
        }