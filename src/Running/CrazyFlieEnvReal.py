"""
CrazyFlie Real Hardware Environment — direct-policy deployment with strict real-hardware safety filter.

The policy still sends the same action:
    [roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd] in [-1, 1]

But the real environment converts that raw policy action into a safer hardware command:
    raw policy action
        -> altitude/tilt/radius/vxy authority gate
        -> low-altitude descent protection
        -> rate limiter
        -> lowered real roll/pitch/yaw limits
        -> Crazyflie commander setpoint

This lets you test the policy from reset without a scripted takeoff, while preventing
full attitude authority near the ground or during lateral flyaway.
"""

import time
import logging
from collections import deque
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from CrazyFlieStateObserver import CrazyFlieStateObserver

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Hardware / physical constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_M_REAL_KG = 0.033
DEFAULT_G = 9.81
DEFAULT_W_HOVER = DEFAULT_M_REAL_KG * DEFAULT_G

DEFAULT_U_HOVER_COUNTS = 41940.0
DEFAULT_COUNTS_PER_NEWTON = DEFAULT_U_HOVER_COUNTS / DEFAULT_W_HOVER

MIN_THRUST_COUNTS = 10001
MAX_THRUST_COUNTS = 60000

SIM_T_MIN = 0.0
SIM_T_MAX = 0.4
SIM_HOVER_THRUST_N = 0.34335


class CrazyFlieRealEnvVelocity(gym.Env):
    """
    Real-hardware twin of CrazyFlieEnvVelocity2 with extra deployment safety.

    Action:
        Box(4,) in [-1, 1] = [roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd]

    Observation:
        stacked [x_rel, y_rel, z_abs, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        uri: str,
        # ── Task ──────────────────────────────────────────────────────────────
        target_z: float = 1.0,
        max_steps: int = 500,
        n_stack: int = 4,
        hover_band: float = 0.10,
        hover_required_steps: int = 300,
        hard_ceiling_margin: float = 2.0,
        # ── Safety ────────────────────────────────────────────────────────────
        safety_radius: float = 2.0,
        # ── Thrust smoothing ──────────────────────────────────────────────────
        thrust_lowpass_alpha: float = 0.25,
        thrust_slew_per_step: float = 0.08,
        # ── Auto-landing ──────────────────────────────────────────────────────
        auto_landing: bool = True,
        # ── Hardware / radio ──────────────────────────────────────────────────
        log_period_ms: int = 20,
        m_real_kg: float = DEFAULT_M_REAL_KG,
        g: float = DEFAULT_G,
        u_hover_counts: float = DEFAULT_U_HOVER_COUNTS,
        counts_per_newton_override: Optional[float] = None,
        debug: bool = True,
    ) -> None:
        super().__init__()

        # ── Hardware setup ────────────────────────────────────────────────────
        self.uri = uri
        self.debug = bool(debug)
        self.dt = log_period_ms / 1000.0
        self.state_timeout_s = max(0.5, 5 * self.dt)

        self.observer = CrazyFlieStateObserver(uri=uri, log_period_ms=log_period_ms)

        # ── Physical constants ────────────────────────────────────────────────
        self.m_real = float(m_real_kg)
        self.g = float(g)
        self.w_hover = self.m_real * self.g

        self.u_hover_counts = float(u_hover_counts)
        self.counts_per_newton = (
            float(counts_per_newton_override)
            if counts_per_newton_override is not None
            else DEFAULT_COUNTS_PER_NEWTON
        )

        # ── Task ──────────────────────────────────────────────────────────────
        self.target_z = float(target_z)
        self.max_steps = int(max_steps)
        self.band = float(hover_band)
        self.hover_required = int(hover_required_steps)
        self.hard_ceiling_margin = float(hard_ceiling_margin)

        # ── Safety ────────────────────────────────────────────────────────────
        self.safety_radius = float(safety_radius)
        self.ground_z_threshold = 0.05
        self.max_ground_steps = 250

        # ── Spaces ────────────────────────────────────────────────────────────
        self.n_stack = int(n_stack)
        self.obs_dim_single = 13

        hi = np.full(self.obs_dim_single * self.n_stack, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)
        self.obs_stack = deque(maxlen=self.n_stack)

        self.action_space = spaces.Box(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, +1.0, dtype=np.float32),
            dtype=np.float32,
        )

        # ── Thrust smoothing ──────────────────────────────────────────────────
        self.tmin = SIM_T_MIN
        self.tmax = SIM_T_MAX
        self.HOVER_THRUST = float(np.clip(SIM_HOVER_THRUST_N, self.tmin, self.tmax))

        self.alpha = float(thrust_lowpass_alpha)
        self.max_du = float(thrust_slew_per_step)

        self.u_cmd = self.HOVER_THRUST
        self.last_du = 0.0
        self.last_moments = np.zeros(3, dtype=np.float32)
        self.last_dm = 0.0

        # ── Commander limits for REAL deployment ──────────────────────────────
        # Raw policy outputs are saturating. These values reinterpret [-1, 1]
        # conservatively on the real drone.
        self.max_roll_deg = 8.0
        self.max_pitch_deg = 8.0
        self.max_yawrate_deg = 12.0
        self.max_vz_cmd = 1.0

        self.max_roll_rad = np.deg2rad(self.max_roll_deg)
        self.max_pitch_rad = np.deg2rad(self.max_pitch_deg)
        self.max_yawrate = np.deg2rad(self.max_yawrate_deg)

        # ── Real-hardware policy safety adapter ───────────────────────────────
        self.real_action_filter_enabled = True

        # No policy attitude authority until the drone is safely above the floor.
        self.ground_attitude_lock_z = 0.70

        # Full altitude gate opens only around the real hover region.
        self.full_attitude_z = 1.00

        # Even at full height, only give policy a fraction of the configured 8°.
        # 0.25 * 8° = 2° maximum roll/pitch for first direct-policy tests.
        self.max_policy_attitude_authority = 0.25

        # Yaw is not needed for hover; keep it very small.
        self.max_policy_yaw_authority = 0.10

        # Stop earlier than the old 70° flip limit.
        self.real_flip_stop_deg = 35.0

        # Per-step normalized action rate limits.
        self.max_action_delta = np.array([0.010, 0.010, 0.015, 0.050], dtype=np.float32)

        # Vertical command clamp in normalized action units.
        self.real_vz_up_clip = 0.30
        self.real_vz_down_clip = -0.15

        # Do not let the policy descend while still low.
        self.no_descent_below_z = 0.65

        # Low-altitude recovery. If the drone sinks too low, attitude is zeroed
        # and the env forces an upward vertical command.
        self.recovery_climb_below_z = 0.30
        self.recovery_vz_action = 0.25

        # Tilt-based authority reduction.
        self.tilt_soft_limit_rad = np.deg2rad(8.0)
        self.tilt_cut_limit_rad = np.deg2rad(18.0)

        # Radius-based authority reduction. If it starts drifting away,
        # reduce attitude authority instead of letting the policy push harder.
        self.radius_soft_limit = 0.80
        self.radius_cut_limit = 1.50

        # Lateral-velocity authority reduction.
        self.vxy_soft_limit = 0.35
        self.vxy_cut_limit = 0.80

        self.last_raw_action = np.zeros(4, dtype=np.float32)
        self.last_safe_action = np.zeros(4, dtype=np.float32)
        self.last_policy_authority = 0.0

        # ── Controller monitoring terms ───────────────────────────────────────
        self.att_kp = 6.0
        self.att_kd = 0.3
        self.yaw_kp = 1.0
        self.yaw_kd = 0.05
        self.vz_kp = 0.5

        # ── Auto-landing ──────────────────────────────────────────────────────
        self.auto_landing = bool(auto_landing)
        self._init_landing_params()

        # ── Reward shaping / monitoring ───────────────────────────────────────
        self.ff_k = 1.2
        self.max_vz_ff = 0.50
        self.z_scale = 0.60
        self.vz_scale = 0.45
        self.r_scale = 0.30
        self.vxy_scale = 0.20
        self.tilt_scale = np.deg2rad(10.0)
        self.omega_scale = np.deg2rad(100.0)
        self.du_scale = 0.02
        self.dm_scale = 0.40

        self.w_z = 1.5
        self.w_vz = 0.8
        self.w_r = 0.5
        self.w_vxy = 0.4
        self.w_tilt = 1.2
        self.w_omega = 0.02
        self.w_smooth_u = 0.05
        self.w_smooth_m = 0.05

        self.prev_dz = 0.0

        # ── Episode state ─────────────────────────────────────────────────────
        self.step_idx = 0
        self.hover_count = 0
        self.ground_steps = 0
        self.phase = "HOVER"

        self.target_z_abs = self.target_z
        self.hard_ceiling = self.target_z + self.hard_ceiling_margin

        self.spawn_xy = np.zeros(2, dtype=np.float64)
        self.spawn_z = 0.0

        # Connect & prime.
        self._connect_and_prime()

    # ──────────────────────────────────────────────────────────────────────────
    # Landing params
    # ──────────────────────────────────────────────────────────────────────────

    def _init_landing_params(self) -> None:
        self.landing_max_radius = 0.8
        self.landing_safe_radius = 0.5
        self.landing_tilt_abort_deg = 25.0
        self.landing_tilt_ok_deg = 10.0
        self.landing_beta_ramp_steps = 200
        self.landing_max_steps = 800

        self.landing_vz_fast = -0.30
        self.landing_vz_med = -0.20
        self.landing_vz_mid = -0.15
        self.landing_vz_slow = -0.10
        self.landing_k_vz = 0.4

        self.landing_step_idx = 0
        self.landing_beta = 0.0
        self.landing_mode = "DESCEND"
        self.landing_catch_steps = 0
        self.pre_landing_reason: Optional[str] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Hardware setup
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_and_prime(self) -> None:
        if self.debug:
            logger.info("CrazyFlieRealEnvVelocity: connecting to %s", self.uri)

        self.observer.connect()
        self._reset_estimator()

        t0 = time.time()
        while not self.observer.is_ready():
            if time.time() - t0 > 5.0:
                raise RuntimeError("Timed out waiting for Crazyflie state stream.")
            time.sleep(0.01)

        try:
            cf = self.observer.cf
        except RuntimeError:
            logger.warning("Observer has no Crazyflie handle; cannot prime motors.")
            return

        try:
            cf.param.set_value("commander.enHighLevel", "0")
            time.sleep(0.2)
        except Exception as e:
            logger.warning("Could not disable high-level commander: %s", e)

        try:
            cf.platform.send_arming_request(True)
            time.sleep(0.5)
        except Exception as e:
            logger.warning("Arming request skipped/failed: %s", e)

        if self.debug:
            logger.info("Unlocking thrust protection with zero-thrust setpoints...")

        for _ in range(40):
            cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
            time.sleep(self.dt)

    def _reset_estimator(self) -> None:
        try:
            cf = self.observer.cf
            logger.info("Resetting Crazyflie estimator...")
            cf.param.set_value("kalman.resetEstimation", "1")
            time.sleep(0.15)
            cf.param.set_value("kalman.resetEstimation", "0")
            time.sleep(2.0)
        except Exception as e:
            logger.warning("Estimator reset failed/skipped: %s", e)

    def _send_safe_stop(self, n: int = 40) -> None:
        try:
            cf = self.observer.cf
        except RuntimeError:
            return

        for _ in range(n):
            try:
                cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
            except Exception:
                pass
            time.sleep(self.dt)

    def emergency_stop(self) -> None:
        self._send_safe_stop()

    # ──────────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────────

    def _get_single_obs_global(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        timeout = self.state_timeout_s if timeout is None else timeout
        t0 = time.time()

        while time.time() - t0 < timeout:
            state = self.observer.get_state()

            if state is not None:
                state = np.asarray(state, dtype=np.float32)

                if state.shape == (13,) and np.all(np.isfinite(state)):
                    q = state[3:7]
                    q_norm = float(np.linalg.norm(q))

                    if 0.8 <= q_norm <= 1.2:
                        return state

            time.sleep(self.dt)

        return None

    def _to_agent_obs(self, global_state: np.ndarray) -> np.ndarray:
        s = global_state.astype(np.float32).copy()

        pos = np.array(
            [
                s[0] - float(self.spawn_xy[0]),
                s[1] - float(self.spawn_xy[1]),
                s[2],
            ],
            dtype=np.float32,
        )

        quat = s[3:7].copy()
        n = float(np.linalg.norm(quat))
        if n > 1e-6:
            quat /= n

        vel = s[7:10].astype(np.float32)
        omg = s[10:13].astype(np.float32)

        return np.concatenate([pos, quat, vel, omg]).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Controller helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_euler(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
        roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        sinp = float(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        pitch = np.sign(sinp) * np.pi / 2.0 if abs(sinp) >= 1.0 else float(np.arcsin(sinp))
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return float(roll), float(pitch), float(yaw)

    def _decode_commander(self, a_norm: np.ndarray) -> Tuple[float, float, float, float]:
        a = np.clip(np.asarray(a_norm, dtype=np.float32).reshape(4), -1.0, 1.0)

        return (
            float(a[0]) * self.max_roll_rad,
            float(a[1]) * self.max_pitch_rad,
            float(a[2]) * self.max_yawrate,
            float(a[3]) * self.max_vz_cmd,
        )

    def _filter_real_policy_action(self, a_norm: np.ndarray, state_now: np.ndarray) -> np.ndarray:
        """
        Real-hardware safety adapter.

        The PPO policy still produces the action from step 0.
        This function decides how much real-world authority that action gets.

        Main idea:
          - vertical policy is allowed early
          - attitude policy is blocked until the drone is high enough
          - attitude authority is capped to a small fraction
          - radius/vxy/tilt reduce authority if the drone starts flying away
          - low-altitude descent is blocked
        """
        raw = np.clip(np.asarray(a_norm, dtype=np.float32).reshape(4), -1.0, 1.0)
        safe = raw.copy()

        z = float(state_now[2])
        x_rel = float(state_now[0] - self.spawn_xy[0])
        y_rel = float(state_now[1] - self.spawn_xy[1])
        vx = float(state_now[7])
        vy = float(state_now[8])

        radius = float(np.sqrt(x_rel * x_rel + y_rel * y_rel))
        vxy = float(np.sqrt(vx * vx + vy * vy))

        qx = float(state_now[4])
        qy = float(state_now[5])
        tilt = float(2.0 * np.arcsin(float(np.clip(np.sqrt(qx * qx + qy * qy), 0.0, 1.0))))

        # ── Altitude authority ────────────────────────────────────────────────
        # z <= 0.70: no attitude authority
        # z >= 1.00: altitude gate fully open
        if z <= self.ground_attitude_lock_z:
            z_authority = 0.0
        elif z >= self.full_attitude_z:
            z_authority = 1.0
        else:
            t = (z - self.ground_attitude_lock_z) / max(
                1e-6,
                self.full_attitude_z - self.ground_attitude_lock_z,
            )
            t = float(np.clip(t, 0.0, 1.0))
            z_authority = t * t * (3.0 - 2.0 * t)

        # ── Tilt authority ────────────────────────────────────────────────────
        if tilt <= self.tilt_soft_limit_rad:
            tilt_authority = 1.0
        elif tilt >= self.tilt_cut_limit_rad:
            tilt_authority = 0.0
        else:
            t = (tilt - self.tilt_soft_limit_rad) / max(
                1e-6,
                self.tilt_cut_limit_rad - self.tilt_soft_limit_rad,
            )
            tilt_authority = float(1.0 - np.clip(t, 0.0, 1.0))

        # ── Radius authority ──────────────────────────────────────────────────
        if radius <= self.radius_soft_limit:
            radius_authority = 1.0
        elif radius >= self.radius_cut_limit:
            radius_authority = 0.0
        else:
            t = (radius - self.radius_soft_limit) / max(
                1e-6,
                self.radius_cut_limit - self.radius_soft_limit,
            )
            radius_authority = float(1.0 - np.clip(t, 0.0, 1.0))

        # ── Lateral velocity authority ────────────────────────────────────────
        if vxy <= self.vxy_soft_limit:
            vxy_authority = 1.0
        elif vxy >= self.vxy_cut_limit:
            vxy_authority = 0.0
        else:
            t = (vxy - self.vxy_soft_limit) / max(
                1e-6,
                self.vxy_cut_limit - self.vxy_soft_limit,
            )
            vxy_authority = float(1.0 - np.clip(t, 0.0, 1.0))

        attitude_authority = float(
            np.clip(
                z_authority
                * tilt_authority
                * radius_authority
                * vxy_authority
                * self.max_policy_attitude_authority,
                0.0,
                self.max_policy_attitude_authority,
            )
        )

        yaw_authority = float(
            np.clip(
                z_authority
                * tilt_authority
                * radius_authority
                * vxy_authority
                * self.max_policy_yaw_authority,
                0.0,
                self.max_policy_yaw_authority,
            )
        )

        # Policy attitude is allowed, but only through the authority gates.
        safe[0] = raw[0] * attitude_authority
        safe[1] = raw[1] * attitude_authority
        safe[2] = raw[2] * yaw_authority

        # Policy vertical command is allowed, but clipped.
        safe[3] = float(np.clip(raw[3], self.real_vz_down_clip, self.real_vz_up_clip))

        # Do not allow descent while the drone is still low.
        if z < self.no_descent_below_z and safe[3] < 0.0:
            safe[3] = 0.0

        # Emergency low-altitude recovery.
        if z < self.recovery_climb_below_z:
            safe[0] = 0.0
            safe[1] = 0.0
            safe[2] = 0.0
            safe[3] = max(float(safe[3]), float(self.recovery_vz_action))

        # Rate-limit final action.
        delta = np.clip(
            safe - self.last_safe_action,
            -self.max_action_delta,
            self.max_action_delta,
        )
        safe = self.last_safe_action + delta

        self.last_raw_action = raw.astype(np.float32)
        self.last_safe_action = safe.astype(np.float32)
        self.last_policy_authority = attitude_authority

        if self.debug and self.step_idx % 10 == 0:
            logger.info(
                "real_action_filter | z=%.3f r=%.2f vxy=%.2f tilt=%.1f° "
                "z_auth=%.2f att_auth=%.2f | raw=%s | safe=%s",
                z,
                radius,
                vxy,
                float(np.rad2deg(tilt)),
                z_authority,
                attitude_authority,
                np.round(raw, 3).tolist(),
                np.round(safe, 3).tolist(),
            )

        return safe.astype(np.float32)

    def _attitude_pd(
        self,
        roll_cmd: float,
        pitch_cmd: float,
        yawrate_cmd: float,
        state: np.ndarray,
    ) -> np.ndarray:
        qw, qx, qy, qz = state[3], state[4], state[5], state[6]
        wx, wy, wz = float(state[10]), float(state[11]), float(state[12])

        roll, pitch, _ = self._quat_to_euler(qw, qx, qy, qz)

        tau_roll = self.att_kp * (roll_cmd - roll) - self.att_kd * wx
        tau_pitch = self.att_kp * (pitch_cmd - pitch) - self.att_kd * wy
        tau_yaw = self.yaw_kp * (yawrate_cmd - wz) - self.yaw_kd * wz

        return np.clip(
            np.array([tau_roll, tau_pitch, tau_yaw], dtype=np.float32),
            -1.0,
            1.0,
        )

    def _vertical_pd(self, vz_cmd: float, state: np.ndarray) -> float:
        return float(
            np.clip(
                self.HOVER_THRUST - self.vz_kp * (float(state[9]) - vz_cmd),
                self.tmin,
                self.tmax,
            )
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Actuator dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _smooth_thrust(self, u_scalar: float) -> float:
        du = float(np.clip(u_scalar - self.u_cmd, -self.max_du, self.max_du))
        u_slewed = self.u_cmd + du
        new_u = (1.0 - self.alpha) * self.u_cmd + self.alpha * u_slewed

        self.last_du = float(abs(new_u - self.u_cmd))
        self.u_cmd = float(new_u)

        return self.u_cmd

    def _sim_thrust_to_counts(self, u_newton: float) -> int:
        u = float(np.clip(u_newton, self.tmin, self.tmax))
        counts = int(round(u * self.counts_per_newton))
        return int(np.clip(counts, MIN_THRUST_COUNTS, MAX_THRUST_COUNTS))

    def _send_setpoint(
        self,
        u_newton: float,
        roll_cmd_rad: float,
        pitch_cmd_rad: float,
        yawrate_cmd_rad: float,
    ) -> None:
        try:
            cf = self.observer.cf
        except RuntimeError:
            logger.warning("No Crazyflie handle in _send_setpoint.")
            return

        thrust_counts = self._sim_thrust_to_counts(u_newton)
        roll_deg = float(np.rad2deg(roll_cmd_rad))
        pitch_deg = float(np.rad2deg(pitch_cmd_rad))
        yawrate_deg = float(np.rad2deg(yawrate_cmd_rad))

        cf.commander.send_setpoint(roll_deg, pitch_deg, yawrate_deg, thrust_counts)

    def _record_moments(self, m_vec: np.ndarray) -> None:
        m = np.clip(np.asarray(m_vec, dtype=np.float32), -1.0, 1.0)
        self.last_dm = float(np.linalg.norm(m - self.last_moments, ord=2))
        self.last_moments = m.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Tilt / radius
    # ──────────────────────────────────────────────────────────────────────────

    def _tilt_and_radius(self, global_state: np.ndarray) -> Tuple[float, float]:
        qx, qy = float(global_state[4]), float(global_state[5])
        tilt = float(2.0 * np.arcsin(float(np.clip(np.sqrt(qx * qx + qy * qy), 0.0, 1.0))))

        r = float(
            np.sqrt(
                (global_state[0] - self.spawn_xy[0]) ** 2
                + (global_state[1] - self.spawn_xy[1]) ** 2
            )
        )

        return tilt, r

    # ──────────────────────────────────────────────────────────────────────────
    # Gym API: reset
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.u_cmd = self.HOVER_THRUST
        self.last_du = 0.0
        self.last_moments[:] = 0.0
        self.last_dm = 0.0

        self.hover_count = 0
        self.step_idx = 0
        self.ground_steps = 0
        self.prev_dz = 0.0
        self.phase = "HOVER"

        self.landing_step_idx = 0
        self.landing_beta = 0.0
        self.landing_mode = "DESCEND"
        self.landing_catch_steps = 0
        self.pre_landing_reason = None

        self.last_raw_action[:] = 0.0
        self.last_safe_action[:] = 0.0
        self.last_policy_authority = 0.0

        # Acquire stable spawn pose.
        samples = []
        t0 = time.time()
        global_state = None

        while time.time() - t0 < 5.0:
            s = self._get_single_obs_global(timeout=self.state_timeout_s)

            if s is not None:
                samples.append(s)

            if len(samples) >= 30:
                arr = np.stack(samples[-30:], axis=0)

                xyz_std = np.std(arr[:, 0:3], axis=0)
                z_med = float(np.median(arr[:, 2]))

                if (
                    xyz_std[0] < 0.08
                    and xyz_std[1] < 0.08
                    and xyz_std[2] < 0.05
                    and -0.05 <= z_med <= 1.50
                ):
                    global_state = np.median(arr, axis=0).astype(np.float32)
                    break

            time.sleep(self.dt)

        if global_state is None:
            self._send_safe_stop()
            raise RuntimeError("Could not obtain stable initial state at reset.")

        self.spawn_xy[0] = float(global_state[0])
        self.spawn_xy[1] = float(global_state[1])
        self.spawn_z = float(global_state[2])

        self.target_z_abs = self.target_z
        self.hard_ceiling = self.target_z_abs + self.hard_ceiling_margin

        self.obs_stack.clear()

        first = self._to_agent_obs(global_state)
        for _ in range(self.n_stack):
            self.obs_stack.append(first.copy())

        obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)

        info = {
            "spawn_xy": self.spawn_xy.copy(),
            "spawn_z": float(self.spawn_z),
            "target_z_abs": float(self.target_z_abs),
            "raw_action": self.last_raw_action.copy(),
            "safe_action": self.last_safe_action.copy(),
            "policy_authority": self.last_policy_authority,
        }

        return obs, info

    # ──────────────────────────────────────────────────────────────────────────
    # Gym API: step
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)

        if a.shape[0] != 4:
            raise ValueError(f"Action must be shape (4,), got {a.shape}")

        if self.auto_landing and self.phase == "LANDING":
            return self._step_landing(a)

        a = np.clip(a, -1.0, 1.0)

        state_now = self._get_single_obs_global(timeout=self.state_timeout_s)
        if state_now is None:
            self._send_safe_stop()
            obs = self._current_obs_or_blank()
            return obs, -1.0, True, True, {"error": "no_state_from_observer"}

        if self.real_action_filter_enabled:
            a = self._filter_real_policy_action(a, state_now)
        else:
            self.last_raw_action = a.astype(np.float32)
            self.last_safe_action = a.astype(np.float32)
            self.last_policy_authority = 1.0

        roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd = self._decode_commander(a)

        m_vec = self._attitude_pd(roll_cmd, pitch_cmd, yawrate_cmd, state_now)
        u_req = self._vertical_pd(vz_cmd, state_now)

        u_smoothed = self._smooth_thrust(u_req)
        self._record_moments(m_vec)

        self._send_setpoint(u_smoothed, roll_cmd, pitch_cmd, yawrate_cmd)

        time.sleep(self.dt)

        state_next = self._get_single_obs_global(timeout=self.state_timeout_s)
        if state_next is None:
            self._send_safe_stop()
            obs = self._current_obs_or_blank()
            return obs, -1.0, True, True, {"error": "state_stream_lost"}

        single_next = self._to_agent_obs(state_next)
        self.obs_stack.append(single_next)

        obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)

        self.step_idx += 1

        return self._reward_and_termination(obs, state_next)

    # ──────────────────────────────────────────────────────────────────────────
    # Reward + termination
    # ──────────────────────────────────────────────────────────────────────────

    def _reward_and_termination(
        self,
        obs: np.ndarray,
        state_next: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        x2, y2, z2 = float(state_next[0]), float(state_next[1]), float(state_next[2])
        qx2, qy2 = float(state_next[4]), float(state_next[5])
        vx2, vy2, vz2 = float(state_next[7]), float(state_next[8]), float(state_next[9])
        wx2, wy2, wz2 = float(state_next[10]), float(state_next[11]), float(state_next[12])

        r_rad2 = float(np.sqrt((x2 - self.spawn_xy[0]) ** 2 + (y2 - self.spawn_xy[1]) ** 2))

        tilt_sin2 = float(np.clip(np.sqrt(qx2 * qx2 + qy2 * qy2), 0.0, 1.0))
        tilt_angle2 = float(2.0 * np.arcsin(tilt_sin2))
        tilt_deg2 = float(np.rad2deg(tilt_angle2))

        on_ground = z2 < self.ground_z_threshold
        self.ground_steps = self.ground_steps + 1 if on_ground else 0

        dz = z2 - self.target_z_abs
        vxy2 = vx2 * vx2 + vy2 * vy2
        omega2 = wx2 * wx2 + wy2 * wy2 + wz2 * wz2

        vz_desired = float(np.clip(-self.ff_k * dz, -self.max_vz_ff, self.max_vz_ff))

        cost = (
            self.w_z * (dz / self.z_scale) ** 2
            + self.w_vz * ((vz2 - vz_desired) / self.vz_scale) ** 2
            + self.w_r * (r_rad2 / self.r_scale) ** 2
            + self.w_vxy * (vxy2 / self.vxy_scale**2)
            + self.w_tilt * (tilt_angle2 / self.tilt_scale) ** 2
            + self.w_omega * (omega2 / self.omega_scale**2)
            + self.w_smooth_u * (self.last_du / self.du_scale) ** 2
            + self.w_smooth_m * (self.last_dm / self.dm_scale) ** 2
        )

        dense = float(np.clip(1.0 - cost, -5.0, 2.0))
        reward = dense / max(1, self.max_steps)

        if abs(dz) < abs(self.prev_dz):
            reward += 0.3 / max(1, self.max_steps)

        self.prev_dz = dz

        if on_ground:
            reward -= 0.1 / max(1, self.max_steps)

        stable = (
            abs(dz) <= self.band
            and abs(vz2) < 0.05
            and tilt_angle2 < np.deg2rad(10.0)
            and r_rad2 < 0.30
            and np.sqrt(vxy2) < 0.15
            and not on_ground
        )

        if stable:
            self.hover_count += 1
            reward += 0.2 / max(1, self.max_steps)
        else:
            self.hover_count = 0

        info_base = {
            "hover_steps": self.hover_count,
            "tilt_deg": tilt_deg2,
            "radius": r_rad2,
            "vz": vz2,
            "z": z2,
            "att_scale": 1.0,
            "raw_action": self.last_raw_action.copy(),
            "safe_action": self.last_safe_action.copy(),
            "policy_authority": self.last_policy_authority,
        }

        # Success
        if self.hover_count >= self.hover_required:
            reward += 1.0
            info = dict(info_base)
            info["success"] = True

            if self.auto_landing:
                self._start_landing_phase("success")
                info["phase"] = "landing_start"
                return obs, float(np.clip(reward, -5.0, 2.0)), False, False, info

            self._send_safe_stop()
            return obs, float(np.clip(reward, -5.0, 2.0)), True, False, info

        # Terminations
        if self.ground_steps >= self.max_ground_steps:
            self._send_safe_stop()
            info = dict(info_base)
            info.update({"crash": True, "reason": "stalled_on_ground"})
            return obs, -1.0, True, False, info

        if z2 < 0.01 or not np.all(np.isfinite(obs)):
            self._send_safe_stop()
            info = dict(info_base)
            info.update({"crash": True, "reason": "nan_or_below_ground"})
            return obs, -1.0, True, False, info

        if tilt_angle2 > np.deg2rad(self.real_flip_stop_deg):
            self._send_safe_stop()
            info = dict(info_base)
            info.update({"crash": True, "reason": "tilt_limit"})
            return obs, -1.0, True, False, info

        if z2 > self.hard_ceiling:
            self._send_safe_stop()
            info = dict(info_base)
            info.update({"ceiling": True, "reason": "hard_ceiling"})
            return obs, -1.0, True, False, info

        if r_rad2 > self.safety_radius:
            self._send_safe_stop()
            info = dict(info_base)
            info.update({"crash": True, "reason": "out_of_bounds"})
            return obs, -1.0, True, False, info

        if self.step_idx >= self.max_steps:
            info = dict(info_base)
            info["timeout"] = True

            if self.auto_landing:
                self._start_landing_phase("timeout")
                info["phase"] = "landing_start"
                return obs, float(np.clip(reward, -5.0, 2.0)), False, False, info

            self._send_safe_stop()
            return obs, float(np.clip(reward, -5.0, 2.0)), False, True, info

        return obs, float(np.clip(reward, -5.0, 2.0)), False, False, info_base

    # ──────────────────────────────────────────────────────────────────────────
    # Auto-landing
    # ──────────────────────────────────────────────────────────────────────────

    def _start_landing_phase(self, reason: str) -> None:
        self.phase = "LANDING"
        self.landing_step_idx = 0
        self.landing_beta = 0.0
        self.landing_mode = "DESCEND"
        self.landing_catch_steps = 0
        self.pre_landing_reason = reason

    def _step_landing(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        a_pol = np.clip(np.asarray(action, dtype=np.float32).reshape(4), -1.0, 1.0)

        state = self._get_single_obs_global(timeout=self.state_timeout_s)
        if state is None:
            self._send_safe_stop()
            obs = self._current_obs_or_blank()
            return obs, 0.0, True, True, {"error": "state_stream_lost_in_landing"}

        z, vz = float(state[2]), float(state[9])
        vx, vy = float(state[7]), float(state[8])

        if self.real_action_filter_enabled:
            a_pol = self._filter_real_policy_action(a_pol, state)

        roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd_pol = self._decode_commander(a_pol)

        m_pol = self._attitude_pd(roll_cmd, pitch_cmd, yawrate_cmd, state)

        u_pol = float(
            np.clip(
                self.HOVER_THRUST - self.vz_kp * (vz - vz_cmd_pol),
                self.tmin,
                self.tmax,
            )
        )

        tilt, r = self._tilt_and_radius(state)
        tilt_deg = float(np.rad2deg(tilt))

        if tilt_deg > self.landing_tilt_abort_deg or r > self.landing_max_radius:
            self.landing_mode = "CATCH"
            self.landing_catch_steps = 0

        if self.landing_mode == "CATCH":
            v_des = 0.0
            self.landing_beta = max(0.0, self.landing_beta - 0.05)

            stabilised = (
                tilt_deg < self.landing_tilt_ok_deg
                and r < self.landing_safe_radius
                and abs(vx) < 0.2
                and abs(vy) < 0.2
            )

            self.landing_catch_steps = self.landing_catch_steps + 1 if stabilised else 0

            if self.landing_catch_steps > 50:
                self.landing_mode = "DESCEND"
        else:
            h = max(0.0, z - 0.03)

            if h > 0.8:
                v_des = self.landing_vz_fast
            elif h > 0.4:
                v_des = self.landing_vz_med
            elif h > 0.2:
                v_des = self.landing_vz_mid
            else:
                v_des = self.landing_vz_slow

            self.landing_beta = min(
                1.0,
                self.landing_step_idx / max(1, self.landing_beta_ramp_steps),
            )

        beta = float(self.landing_beta)

        u_land = float(
            np.clip(
                self.HOVER_THRUST - self.landing_k_vz * (vz - v_des),
                0.12,
                self.tmax,
            )
        )

        u = float(np.clip((1.0 - beta) * u_pol + beta * u_land, 0.12, self.tmax))

        u_smoothed = self._smooth_thrust(u)
        self._record_moments(m_pol)
        self._send_setpoint(u_smoothed, roll_cmd, pitch_cmd, yawrate_cmd)

        time.sleep(self.dt)

        next_state = self._get_single_obs_global(timeout=self.state_timeout_s)
        if next_state is None:
            self._send_safe_stop()
            obs = self._current_obs_or_blank()
            return obs, 0.0, True, True, {"error": "state_stream_lost_in_landing_post"}

        self.obs_stack.append(self._to_agent_obs(next_state))
        obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)

        self.step_idx += 1
        self.landing_step_idx += 1

        z2, vz2 = float(next_state[2]), float(next_state[9])
        tilt2, r2 = self._tilt_and_radius(next_state)
        tilt_deg2 = float(np.rad2deg(tilt2))

        close_to_ground = z2 <= 0.03
        slow_descent = abs(vz2) < 0.15
        upright = tilt_deg2 < 12.0
        inside_zone = r2 < self.landing_safe_radius

        landed = close_to_ground and slow_descent and upright and inside_zone
        emergency_ground = z2 <= 0.04 and self.landing_step_idx > 80
        timeout = self.landing_step_idx >= self.landing_max_steps

        terminated = False

        if landed or emergency_ground or timeout:
            self.phase = "HOVER"
            terminated = True
            self._send_safe_stop()

        return obs, 0.0, terminated, False, {
            "phase": "landing",
            "landing_mode": self.landing_mode,
            "landing_beta": beta,
            "landing_landed": landed or emergency_ground,
            "landing_timeout": timeout,
            "tilt_deg": tilt_deg2,
            "radius": r2,
            "vz": float(vz2),
            "z": float(z2),
            "pre_landing_reason": self.pre_landing_reason,
            "att_scale": 1.0,
            "raw_action": self.last_raw_action.copy(),
            "safe_action": self.last_safe_action.copy(),
            "policy_authority": self.last_policy_authority,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Misc
    # ──────────────────────────────────────────────────────────────────────────

    def _current_obs_or_blank(self) -> np.ndarray:
        if len(self.obs_stack) == self.n_stack:
            return np.concatenate(list(self.obs_stack)).astype(np.float32)

        if len(self.obs_stack) > 0:
            last = self.obs_stack[-1]
            return np.concatenate([last] * self.n_stack).astype(np.float32)

        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def get_altitude(self) -> float:
        s = self.observer.get_state()
        return float(s[2]) if s is not None else 0.0

    def cut_motors(self) -> None:
        self.u_cmd = 0.0
        self._send_safe_stop()

    def render(self) -> None:
        return

    def close(self) -> None:
        try:
            self._send_safe_stop()
        finally:
            try:
                self.observer.close()
            except Exception:
                pass
