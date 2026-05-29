"""
CrazyFlie Real Hardware Environment — mirror of CrazyFlieEnvVelocity2 (sim).

Drives a real Bitcraze Crazyflie 2.x via cflib using the same:
  - 13D state layout            : [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
  - 4D normalized commander     : [roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd]
  - Observation transform       : [x_rel, y_rel, z_abs, quat, lin_vel, ang_vel]
  - Frame stacking (n_stack=4)
  - Commander decode → PD attitude controller → thrust mapping
  - Thrust smoothing (slew + low-pass)
  - Auto-landing state machine (HOVER → LANDING with DESCEND/CATCH modes)

Domain randomisation is NOT applied here — this is the real drone, the noise
is genuine. All `obs_noise_std` / `action_noise_std` / etc. parameters from
the sim env are not present.

Connection:
  - cflib radio link via CrazyFlieStateObserver (handles state telemetry)
  - send_setpoint(roll_deg, pitch_deg, yawrate_deg, thrust_counts) for control

Thrust mapping (sim Newtons → CF PWM counts):
  At hover the real CF needs ~41940 counts to fight 0.033 kg × 9.81 m/s² ≈ 0.324 N.
  So counts_per_newton ≈ 41940 / 0.324 ≈ 129444. The sim's HOVER_THRUST is
  0.34335 N (gear=1.0, hover ctrl from cf2.xml keyframe). We normalize the
  sim thrust against that hover value, then scale to count units.
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
DEFAULT_M_REAL_KG     = 0.033        # Crazyflie 2.x mass (kg)
DEFAULT_G             = 9.81
DEFAULT_W_HOVER       = DEFAULT_M_REAL_KG * DEFAULT_G   # weight at hover (~0.324 N)

DEFAULT_U_HOVER_COUNTS = 41940.0     # PWM counts to hover real CF (well-known)
DEFAULT_COUNTS_PER_NEWTON = DEFAULT_U_HOVER_COUNTS / DEFAULT_W_HOVER

# CF firmware safe range for thrust setpoints
MIN_THRUST_COUNTS = 10001
MAX_THRUST_COUNTS = 60000

# Sim thrust range — sim env uses 0.0 to 0.4 N (cf2.xml ctrlrange).
# Hover thrust from sim is 0.34335 N (clipped from MuJoCo keyframe).
SIM_T_MIN          = 0.0
SIM_T_MAX          = 0.4
SIM_HOVER_THRUST_N = 0.34335


class CrazyFlieRealEnvVelocity(gym.Env):
    """
    Real-hardware twin of CrazyFlieEnvVelocity2 (sim). The action space, the
    observation space, and the inner-loop control transformations are
    identical so a sim-trained policy can be deployed without modification.

    Action  : Box(4,) in [-1, 1]   = [roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd]
    Obs     : Box(13 * n_stack,)   = stacked single-state vectors
    Reward  : same shaping function as sim (used for monitoring, not learning)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        uri: str,
        # ── Task ──────────────────────────────────────────────────────────────
        target_z: float = 1.0,
        max_steps: int = 1500,
        n_stack: int = 4,
        hover_band: float = 0.10,
        hover_required_steps: int = 300,
        hard_ceiling_margin: float = 2.0,
        # ── Safety ────────────────────────────────────────────────────────────
        safety_radius: float = 2.0,
        # ── Thrust smoothing (must match sim) ─────────────────────────────────
        thrust_lowpass_alpha: float = 0.25,
        thrust_slew_per_step: float = 0.08,
        # ── Auto-landing (must match sim) ─────────────────────────────────────
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
        self.uri    = uri
        self.debug  = debug
        self.dt     = log_period_ms / 1000.0   # control period (s)
        self.state_timeout_s = max(0.5, 5 * self.dt)

        self.observer = CrazyFlieStateObserver(uri=uri, log_period_ms=log_period_ms)

        # ── Physical constants ────────────────────────────────────────────────
        self.m_real  = float(m_real_kg)
        self.g       = float(g)
        self.w_hover = self.m_real * self.g

        self.u_hover_counts    = float(u_hover_counts)
        self.counts_per_newton = (
            float(counts_per_newton_override)
            if counts_per_newton_override is not None
            else DEFAULT_COUNTS_PER_NEWTON
        )

        # ── Task ──────────────────────────────────────────────────────────────
        self.target_z            = float(target_z)
        self.max_steps           = int(max_steps)
        self.band                = float(hover_band)
        self.hover_required      = int(hover_required_steps)
        self.hard_ceiling_margin = float(hard_ceiling_margin)

        # ── Safety ────────────────────────────────────────────────────────────
        self.safety_radius      = float(safety_radius)
        self.ground_z_threshold = 0.05
        self.max_ground_steps   = 100

        # ── Spaces (identical to sim) ─────────────────────────────────────────
        self.n_stack        = int(n_stack)
        self.obs_dim_single = 13
        hi = np.full(self.obs_dim_single * self.n_stack, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)
        self.obs_stack = deque(maxlen=self.n_stack)
        self.action_space = spaces.Box(
            low  = np.full(4, -1.0, dtype=np.float32),
            high = np.full(4, +1.0, dtype=np.float32),
            dtype = np.float32,
        )

        # ── Thrust smoothing (matches sim) ────────────────────────────────────
        # Sim envelope is 0.0 → 0.4 N. We smooth in Newtons, then convert
        # to CF counts only at the very end (in _send_thrust).
        self.tmin  = SIM_T_MIN
        self.tmax  = SIM_T_MAX
        self.HOVER_THRUST = float(np.clip(SIM_HOVER_THRUST_N, self.tmin, self.tmax))
        self.alpha   = float(thrust_lowpass_alpha)
        self.max_du  = float(thrust_slew_per_step)
        self.u_cmd   = self.HOVER_THRUST
        self.last_du = 0.0
        self.last_moments = np.zeros(3, dtype=np.float32)
        self.last_dm = 0.0

        # ── Commander limits for REAL deployment ──────────────────────────────
        # This is still pure policy control: the policy action is sent directly
        # into env.step(action). These values only define how [-1, +1] maps to
        # real Crazyflie command units.
        #
        # Your logs showed the policy saturating nearly every action axis.
        # With 70° roll/pitch, saturation immediately flips the drone.
        # With ~4° roll/pitch, it gets off the ground but drifts slowly.
        # So use conservative real-world command scaling here.
        self.max_roll_deg    = 4.0
        self.max_pitch_deg   = 4.0
        self.max_yawrate_deg = 4.0
        self.max_vz_cmd      = 0.35
        self.max_roll_rad  = np.deg2rad(self.max_roll_deg)
        self.max_pitch_rad = np.deg2rad(self.max_pitch_deg)
        self.max_yawrate   = np.deg2rad(self.max_yawrate_deg)

        # ── Attitude PD (match sim) ───────────────────────────────────────────
        # On the real drone these gains drive the firmware's setpoint generator
        # — the firmware itself runs an inner PID at ~500 Hz against the
        # roll/pitch angles we send. We compute identical PD output for
        # monitoring + reward, but the actual flight uses send_setpoint.
        self.att_kp = 6.0
        self.att_kd = 0.3
        self.yaw_kp = 1.0
        self.yaw_kd = 0.05
        self.vz_kp  = 0.5

        # ── Auto-landing (match sim) ──────────────────────────────────────────
        self.auto_landing = bool(auto_landing)
        self._init_landing_params()

        # ── Reward shaping (match sim, monitoring-only) ───────────────────────
        self.ff_k       = 1.2
        self.max_vz_ff  = 0.50
        self.z_scale    = 0.60
        self.vz_scale   = 0.45
        self.r_scale    = 0.30
        self.vxy_scale  = 0.20
        self.tilt_scale  = np.deg2rad(10.0)
        self.omega_scale = np.deg2rad(100.0)
        self.du_scale    = 0.02
        self.dm_scale    = 0.40
        self.w_z, self.w_vz       = 1.5, 0.8
        self.w_r, self.w_vxy      = 0.5, 0.4
        self.w_tilt, self.w_omega = 1.2, 0.02
        self.w_smooth_u           = 0.05
        self.w_smooth_m           = 0.05
        self.prev_dz = 0.0

        # ── Episode state ─────────────────────────────────────────────────────
        self.step_idx     = 0
        self.hover_count  = 0
        self.ground_steps = 0
        self.phase        = "HOVER"
        self.target_z_abs = self.target_z
        self.hard_ceiling = self.target_z + self.hard_ceiling_margin

        # Spawn frame (filled at reset)
        self.spawn_xy = np.zeros(2, dtype=np.float64)
        self.spawn_z  = 0.0

        # Connect & prime the radio link
        self._connect_and_prime()

    # ──────────────────────────────────────────────────────────────────────────
    def _init_landing_params(self):
        # Identical to sim landing logic
        self.landing_max_radius      = 0.8
        self.landing_safe_radius     = 0.5
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
    # Hardware setup
    # ──────────────────────────────────────────────────────────────────────────
    def _connect_and_prime(self) -> None:
        """Open the radio link and unlock thrust protection by sending zeros."""
        if self.debug:
            logger.info("CrazyFlieRealEnvVelocity: connecting to %s", self.uri)
        self.observer.connect()

        # Wait until state stream is stable
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

        if self.debug:
            logger.info("Unlocking thrust protection with zero-thrust setpoints...")
        for _ in range(40):
            cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
            time.sleep(self.dt)

    def _send_safe_stop(self, n: int = 40) -> None:
        """Spam zero-thrust setpoints to make sure the motors stop."""
        try:
            cf = self.observer.cf
        except RuntimeError:
            return
        for _ in range(n):
            cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
            time.sleep(self.dt)

    def emergency_stop(self) -> None:
        """Public alias for clean external use."""
        self._send_safe_stop()

    # ──────────────────────────────────────────────────────────────────────────
    # Observation: 13D global state from Crazyflie -> agent obs (xy-relative, z-abs)
    # ──────────────────────────────────────────────────────────────────────────
    def _get_single_obs_global(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Block until a fresh 13D global state is available, else return None."""
        timeout = self.state_timeout_s if timeout is None else timeout
        t0 = time.time()
        while time.time() - t0 < timeout:
            state = self.observer.get_state()
            if state is not None:
                state = np.asarray(state, dtype=np.float32)
                if state.shape == (13,):
                    return state
            time.sleep(self.dt)
        return None

    def _to_agent_obs(self, global_state: np.ndarray) -> np.ndarray:
        """
        Convert global (x,y,z, quat, vel, omega) to the layout the policy
        was trained on:  [x_rel, y_rel, z_abs, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz].
        Mirrors `_get_single_obs` + the spawn-relative remap done in
        `_apply_obs_noise` of the sim env (with noise removed).
        """
        s   = global_state.astype(np.float32).copy()
        pos = np.array([
            s[0] - float(self.spawn_xy[0]),
            s[1] - float(self.spawn_xy[1]),
            s[2],   # z absolute (matches sim observation layout)
        ], dtype=np.float32)
        quat = s[3:7].copy()
        n = float(np.linalg.norm(quat))
        if n > 1e-6:
            quat /= n
        vel  = s[7:10].astype(np.float32)
        omg  = s[10:13].astype(np.float32)
        return np.concatenate([pos, quat, vel, omg]).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Controllers (identical math to sim)
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _quat_to_euler(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
        roll  = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
        sinp  = float(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0))
        pitch = np.sign(sinp) * np.pi / 2.0 if abs(sinp) >= 1.0 else float(np.arcsin(sinp))
        yaw   = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
        return float(roll), float(pitch), float(yaw)

    def _decode_commander(self, a_norm: np.ndarray) -> Tuple[float, float, float, float]:
        """[-1,1]^4 → (roll_rad, pitch_rad, yawrate_rad/s, vz m/s-ish)."""
        a = np.clip(np.asarray(a_norm, dtype=np.float32).reshape(4), -1.0, 1.0)
        return (
            float(a[0]) * self.max_roll_rad,
            float(a[1]) * self.max_pitch_rad,
            float(a[2]) * self.max_yawrate,
            float(a[3]) * self.max_vz_cmd,
        )

    def _attitude_pd(
        self, roll_cmd: float, pitch_cmd: float, yawrate_cmd: float, state: np.ndarray
    ) -> np.ndarray:
        """Same PD as sim — used for monitoring + smoothness reward terms."""
        qw, qx, qy, qz = state[3], state[4], state[5], state[6]
        wx, wy, wz     = float(state[10]), float(state[11]), float(state[12])
        roll, pitch, _ = self._quat_to_euler(qw, qx, qy, qz)
        tau_roll  = self.att_kp * (roll_cmd  - roll)  - self.att_kd * wx
        tau_pitch = self.att_kp * (pitch_cmd - pitch) - self.att_kd * wy
        tau_yaw   = self.yaw_kp * (yawrate_cmd - wz)  - self.yaw_kd * wz
        return np.clip(np.array([tau_roll, tau_pitch, tau_yaw], dtype=np.float32), -1.0, 1.0)

    def _vertical_pd(self, vz_cmd: float, state: np.ndarray) -> float:
        """vz error → thrust (Newtons in sim frame). Matches sim."""
        return float(np.clip(
            self.HOVER_THRUST - self.vz_kp * (float(state[9]) - vz_cmd),
            self.tmin, self.tmax,
        ))

    # ──────────────────────────────────────────────────────────────────────────
    # Actuator dispatch — apply thrust smoothing then send to CF
    # ──────────────────────────────────────────────────────────────────────────
    def _smooth_thrust(self, u_scalar: float) -> float:
        """Same slew + low-pass as sim's `_apply_thrust`. Returns smoothed u in N."""
        du       = float(np.clip(u_scalar - self.u_cmd, -self.max_du, self.max_du))
        u_slewed = self.u_cmd + du
        new_u    = (1.0 - self.alpha) * self.u_cmd + self.alpha * u_slewed
        self.last_du = float(abs(new_u - self.u_cmd))
        self.u_cmd   = float(new_u)
        return self.u_cmd

    def _sim_thrust_to_counts(self, u_newton: float) -> int:
        """Map sim-frame Newtons (0..tmax) → Crazyflie PWM counts."""
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
        """
        Send roll/pitch (deg), yawrate (deg/s), thrust (counts) to the CF.
        Note: cflib send_setpoint convention is (roll, pitch, yawrate, thrust).
        """
        try:
            cf = self.observer.cf
        except RuntimeError:
            logger.warning("No Crazyflie handle in _send_setpoint.")
            return

        thrust_counts = self._sim_thrust_to_counts(u_newton)
        roll_deg     = float(np.rad2deg(roll_cmd_rad))
        pitch_deg    = float(np.rad2deg(pitch_cmd_rad))
        yawrate_deg  = float(np.rad2deg(yawrate_cmd_rad))

        cf.commander.send_setpoint(roll_deg, pitch_deg, yawrate_deg, thrust_counts)

    def _record_moments(self, m_vec: np.ndarray) -> None:
        """Track moment smoothness for reward shaping (monitoring only)."""
        m = np.clip(np.asarray(m_vec, dtype=np.float32), -1.0, 1.0)
        self.last_dm      = float(np.linalg.norm(m - self.last_moments, ord=2))
        self.last_moments = m.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Tilt + radius helper (uses observer's quaternion)
    # ──────────────────────────────────────────────────────────────────────────
    def _tilt_and_radius(self, global_state: np.ndarray) -> Tuple[float, float]:
        qx, qy = float(global_state[4]), float(global_state[5])
        tilt = float(2.0 * np.arcsin(float(np.clip(np.sqrt(qx*qx + qy*qy), 0.0, 1.0))))
        r    = float(np.sqrt(
            (global_state[0] - self.spawn_xy[0])**2 +
            (global_state[1] - self.spawn_xy[1])**2,
        ))
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

        # Rebuild controller state
        self.u_cmd        = self.HOVER_THRUST
        self.last_du      = 0.0
        self.last_moments[:] = 0.0
        self.last_dm      = 0.0
        self.hover_count  = 0
        self.step_idx     = 0
        self.ground_steps = 0
        self.prev_dz      = 0.0
        self.phase        = "HOVER"
        self.landing_step_idx    = 0
        self.landing_beta        = 0.0
        self.landing_mode        = "DESCEND"
        self.landing_catch_steps = 0
        self.pre_landing_reason  = None

        # Acquire spawn pose from real drone — anchors the relative xy frame
        global_state = None
        t0 = time.time()
        while global_state is None:
            global_state = self._get_single_obs_global(timeout=self.state_timeout_s)
            if global_state is None and time.time() - t0 > 3.0:
                raise RuntimeError("Could not obtain initial state at reset.")
            if global_state is None:
                time.sleep(self.dt)

        self.spawn_xy[0] = float(global_state[0])
        self.spawn_xy[1] = float(global_state[1])
        self.spawn_z     = float(global_state[2])
        self.target_z_abs = self.target_z   # sim env uses absolute z too
        self.hard_ceiling = self.target_z_abs + self.hard_ceiling_margin

        self.obs_stack.clear()
        first = self._to_agent_obs(global_state)
        for _ in range(self.n_stack):
            self.obs_stack.append(first.copy())

        info = {
            "spawn_xy":     self.spawn_xy.copy(),
            "spawn_z":      float(self.spawn_z),
            "target_z_abs": float(self.target_z_abs),
        }
        obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)
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

        # Read fresh state — needed for the inner controllers
        state_now = self._get_single_obs_global(timeout=self.state_timeout_s)
        if state_now is None:
            self._send_safe_stop()
            obs = self._current_obs_or_blank()
            return obs, -1.0, True, True, {"error": "no_state_from_observer"}

        # Decode commander → PD → thrust (Newtons in sim frame)
        roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd = self._decode_commander(a)
        m_vec = self._attitude_pd(roll_cmd, pitch_cmd, yawrate_cmd, state_now)
        u_req = self._vertical_pd(vz_cmd, state_now)

        # Smooth thrust (slew + low-pass) before sending to CF
        u_smoothed = self._smooth_thrust(u_req)
        self._record_moments(m_vec)

        # Hand commander values directly to CF firmware (it runs the inner PID).
        self._send_setpoint(u_smoothed, roll_cmd, pitch_cmd, yawrate_cmd)

        # Pace the loop — observer is logging at log_period_ms anyway
        time.sleep(self.dt)

        # Sample next state for observation + reward + termination logic
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
    # Reward + termination — mirrors sim env step() bottom half
    # ──────────────────────────────────────────────────────────────────────────
    def _reward_and_termination(
        self, obs: np.ndarray, state_next: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        x2, y2, z2     = float(state_next[0]), float(state_next[1]), float(state_next[2])
        qx2, qy2       = float(state_next[4]), float(state_next[5])
        vx2, vy2, vz2  = float(state_next[7]), float(state_next[8]), float(state_next[9])
        wx2, wy2, wz2  = float(state_next[10]), float(state_next[11]), float(state_next[12])

        r_rad2      = float(np.sqrt((x2 - self.spawn_xy[0])**2 + (y2 - self.spawn_xy[1])**2))
        tilt_sin2   = float(np.clip(np.sqrt(qx2*qx2 + qy2*qy2), 0.0, 1.0))
        tilt_angle2 = float(2.0 * np.arcsin(tilt_sin2))
        tilt_deg2   = float(np.rad2deg(tilt_angle2))

        on_ground = z2 < self.ground_z_threshold
        self.ground_steps = self.ground_steps + 1 if on_ground else 0

        dz     = z2 - self.target_z_abs
        vxy2   = vx2*vx2 + vy2*vy2
        omega2 = wx2*wx2 + wy2*wy2 + wz2*wz2
        vz_desired = float(np.clip(-self.ff_k * dz, -self.max_vz_ff, self.max_vz_ff))

        cost = (
            self.w_z      * (dz / self.z_scale) ** 2
            + self.w_vz   * ((vz2 - vz_desired) / self.vz_scale) ** 2
            + self.w_r    * (r_rad2 / self.r_scale) ** 2
            + self.w_vxy  * (vxy2 / self.vxy_scale**2)
            + self.w_tilt * (tilt_angle2 / self.tilt_scale) ** 2
            + self.w_omega * (omega2 / self.omega_scale**2)
            + self.w_smooth_u * (self.last_du / self.du_scale) ** 2
            + self.w_smooth_m * (self.last_dm / self.dm_scale) ** 2
        )
        dense  = float(np.clip(1.0 - cost, -5.0, 2.0))
        reward = dense / max(1, self.max_steps)

        # Altitude progress bonus (matches sim)
        if abs(dz) < abs(self.prev_dz):
            reward += 0.3 / max(1, self.max_steps)
        self.prev_dz = dz

        if on_ground:
            reward -= 0.1 / max(1, self.max_steps)

        # ── Hover tracking ────────────────────────────────────────────────────
        stable = (
            abs(dz)         <= self.band
            and abs(vz2)    <  0.05
            and tilt_angle2 <  np.deg2rad(10.0)
            and r_rad2      <  0.30
            and np.sqrt(vxy2) < 0.15
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
            self._send_safe_stop()
            return obs, float(np.clip(reward, -5.0, 2.0)), True, False, info

        # ── Terminations ──────────────────────────────────────────────────────
        if self.ground_steps >= self.max_ground_steps:
            self._send_safe_stop()
            return obs, -1.0, True, False, {"crash": True, "reason": "stalled_on_ground"}
        if z2 < 0.01 or not np.all(np.isfinite(obs)):
            self._send_safe_stop()
            return obs, -1.0, True, False, {"crash": True, "reason": "nan_or_below_ground"}
        if tilt_angle2 > np.deg2rad(70.0):
            self._send_safe_stop()
            return obs, -1.0, True, False, {"crash": True, "reason": "flipped"}
        if z2 > self.hard_ceiling:
            self._send_safe_stop()
            return obs, -1.0, True, False, {"ceiling": True, "reason": "hard_ceiling"}
        if r_rad2 > self.safety_radius:
            self._send_safe_stop()
            return obs, -1.0, True, False, {"crash": True, "reason": "out_of_bounds"}
        if self.step_idx >= self.max_steps:
            info = {"hover_steps": self.hover_count, "timeout": True, "att_scale": 1.0}
            if self.auto_landing:
                self._start_landing_phase("timeout")
                info["phase"] = "landing_start"
                return obs, float(np.clip(reward, -5.0, 2.0)), False, False, info
            self._send_safe_stop()
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
    # Auto-landing (mirrors sim _step_landing exactly)
    # ──────────────────────────────────────────────────────────────────────────
    def _start_landing_phase(self, reason: str):
        self.phase               = "LANDING"
        self.landing_step_idx    = 0
        self.landing_beta        = 0.0
        self.landing_mode        = "DESCEND"
        self.landing_catch_steps = 0
        self.pre_landing_reason  = reason

    def _step_landing(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        a_pol = np.clip(np.asarray(action, dtype=np.float32).reshape(4), -1.0, 1.0)

        state = self._get_single_obs_global(timeout=self.state_timeout_s)
        if state is None:
            self._send_safe_stop()
            obs = self._current_obs_or_blank()
            return obs, 0.0, True, True, {"error": "state_stream_lost_in_landing"}

        z, vz   = float(state[2]), float(state[9])
        vx, vy  = float(state[7]), float(state[8])

        roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd_pol = self._decode_commander(a_pol)
        m_pol = self._attitude_pd(roll_cmd, pitch_cmd, yawrate_cmd, state)
        u_pol = float(np.clip(
            self.HOVER_THRUST - self.vz_kp * (vz - vz_cmd_pol), self.tmin, self.tmax,
        ))

        tilt, r  = self._tilt_and_radius(state)
        tilt_deg = float(np.rad2deg(tilt))

        if tilt_deg > self.landing_tilt_abort_deg or r > self.landing_max_radius:
            self.landing_mode        = "CATCH"
            self.landing_catch_steps = 0

        if self.landing_mode == "CATCH":
            v_des = 0.0
            self.landing_beta = max(0.0, self.landing_beta - 0.05)
            stabilised = (
                tilt_deg < self.landing_tilt_ok_deg and r < self.landing_safe_radius
                and abs(vx) < 0.2 and abs(vy) < 0.2
            )
            self.landing_catch_steps = self.landing_catch_steps + 1 if stabilised else 0
            if self.landing_catch_steps > 50:
                self.landing_mode = "DESCEND"
        else:
            h = max(0.0, z - 0.03)
            if   h > 0.8: v_des = self.landing_vz_fast
            elif h > 0.4: v_des = self.landing_vz_med
            elif h > 0.2: v_des = self.landing_vz_mid
            else:         v_des = self.landing_vz_slow
            self.landing_beta = min(
                1.0, self.landing_step_idx / max(1, self.landing_beta_ramp_steps)
            )

        beta   = float(self.landing_beta)
        u_land = float(np.clip(self.HOVER_THRUST - self.landing_k_vz * (vz - v_des), 0.12, self.tmax))
        u      = float(np.clip((1.0 - beta) * u_pol + beta * u_land, 0.12, self.tmax))

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

        self.step_idx         += 1
        self.landing_step_idx += 1

        z2, vz2   = float(next_state[2]), float(next_state[9])
        tilt2, r2 = self._tilt_and_radius(next_state)
        tilt_deg2 = float(np.rad2deg(tilt2))

        close_to_ground = (z2 <= 0.03)
        slow_descent    = (abs(vz2) < 0.15)
        upright         = (tilt_deg2 < 12.0)
        inside_zone     = (r2 < self.landing_safe_radius)
        landed = close_to_ground and slow_descent and upright and inside_zone
        emergency_ground = (z2 <= 0.04 and self.landing_step_idx > 80)
        timeout = self.landing_step_idx >= self.landing_max_steps

        terminated = False
        if landed or emergency_ground or timeout:
            self.phase = "HOVER"
            terminated = True
            self._send_safe_stop()

        return obs, 0.0, terminated, False, {
            "phase":              "landing",
            "landing_mode":       self.landing_mode,
            "landing_beta":       beta,
            "landing_landed":     landed or emergency_ground,
            "landing_timeout":    timeout,
            "tilt_deg":           tilt_deg2,
            "radius":             r2,
            "vz":                 float(vz2),
            "z":                  float(z2),
            "pre_landing_reason": self.pre_landing_reason,
            "att_scale":          1.0,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Misc helpers
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
        # Real hardware — nothing to render
        return

    def close(self) -> None:
        try:
            self._send_safe_stop()
        finally:
            try:
                self.observer.close()
            except Exception:
                pass