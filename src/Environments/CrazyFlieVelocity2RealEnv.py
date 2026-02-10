
import time
import logging
from collections import deque
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Helper.CrazyFlieStateObserver import CrazyFlieStateObserver

logger = logging.getLogger(__name__)



DEFAULT_M_REAL_KG = 0.033          
DEFAULT_G = 9.81
DEFAULT_W = DEFAULT_M_REAL_KG * DEFAULT_G  

DEFAULT_U_HOVER_COUNTS = 41940.0
DEFAULT_COUNTS_PER_NEWTON = DEFAULT_U_HOVER_COUNTS / DEFAULT_W


MIN_THRUST_COUNTS = 10001
MAX_THRUST_COUNTS = 60000


MIN_SIM_THRUST = 0.0
MAX_SIM_THRUST = DEFAULT_W * 2.0    


class CrazyFlieRealEnvVelocity(gym.Env):
    """
    Real Crazyflie hover environment driven by a velocity-based policy.

    - Observation: stacked 13D state vector from CrazyFlieStateObserver
    - Action: 4D vector in [-1, 1]:
        a[0] → roll command   (left/right tilt)
        a[1] → pitch command  (forward/back tilt)
        a[2] → yaw rate       (spin)
        a[3] → vertical velocity setpoint (m/s-like, via PD → thrust)

    The reward structure mirrors the simulated CrazyFlieEnvVelocity2:
    - Track a target hover height
    - Stay near the lateral origin
    - Stay upright and smooth in attitude and thrust
    - Avoid high tilt, fast lateral motion, leaving a safety radius, or hitting ceilings
    - Bonus for long, smooth hovers inside a narrow band
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        uri: str,
        target_z: float = 1.0,
        max_steps: int = 600,
        frame_stack: int = 4,
        log_period_ms: int = 20,
        m_real_kg: float = DEFAULT_M_REAL_KG,
        g: float = DEFAULT_G,
        u_hover_counts: float = DEFAULT_U_HOVER_COUNTS,
        counts_per_newton_override: Optional[float] = None,
        auto_landing: bool = False,
        debug: bool = True,
    ) -> None:
        super().__init__()

        # ---- Connection / logging ----
        self.uri = uri
        self.debug = debug
        self.dt = log_period_ms / 1000.0
        self.state_timeout_s = max(0.5, 5 * self.dt)

        # Crazyflie state observer (handles the radio + log configs)
        self.observer = CrazyFlieStateObserver(uri=uri, log_period_ms=log_period_ms)

        # ---- Physical constants ----
        self.m_real = float(m_real_kg)
        self.g = float(g)
        self.w_hover = self.m_real * self.g

        self.u_hover_counts = float(u_hover_counts)
        self.counts_per_newton = (
            float(counts_per_newton_override)
            if counts_per_newton_override is not None
            else DEFAULT_COUNTS_PER_NEWTON
        )

        # Target altitude relative to spawn point (m)
        self.target_z_rel = float(target_z)

       
        # Vertical velocity → thrust PD controller
        self.vz_kp = 1.8
        self.vz_kd = 0.4
        self._last_vz_err = 0.0

        # Attitude PD for roll/pitch, yaw-rate
        self.att_kp = 6.0
        self.att_kd = 1.5
        self.yaw_kp = 0.15
        self.yaw_kd = 0.05

        # Attitude command limits (in radians / rad/s)
        self.max_roll_cmd_rad = np.deg2rad(20.0)
        self.max_pitch_cmd_rad = np.deg2rad(20.0)
        self.max_yawrate_rad = np.deg2rad(120.0)  
        self.max_vz_cmd = 1.0                     

        # ---- Attitude assist & scaling (mirrors sim structure) ----
        self.enable_upright_assist = True
        self.k_att = 0.6  

        # Scale attitude commands more aggressively away from hover to keep it tame
        self.att_base_scale = 0.90
        self.att_min_scale = 0.25

        # Near-ground attenuation: below this height above spawn, damp attitude
        self.near_ground_z = 0.30      
        self.near_ground_scale = 0.45

        # Tilt-based attenuation
        self.tilt_soft_deg = 6.0
        self.tilt_hard_deg = 25.0

        # Lateral radius shaping / safety
        self.lateral_soft_radius = 0.25   
        self.safety_radius = 1.5         

        # Lateral drift assist (small PD that nudges back to center)
        self.k_xy_p = 0.6
        self.k_xy_d = 0.2
        self.max_assist_deg = 15.0
        self.assist_base = 0.6

        # ---- Vertical safety ----
        self.soft_ceiling = 1.6
        self.hard_ceiling = 1.9
        self.safe_ground_height = 0.03   
        self.ground_stall_max_steps = 60  # how long you can sit on ground before "crash"

        # ---- Reward / success tracking ----
        self.hover_band_half_width = 0.10      # +/-10 cm around target
        self.hover_required_steps = 350        # continuous steps inside band for success

        # ---- Thrust smoothing state ----
        self.u_cmd = 0.0                     
        self.max_du = self.w_hover * 0.15     
        self.alpha = 0.4                      
        self.last_du = 0.0

        # "Moment" tracking (used for smoothness reward only, not actually applied on hardware)
        self.last_moments = np.zeros(3, dtype=np.float32)
        self.last_dm = 0.0

        # Track previous action (for command-rate penalty)
        self.prev_cmd = np.zeros(4, dtype=np.float32)

        # ---- Episode state ----
        self.max_steps = int(max_steps)
        self.frame_stack = int(frame_stack)

        self.step_idx = 0
        self.ground_steps = 0
        self.hover_steps = 0
        self.phase = "HOVER"  

       
        self._spawn_state: Optional[np.ndarray] = None
        self.target_z_abs: float = 0.0

        self._frame_buffer: deque[np.ndarray] = deque(maxlen=self.frame_stack)


        self._connect_and_prime()


        obs_dim = 13 * self.frame_stack
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )


    def _connect_and_prime(self) -> None:
        """Connect to Crazyflie via the observer and unlock thrust protection."""
        if self.debug:
            logger.info("CrazyFlieRealEnvVelocity: connecting to Crazyflie at %s", self.uri)

        self.observer.connect()

        t0 = time.time()
        timeout_s = 5.0
        while not getattr(self.observer, "is_ready", lambda: False)():
            if time.time() - t0 > timeout_s:
                raise RuntimeError("Timeout waiting for Crazyflie state to become ready.")
            time.sleep(0.01)

        cf = getattr(self.observer, "_cf", None)
        if cf is None:
            logger.warning("CrazyFlieRealEnvVelocity: observer has no Crazyflie handle; cannot prime motors.")
            return

        if self.debug:
            logger.info("CrazyFlieRealEnvVelocity: unlocking thrust protection with zero-thrust setpoints...")

        for _ in range(40):
            cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
            time.sleep(self.dt)

    def _send_safe_stop(self, n: int = 40) -> None:
        """Spam zero-thrust setpoints to make sure the motors stop."""
        cf = getattr(self.observer, "_cf", None)
        if cf is None:
            return

        for _ in range(n):
            cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
            time.sleep(self.dt)


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.step_idx = 0
        self.ground_steps = 0
        self.hover_steps = 0
        self.phase = "HOVER"

        self.u_cmd = 0.0
        self.last_du = 0.0
        self.last_moments[:] = 0.0
        self.last_dm = 0.0
        self.prev_cmd[:] = 0.0
        self._last_vz_err = 0.0

        state = None
        t0 = time.time()
        while state is None:
            state = self._get_single_obs_global(timeout=self.state_timeout_s)
            if state is None and time.time() - t0 > 3.0:
                raise RuntimeError("Could not obtain initial Crazyflie state during reset.")
            if state is None:
                time.sleep(self.dt)

        self._spawn_state = state.copy()
        x0, y0, z0 = self._spawn_state[0:3]
        self.target_z_abs = z0 + self.target_z_rel

        obs0 = self._transform_to_agent_obs(state)
        self._frame_buffer.clear()
        for _ in range(self.frame_stack):
            self._frame_buffer.append(obs0.copy())

        obs_stack = np.concatenate(list(self._frame_buffer)).astype(np.float32)

        info = {
            "spawn_state": self._spawn_state.copy(),
            "target_z_abs": float(self.target_z_abs),
        }
        return obs_stack, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 4:
            raise ValueError(f"Action must have shape (4,), got {action.shape}")

        info: Dict[str, Any] = {}
        terminated = False
        truncated = False

   
        state_prev = self._get_single_obs_global(timeout=self.state_timeout_s)
        if state_prev is None:

            self._send_safe_stop()
            obs = self._current_obs_or_blank()
            return obs, -50.0, True, True, {"error": "no_state_from_observer"}

        roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd = self._decode_commander(action)


        roll_cmd, pitch_cmd, yawrate_cmd = self._apply_attitude_assist_and_scaling(
            roll_cmd, pitch_cmd, yawrate_cmd, state_prev
        )

 
        u_scalar = self._vertical_pd(vz_cmd, state_prev)


        m_vec = self._attitude_pd(roll_cmd, pitch_cmd, yawrate_cmd, state_prev)
        m_clipped = np.clip(m_vec, -1.0, 1.0)
        dm = m_clipped - self.last_moments
        self.last_dm = float(np.linalg.norm(dm, ord=2))
        self.last_moments = m_clipped


        self._apply_thrust(u_scalar, roll_cmd, pitch_cmd, yawrate_cmd)


        time.sleep(self.dt)


        state_curr = self._get_single_obs_global(timeout=self.state_timeout_s)
        if state_curr is None:
            self._send_safe_stop()
            obs = self._current_obs_or_blank()
            return obs, -50.0, True, True, {"error": "state_stream_lost"}


        obs_curr = self._transform_to_agent_obs(state_curr)
        self._frame_buffer.append(obs_curr)
        obs_stack = np.concatenate(list(self._frame_buffer)).astype(np.float32)

        reward, terminated, truncated_rew, info_rew = self._compute_reward_and_termination(
            state_prev, state_curr, action, u_scalar
        )
        truncated = truncated or truncated_rew
        info.update(info_rew)

        self.step_idx += 1
        self.prev_cmd = action.copy()


        if self.step_idx >= self.max_steps and not terminated:
            truncated = True
            info.setdefault("timeout", True)

        return obs_stack, float(reward), bool(terminated), bool(truncated), info


    def _get_single_obs_global(self, timeout: float) -> Optional[np.ndarray]:
        """
        Get a single 13D state vector from the observer.

        Returns None if no fresh state is available within `timeout`.
        """
        t0 = time.time()
        state: Optional[np.ndarray] = None
        while state is None and time.time() - t0 < timeout:
            state = self.observer.get_state()
            if state is not None:
                break
            time.sleep(self.dt)

        if state is None:
            return None

        state = np.asarray(state, dtype=np.float32)
        if state.shape != (13,):
            logger.warning("CrazyFlieRealEnvVelocity: expected state shape (13,), got %s", state.shape)
            return None

        return state

    def _transform_to_agent_obs(self, global_state: np.ndarray) -> np.ndarray:
        """
        Map the raw 13D state into the same observation format that the sim env uses:

            [x_rel, y_rel, z_rel,
             qw, qx, qy, qz,
             vx, vy, vz,
             wx, wy, wz]
        """
        if self._spawn_state is None:
            raise RuntimeError("Spawn state not set before calling _transform_to_agent_obs.")

        x, y, z = global_state[0:3]
        qw, qx, qy, qz = global_state[3:7]
        vx, vy, vz = global_state[7:10]
        wx, wy, wz = global_state[10:13]

        x0, y0, z0 = self._spawn_state[0:3]

        pos_rel = np.array([x - x0, y - y0, z - z0], dtype=np.float32)
        quat = np.array([qw, qx, qy, qz], dtype=np.float32)  
        vel = np.array([vx, vy, vz], dtype=np.float32)
        omega = np.array([wx, wy, wz], dtype=np.float32)

        obs = np.concatenate([pos_rel, quat, vel, omega]).astype(np.float32)
        return obs

    def _decode_commander(self, action: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Map a normalized 4D action in [-1, 1] into physical commands:
        roll [rad], pitch [rad], yawrate [rad/s], vz_cmd [m/s-like].
        """
        a = np.clip(action, -1.0, 1.0).astype(np.float32)
        roll_norm, pitch_norm, yaw_norm, vz_norm = a

        roll_cmd = float(roll_norm * self.max_roll_cmd_rad)
        pitch_cmd = float(pitch_norm * self.max_pitch_cmd_rad)
        yawrate_cmd = float(yaw_norm * self.max_yawrate_rad)
        vz_cmd = float(vz_norm * self.max_vz_cmd)

        return roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd

    def _vertical_pd(self, vz_cmd: float, global_state: np.ndarray) -> float:
        """
        Vertical velocity PD: takes a `vz_cmd` (target vertical velocity)
        and measured vz, and returns a thrust value in Newtons around hover.

        This is the same idea as in the sim env: the RL agent works in
        "velocity space" and this controller turns that into thrust.
        """
        vz = float(global_state[9])  # index 9 = vz


        err = vz_cmd - vz
        derr = (err - self._last_vz_err) / self.dt
        self._last_vz_err = err


        u = self.w_hover + self.vz_kp * err + self.vz_kd * derr


        u = float(np.clip(u, MIN_SIM_THRUST, MAX_SIM_THRUST))
        return u

    def _attitude_pd(
        self,
        roll_cmd: float,
        pitch_cmd: float,
        yawrate_cmd: float,
        global_state: np.ndarray,
    ) -> np.ndarray:
        """
        PD controller in attitude space.

        We don't directly send torques to the Crazyflie, but this gives a
        "virtual" moment vector used for reward terms mirroring the sim env.
        """
        qw, qx, qy, qz = global_state[3:7]
        wx, wy, wz = global_state[10:13]

        roll, pitch, yaw = self._quat_to_rpy(np.array([qw, qx, qy, qz], dtype=np.float32))


        e_roll = roll_cmd - roll
        e_pitch = pitch_cmd - pitch


        e_yawrate = yawrate_cmd - wz


        tau_roll = self.att_kp * e_roll - self.att_kd * wx
        tau_pitch = self.att_kp * e_pitch - self.att_kd * wy
        tau_yaw = self.yaw_kp * e_yawrate - self.yaw_kd * wz


        scale = 10.0
        m_vec = np.array([tau_roll, tau_pitch, tau_yaw], dtype=np.float32) / scale
        return m_vec

    def _apply_attitude_assist_and_scaling(
        self,
        roll_cmd: float,
        pitch_cmd: float,
        yawrate_cmd: float,
        global_state: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Apply:
        - Upright assist (small PD towards level)
        - Near-ground damping
        - Tilt-based damping (soft/hard)
        - Lateral-distance damping
        - Optional lateral drift assist that nudges back to the center.
        """
        if self._spawn_state is None:
            return roll_cmd, pitch_cmd, yawrate_cmd

        x, y, z = global_state[0:3]
        qw, qx, qy, qz = global_state[3:7]
        vx, vy, vz = global_state[7:10]

        roll, pitch, yaw = self._quat_to_rpy(np.array([qw, qx, qy, qz], dtype=np.float32))
        tilt = float(np.sqrt(roll * roll + pitch * pitch))


        if self.enable_upright_assist:
            roll_cmd = roll_cmd - self.k_att * roll
            pitch_cmd = pitch_cmd - self.k_att * pitch


        scale = self.att_base_scale


        z_rel = z - self._spawn_state[2]
        if z_rel < self.near_ground_z:
            scale *= self.near_ground_scale


        tilt_soft = np.deg2rad(self.tilt_soft_deg)
        tilt_hard = np.deg2rad(self.tilt_hard_deg)
        if tilt > tilt_soft:

            t = (tilt_hard - tilt) / max(1e-6, (tilt_hard - tilt_soft))
            t = float(np.clip(t, 0.0, 1.0))
            scale *= t


        dx = x - self._spawn_state[0]
        dy = y - self._spawn_state[1]
        r = float(np.sqrt(dx * dx + dy * dy))
        if r > self.lateral_soft_radius:

            scale *= max(0.0, 1.0 - 0.5 * (r - self.lateral_soft_radius))


        scale = float(np.clip(scale, self.att_min_scale, 1.0))

        roll_cmd *= scale
        pitch_cmd *= scale
        yawrate_cmd *= scale


        in_hover_band = abs(z_rel - self.target_z_rel) < self.hover_band_half_width
        if in_hover_band:

            ax = -self.k_xy_p * dx - self.k_xy_d * vx
            ay = -self.k_xy_p * dy - self.k_xy_d * vy


            max_assist_rad = np.deg2rad(self.max_assist_deg)
            roll_assist = np.clip(-ay / self.g, -max_assist_rad, max_assist_rad)
            pitch_assist = np.clip(ax / self.g, -max_assist_rad, max_assist_rad)

            roll_cmd += self.assist_base * roll_assist
            pitch_cmd += self.assist_base * pitch_assist

        return roll_cmd, pitch_cmd, yawrate_cmd

    def _sim_thrust_to_int(self, u_newton: float) -> int:
        """
        Map a "sim-style" thrust level in Newtons (0..~2W) to Crazyflie PWM counts.
        """
        u_clipped = float(np.clip(u_newton, MIN_SIM_THRUST, MAX_SIM_THRUST))
        counts = int(round(u_clipped * self.counts_per_newton))
        return counts

    def _apply_thrust(
        self,
        u_scalar: float,
        roll_cmd: float,
        pitch_cmd: float,
        yawrate_cmd: float,
    ) -> None:
        """
        Smooth thrust, convert into motor counts, and send roll/pitch/yawrate + thrust
        to the Crazyflie via the underlying cflib commander.
        """

        du = float(np.clip(u_scalar - self.u_cmd, -self.max_du, self.max_du))
        u_slewed = self.u_cmd + du
        new_u = (1.0 - self.alpha) * self.u_cmd + self.alpha * u_slewed

        self.last_du = float(abs(new_u - self.u_cmd))
        self.u_cmd = float(new_u)

 
        thrust_counts = self._sim_thrust_to_int(self.u_cmd)
        thrust_counts = int(np.clip(thrust_counts, MIN_THRUST_COUNTS, MAX_THRUST_COUNTS))


        roll_deg = float(np.rad2deg(roll_cmd))
        pitch_deg = float(np.rad2deg(pitch_cmd))
        yawrate_deg = float(np.rad2deg(yawrate_cmd))

        cf = getattr(self.observer, "_cf", None)
        if cf is None:
            logger.warning(
                "CrazyFlieRealEnvVelocity: Crazyflie handle is None when trying to send_setpoint."
            )
            return

        cf.commander.send_setpoint(roll_deg, pitch_deg, yawrate_deg, thrust_counts)


    def _compute_reward_and_termination(
        self,
        state_prev: np.ndarray,
        state_curr: np.ndarray,
        action: np.ndarray,
        u_scalar: float,
    ) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """
        Reward shaping and termination logic. Closely follows the simulated environment:
        - Good things: close to target height, moving toward it, smooth motion.
        - Bad things: high tilt, lateral drift, jerky commands, ceilings, leaving safety bubble.
        """
        info: Dict[str, Any] = {}

        if self._spawn_state is None:
            raise RuntimeError("Spawn state not set in _compute_reward_and_termination.")

        x1, y1, z1 = state_prev[0:3]
        x2, y2, z2 = state_curr[0:3]
        qw2, qx2, qy2, qz2 = state_curr[3:7]
        vx2, vy2, vz2 = state_curr[7:10]
        wx2, wy2, wz2 = state_curr[10:13]

        x0, y0, z0 = self._spawn_state[0:3]

        dz_prev = z1 - self.target_z_abs
        dz = z2 - self.target_z_abs

        r_prev = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
        r = float(np.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2))


        roll2, pitch2, yaw2 = self._quat_to_rpy(np.array([qw2, qx2, qy2, qz2], dtype=np.float32))
        tilt = float(np.sqrt(roll2 * roll2 + pitch2 * pitch2))


        near = float(np.exp(-(dz / 0.12) ** 2))


        k_z = 8.0
        r_z = -k_z * dz * dz


        k_progress = 2.0
        r_progress = k_progress * (dz_prev * dz_prev - dz * dz)

  
        k_vz_near = 2.5
        k_vz_far = 0.15
        vz2_sq = vz2 * vz2
        r_vz = -(k_vz_near * near + k_vz_far * (1.0 - near)) * vz2_sq

        k_xy_center = 3.0
        k_xy_outer = 8.0
        r_xy_center = -k_xy_center * r * r

        r_xy_soft = 0.0
        if r > self.lateral_soft_radius:
            dr = r - self.lateral_soft_radius
            r_xy_soft = -k_xy_outer * dr * dr

 
        k_vxy = 0.5
        r_vxy = -k_vxy * (vx2 * vx2 + vy2 * vy2)


        k_tilt = 4.0
        k_omega = 0.02
        r_tilt = -k_tilt * tilt * tilt
        r_omega = -k_omega * (wx2 * wx2 + wy2 * wy2 + wz2 * wz2)


        k_du = 0.5
        k_m_abs = 0.1
        k_m_jump = 0.2

        r_thrust_smooth = -k_du * (self.last_du ** 2)

        m_mag = float(np.linalg.norm(self.last_moments, ord=2))
        r_moment_abs = -k_m_abs * (m_mag ** 2)
        r_moment_jump = -k_m_jump * (self.last_dm ** 2)

        action = np.asarray(action, dtype=np.float32)
        cmd_diff = action - self.prev_cmd

        k_cmd_rate = 0.2
        k_vz_cmd = 0.05

        r_cmd_rate = -k_cmd_rate * float(np.dot(cmd_diff, cmd_diff))
        r_vz_cmd = -k_vz_cmd * float(action[3] ** 2)


        ground_penalty = 0.0
        crashed = False
        truncated = False


        if z2 < (z0 + self.safe_ground_height) and abs(vz2) < 0.05:
            self.ground_steps += 1
            ground_penalty += -0.5
        else:
            self.ground_steps = 0

        if self.ground_steps >= self.ground_stall_max_steps:
            crashed = True
            ground_penalty += -100.0
            info["crash"] = True
            info["reason"] = "stalled_on_ground"


        r_ceiling = 0.0
        if z2 > self.soft_ceiling:
            dz_ceiling = z2 - self.soft_ceiling
            r_ceiling += -15.0 * dz_ceiling * dz_ceiling
            if z2 > self.hard_ceiling:
                crashed = True
                info["crash"] = True
                info["reason"] = "hit_hard_ceiling"

        # Lateral safety radius
        if r > self.safety_radius:
            crashed = True
            info["crash"] = True
            info["reason"] = "lateral_escape"


        if tilt > np.deg2rad(70.0) or not np.isfinite(state_curr).all():
            crashed = True
            info["crash"] = True
            info.setdefault("reason", "tilt_or_numerical")


        alive_air_reward = 0.0
        if not crashed and z2 > (z0 + self.safe_ground_height):
            alive_air_reward = 0.02

 
        in_hover_band = (
            abs(dz) < self.hover_band_half_width
            and r < self.lateral_soft_radius
            and tilt < np.deg2rad(15.0)
            and abs(vz2) < 0.15
        )

        r_hover = 0.0
        if in_hover_band and not crashed:
            self.hover_steps += 1
            r_hover += 0.5
        else:
            self.hover_steps = 0

        success = False
        if self.hover_steps >= self.hover_required_steps and not crashed:
            success = True
            r_hover += 10.0
            info["success"] = True
            info["hover_steps"] = self.hover_steps

        # Final reward sum
        reward = (
            r_z
            + r_progress
            + r_vz
            + r_xy_center
            + r_xy_soft
            + r_vxy
            + r_tilt
            + r_omega
            + r_thrust_smooth
            + r_moment_abs
            + r_moment_jump
            + r_cmd_rate
            + r_vz_cmd
            + ground_penalty
            + r_ceiling
            + alive_air_reward
            + r_hover
        )

        terminated = crashed or success
        return float(reward), terminated, truncated, info



    @staticmethod
    def _quat_to_rpy(q: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert quaternion [qw, qx, qy, qz] to roll, pitch, yaw (radians).
        """
        qw, qx, qy, qz = q

 
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

    
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = np.sign(sinp) * (np.pi / 2.0)
        else:
            pitch = np.arcsin(sinp)

    
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return float(roll), float(pitch), float(yaw)

    def _current_obs_or_blank(self) -> np.ndarray:
        """Return the current stacked observation, or a zero vector if not available."""
        if len(self._frame_buffer) == self.frame_stack:
            return np.concatenate(list(self._frame_buffer)).astype(np.float32)
        if len(self._frame_buffer) > 0:
            last = self._frame_buffer[-1]
            buf = [last for _ in range(self.frame_stack)]
            return np.concatenate(buf).astype(np.float32)
        return np.zeros(self.observation_space.shape, dtype=np.float32)



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
