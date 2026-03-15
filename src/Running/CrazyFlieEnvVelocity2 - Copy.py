# using os for file pathing and file loading
##using tuple,Dict,and deque as datastructures
##nump is also used to create arrays and perform math operations, we'll use these arrays to hold data like observations and actions
##gymnasium is used to implement the gym.Env interface and also gives us access to the Spaces data structures for actiona and observation space
### mujoco imported for our physics engine, it is what we are applying the environment and actions too
import os
from typing import Optional, Tuple, Dict, Any
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco as mj

##this class right here we define our custome RL environment by first inheriting from the gym itnerface gym.Env
## which demands the implementaiton of __init__, step(action), reset functions
class CrazyFlieEnvVelocity(gym.Env):
    """RL environment for the CrazyFlie 2.1 simulted drone in Mujjoco, designed for safe hover at a target height.
    """
   

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}##meta data

    def __init__(
        self,
        xml_path: str,
        target_z: float = 1.0,
        max_steps: int = 1000,
        n_stack: int = 4,##frame stack, we create a stack of observations and n_stack is the number of observations in the stack
        ## Smoothing params, alpha and slew, it's how we soften/limit the thrust at each step
        thrust_lowpass_alpha: float = 0.25,
        thrust_slew_per_step: float = 0.08,
        ## Hover success
        hover_band: float = 0.1, ## band +- hover height to indicate a valid hover
        hover_required_steps: int = 300,## number of steps in the simulator needed for a successful hover(300 steps=5 seconds)
        smooth_window: int = 300, ##the size of our sliding window for our histories data
        ##anti-overshoot guards defining a soft ceiling for the drone to not go over(small penalty) and a hard ceiling where the episode terminates and big penalty
        soft_ceiling_margin: float = 0.20,## soft_ceiling+target_height is the z-value where the drone gets penalized for wandering, 
        hard_ceiling_margin: float = 2.0,## hard_ceiling+target_height is the z-value where episode terminates and gets punished

        ##DOmain raNdomization Paramaters
        ##NOTE: bias indicates slightly incorrect values, scaling indicates issues with sensors
        obs_noise_std: float = 0.0,## standard deviation forper-step randomization of noise
        obs_bias_std: float = 0.0,##episodic constant bias(additional values) to observations
        action_noise_std: float = 0.0,
        motor_scale_std: float = 0.0,##per-episode thrust-scale to simulate any motor mismatch
        frame_skip: int = 10, # agent chooses an action and agent applies it for 10 physics steps
        ##the value to +- the frame skip value to change the number of physics steps the agent applies an aciton
        ##simulates communication delay in drone
        frame_skip_jitter: int = 0,
        ###Auto landing
        auto_landing: bool = False,
        ###Randomization pramaters
        start_xy_range: float = 0.0,
        start_z_min: float = 0.01,
        start_z_max: float = 0.01,
        random_start: bool = True,
        
        torque_bias_std: float = 0.0, ##constant offset value for rotational axis'
        torque_gust_std: float = 0.0,   ## constant time-varying white noise to simulate gusts for rotaitonal values
        torque_gust_tau: float = 1.5,##time constant to apply torque_gust to the drone before smoothing out
        drag_lin_min: float = 0.00,##simulation of drag for linear velocity(drag as in resistance to velocity)
        drag_lin_max: float = 0.06,
        drag_quad_min: float = 0.00,##quaternion drag(angular velocity dampening)
        drag_quad_max: float = 0.03,

        # Sim-side upright assist & attitude scaling
        enable_upright_assist: bool = True,##upright assist helps corrective tilt to prevent policy from slamming a tilt and flipping too often
        k_att: float = 0.5,##strength of the upright assist, right now half strong to allow policy some command
        ##default attitute scaling to smooth otu atitute to reduce aggressive commands
        att_base_scale: float = 0.90,
        att_min_scale: float = 0.25,
        ##near ground instantiation to apply atitutde scaling since closer to ground is more dangerous
        near_ground_z: float = 0.30,
        near_ground_scale: float = 0.45,##how much of command is given to near_ground tilt assist
        tilt_soft_deg: float = 6.0,##min and max where we apply attiute scaling, once surpassing hard degree we stop allowing larger commands to takeover
        tilt_hard_deg: float = 25.0,
        
    ):
        ## call the base class initialization(in this case the gym constructor we are inheriting from)
        super().__init__()

        if not os.path.exists(xml_path):## check if mujjoco file exists
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        self.on_ground_prev = True##initalize with on_ground condition
        # the model is the world absolute containing all of the model values and actuators
        self.model = mj.MjModel.from_xml_path(xml_path)
        ##the data is the current state of the drone, it's velocities etc.
        self.data = mj.MjData(self.model)
        #intialize env properties for the target height and max steps
        self.target_z = float(target_z)
        self.max_steps = int(max_steps)
        ##randomization of x,y,z variables for starting position
        self.random_start = bool(random_start)
        self.start_xy_range = float(start_xy_range)
        self.start_z_min = float(start_z_min)
        self.start_z_max = float(start_z_max)
        self.spawn_xy = np.zeros(2, dtype=np.float64)
        self.spawn_z = 0.0

        #FRAME STACKING
        self.n_stack = int(n_stack)
        self.obs_dim_single = 13##size of 1 single observation without frame stcking
        ##Our observation is 13D, position(x,y,z), quaternion pos(w,x,y,z), lin velocity (vx,vy,vz), angular velocity(wx,wy,wz)
        ##frame stacking of size n_stack means we are storing the last n_stack observations, so in total 4x13=52
        ##we constantly hold a 52D observation space containing all 4 past observations
        hi = np.inf * np.ones(self.obs_dim_single * self.n_stack, dtype=np.float32)
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)
        self.obs_stack = deque(maxlen=self.n_stack)##using a deque(using deque to hold frame_stack)

        #### Domain randomization initializaiton paramaters
        self.obs_noise_std = float(obs_noise_std)
        self.obs_bias_std = float(obs_bias_std)
        self.action_noise_std = float(action_noise_std)
        self.motor_scale_std = float(motor_scale_std)
        self.frame_skip_base = int(frame_skip)
        self.frame_skip_jitter = int(frame_skip_jitter)
        self.frame_skip = self.frame_skip_base# will be randomized in reset()
        ##Per-episode randomization values     
        self.obs_bias = np.zeros(self.obs_dim_single, dtype=np.float32)##13D vector containing offset values for observaiton bias
        self.motor_scale = 1.0##scaling for motor(thrust)
        self.pos_gain = np.ones(3, dtype=np.float32)
        self.vel_gain = np.ones(3, dtype=np.float32)
        ##random walk-drift accumulating through an episode, random walk adds or subtracts as a bias
        self.bias_drift = np.zeros(self.obs_dim_single, dtype=np.float32)

        ##Mujjoco thrust limits(in newtons)
        tmin, tmax = 0.0, 0.4
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        ##hover-thrust in mujjoco based on weight of drone multipled by gravity
        ##we use this as base hover for stable learning earlier
        self.HOVER_THRUST = float(np.clip(0.34335, tmin, tmax))

       #Smoothing
        ##alpha and max_du are properties we have for the slew rate and lowpass filter
        self.alpha = float(thrust_lowpass_alpha)
  
        self.max_du = float(thrust_slew_per_step)
        ##u_cmd is our filtered thrust which we will pass into the data control rather than the raw action
        self.u_cmd = self.HOVER_THRUST
        #last_du stands for last delta u(thrust), so last change of thrust of the last episode, we use this in the reward funciton to discourage jerky changes
        self.last_du = 0.0
        ##storing the last moment values(x,y,z) rotations, as array of 3 zeros(0,0,0) since there is no moments yet
        self.last_moments = np.zeros(3, dtype=np.float32)
        self.last_dm = 0.0##last change in moments

        
        ### tracking for successful hover
        self.band = float(hover_band)
        self.hover_required = int(hover_required_steps)
        self.hover_count = 0
        
        ##track thrust jerk and vertical speed across the smooth_window as a history to detect jerky changes in either
        self.du_hist = deque(maxlen=int(smooth_window))##tracks thrust change
        self.vz_hist = deque(maxlen=int(smooth_window))##tracks vertical velocity
        self.prev_dz = 0.0##prev change in z
        
        self.soft_ceiling_margin = float(soft_ceiling_margin)
        self.hard_ceiling_margin = float(hard_ceiling_margin)

        self.target_z_abs = self.target_z
        self.soft_ceiling = self.target_z_abs + self.soft_ceiling_margin
        self.hard_ceiling = self.target_z_abs + self.hard_ceiling_margin
        self.step_idx = 0 ##our timer and step counter for the whole episode

        #######Ground-stall detection Paramaters
        self.ground_z_threshold = 0.05
        self.max_ground_steps = 100
        self.ground_steps = 1

        ###SAFETY THRESHOLDS
        self.safety_max_tilt_rad = np.deg2rad(35.0)##max tilt allowed in radians
        self.safety_max_abs_vz = 5.0##max velocity
        self.safety_radius = 2.0 ##lateral bounds
        self.lateral_soft_radius = 0.6 * self.safety_radius##soft lateral bounds for penalizing reward(60%)

        ##Landing controller paramaters for hard-coded landing blended with policy
        self.phase = "HOVER"
        self.auto_landing = bool(auto_landing)
        self.landing_step_idx = 0
        self.landing_beta = 0.0##blending rate for the landing for using policy thrust value
        self.landing_mode = "DESCEND"
        self.landing_catch_steps = 0
        self.pre_landing_reason: Optional[str] = None
        self.landing_max_radius = 0.8
        self.landing_safe_radius = 0.4
        self.landing_tilt_abort_deg = 25.0
        self.landing_tilt_ok_deg = 10.0
        self.landing_beta_ramp_steps = 200## how many steps to ramp down our blending between policy and hard-coded thrust
        self.landing_max_steps = 800
        ##hard-coded thrust values for landing
        self.landing_vz_fast = -0.30
        self.landing_vz_med = -0.20
        self.landing_vz_mid = -0.15
        self.landing_vz_slow = -0.10
        self.landing_k_vz = 0.4##vertical speed gain for landing, how aggressive thrust responds to speed mismatch 


        ###COMMANDER ACTIONS and max values for pitch and vertical velocity
        self.max_roll_deg   = 3.0 
        self.max_pitch_deg  = 3.0
        self.max_yawrate_deg= 45.0
        self.max_vz_cmd = 0.6
        ##converting degrees to radians
        self.max_roll_rad = np.deg2rad(self.max_roll_deg)
        self.max_pitch_rad = np.deg2rad(self.max_pitch_deg)
        self.max_yawrate = np.deg2rad(self.max_yawrate_deg)
        
        ##These are our commands->Actuators paramaters for aciton controllers
        ##ATITUDE PD controller params
        self.att_kp = 6.0##attiute proportional gain on angle errors(how aggressive we correct tilt error)
        self.att_kd = 0.3##derivative gain on angular rate, how much we damp wobble
        self.yaw_kp = 1.0##proportion gain on yaw
        self.yaw_kd = 0.05##derivative gain on yaw
        self.vz_kp = 0.4##proportional gain for thrust offset

        self.action_space = spaces.Box(##normalized action space for vertical velocity and torques
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([+1.0, +1.0, +1.0, +1.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        ###randomization for torque and drag
        self.torque_bias_std = float(torque_bias_std)
        self.torque_gust_std = float(torque_gust_std)
        self.torque_gust_tau = float(max(1e-3, torque_gust_tau))
        self.drag_lin_min = float(drag_lin_min)
        self.drag_lin_max = float(drag_lin_max)
        self.drag_quad_min = float(drag_quad_min)
        self.drag_quad_max = float(drag_quad_max)
        self.torque_bias = np.zeros(3, dtype=np.float32)
        self.torque_ou = np.zeros(3, dtype=np.float32)
        self.drag_lin = np.zeros(3, dtype=np.float32)
        self.drag_quad = np.zeros(3, dtype=np.float32)
        ###upright assist and atitute params
        self.enable_upright_assist = bool(enable_upright_assist)
        self.k_att = float(k_att)
        self.att_base_scale = float(att_base_scale)
        self.att_min_scale = float(att_min_scale)
        self.near_ground_z = float(near_ground_z)
        self.near_ground_scale = float(near_ground_scale)
        self.tilt_soft_deg = float(tilt_soft_deg)
        self.tilt_hard_deg = float(tilt_hard_deg)
       
        self.prev_cmd = np.zeros(4, dtype=np.float32)##stores previous action/command for reward func

        ##reward weights
        # "Good enough" scales â€” used to normalize squared errors (dimensionless)
        self.z_scale = 1.0                # meters (hover band-ish)
        self.vz_scale = 0.20                # m/s (vertical stability)
        self.r_scale = 0.25                 # meters (lateral drift radius)
        self.vxy_scale = 0.25               # m/s (lateral speed)
        self.tilt_scale = np.deg2rad(10.0)  # radians (tilt threshold)
        self.omega_scale = np.deg2rad(200.0)  # rad/s-ish, just damping spikes

        # Weights for each normalized cost term
        self.w_z = 1.0
        self.w_vz = 0.5
        self.w_r = 0.5
        self.w_vxy = 0.5
        self.w_tilt = 1.0
        self.w_omega = 0.05
        self.w_smooth_u = 0.05
        self.w_smooth_m = 0.05

        # Smoothness normalization (typical â€œsmallâ€ changes)
        self.du_scale = 0.02   # "small" thrust change per env step
        self.dm_scale = 0.50   # "small" moment change norm (moments are clipped [-1, 1])

    
    def _sample_episode_randomization(self):
        """
        Samples randomization values for the DR values tht stay fixed for the whole episode
        """
        rng = getattr(self, "np_random", np.random)##random number generator

        ##OBSERVATION BIAS: generate 13D vector for each obs index
        ##each value is sampled from a normal distribution of mean 0 and std of obs_bias_std
        if self.obs_bias_std > 0.0:
            bias = rng.normal(0.0, self.obs_bias_std, size=self.obs_dim_single).astype(np.float32)##set obs_bias offset as an array for all values
            bias[3:7] = 0.0##zero out the rotational bias as it can break orientaiton math
            self.obs_bias = bias##vector of obs_bias paramaters for each one
        else:
            self.obs_bias[:] = 0.0

        ##OBSERVATION GAIN: scaling errors from sensors
        self.pos_gain[:] = 1.0
        self.vel_gain[:] = 1.0
        if (self.obs_bias_std > 0.0) or (self.obs_noise_std > 0.0):##if there is any bias or noise then 
            self.pos_gain[0:2] = 1.0 + rng.normal(0.0, 0.02, size=2).astype(np.float32)##2% gain on position
            self.pos_gain[2] = 1.0 + float(rng.normal(0.0, 0.03))##3% gain for z
            self.vel_gain[:] = 1.0 + rng.normal(0.0, 0.05, size=3).astype(np.float32)##5% gain for rest of obs

        ##THRUST GAIN: simulating scaling with thrust which can be stronger/weaker like in real
        ##real drone's thrust is PWM drive dependant on 16-bit thrust integer, battery voltage and propellor position 
        if self.motor_scale_std > 0.0:
            self.motor_scale = float(1.0 + rng.normal(0.0, self.motor_scale_std))
        else:
            self.motor_scale = 1.0

        ##FRAME-SKIP JITTER: offsets base frame skip rate
        if self.frame_skip_jitter > 0:
            jitter = int(rng.integers(-self.frame_skip_jitter, self.frame_skip_jitter + 1))
            self.frame_skip = max(1, self.frame_skip_base + jitter)
        else:
            self.frame_skip = self.frame_skip_base

        ##reset drifting bias
        self.bias_drift[:] = 0.0

        ###TORQUE BIAS: sample random torque bias for constant rotational disturbance to simulate imbalance
        if self.torque_bias_std > 0.0:
            self.torque_bias = rng.normal(0.0, self.torque_bias_std, size=3).astype(np.float32)
        else:
            self.torque_bias[:] = 0.0
        self.torque_ou[:] = 0.0##torque gust unimplemented

       ##DRAG: velocity dampening sampling
        if self.drag_lin_max > 0.0:##sample a value for linear velocity drag
            self.drag_lin = rng.uniform(self.drag_lin_min, self.drag_lin_max, size=3).astype(np.float32)
        else:
            self.drag_lin[:] = 0.0

        if self.drag_quad_max > 0.0:##sample value for quadratic drag for higher speeds
            self.drag_quad = rng.uniform(self.drag_quad_min, self.drag_quad_max, size=3).astype(np.float32)
        else:
            self.drag_quad[:] = 0.0


    def _apply_obs_noise(self, single: np.ndarray) -> np.ndarray:
        """Takes in a single 13D array and then applies noise to the observation
        """
        
        rng = getattr(self, "np_random", np.random)
        s = single.astype(np.float32).copy()##copy the state/obs and conver to np.float32 for consistency
        pos = s[0:3]
        quat = s[3:7]
        linv = s[7:10]
        angv = s[10:13]
        pos_meas = self.pos_gain * pos ##apply gain/scaling to pos and linv
        vel_meas = self.vel_gain * linv

        pos_meas += self.obs_bias[0:3]##add bias offset to pos and lin velocity
        vel_meas += self.obs_bias[7:10]
        angv_meas = angv + self.obs_bias[10:13]##add offset to angular vel
        ##Drift/Random WALK
        if self.obs_noise_std > 0.0:##sample a drift value then add to drift
            drift_step = rng.normal(0.0, self.obs_noise_std * 0.01, size=self.obs_dim_single).astype(np.float32)
            self.bias_drift += drift_step
        else:
            self.bias_drift[:] = 0.0
        ##add drift offset to pos, lin velocity and angular vel
        pos_meas += self.bias_drift[0:3]
        vel_meas += self.bias_drift[7:10]
        angv_meas += self.bias_drift[10:13]
        ##WHITE NOISE: each time step add a fresh random jitter to each obs value
        #different multipliers for different components to simualte sensor error differences
        if self.obs_noise_std > 0.0:
            pos_meas[0:2] += rng.normal(0.0, self.obs_noise_std * 0.5, size=2).astype(np.float32)
            pos_meas[2] += float(rng.normal(0.0, self.obs_noise_std * 1.0))
            vel_meas += rng.normal(0.0, self.obs_noise_std * 0.7, size=3).astype(np.float32)
            angv_meas += rng.normal(0.0, self.obs_noise_std * 0.4, size=3).astype(np.float32)

        quat_meas = quat.copy()
        norm_q = float(np.linalg.norm(quat_meas))##renormalize quaternion if we add noise
        if norm_q > 1e-6:
            quat_meas /= norm_q

        pos_rel = pos_meas.copy()
        ##make all positions relative to where drone started
        pos_rel[0] -= float(self.spawn_xy[0])
        pos_rel[1] -= float(self.spawn_xy[1])
        pos_rel[2] -= float(self.spawn_z)
        ##repack all of our noise-applied components
        noisy = np.concatenate([pos_rel, quat_meas, vel_meas, angv_meas]).astype(np.float32)
        
        ##very rare glitch injection to simulate sensor spikes
        if self.obs_noise_std > 0.0 and rng.random() < 1e-3:
            glitch = rng.normal(0.0, self.obs_noise_std * 10.0, size=3).astype(np.float32)
            noisy[0:3] += glitch

        return noisy


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)##reset mj simulation state

        ##mujocco data has 3 parts
        ##data.qpos = (x,y,z)+(w,x,y,z) position and quaternion pos
        ##data.qpvel = (vx,vy,vz)+(wx,wy,wz) = linear velocity and angular velocity
        ##reset each to base values so 0 for all of them
        ##quaternion position means drone is straight up with no rotation represented by the identity matrix (1,0,0,0)

        rng = getattr(self, "np_random", np.random)
        self.on_ground_prev = True
        ##if random start then randomize x,y,z
        if self.random_start and ((self.start_xy_range > 0.0) or (self.start_z_max > self.start_z_min)):
            x0 = float(rng.uniform(-self.start_xy_range, self.start_xy_range))
            y0 = float(rng.uniform(-self.start_xy_range, self.start_xy_range))
            z0 = float(rng.uniform(self.start_z_min, self.start_z_max))
        else:##default start
            x0, y0, z0 = 0.0, 0.0, 0.01

        z0 = float(np.clip(z0, 0.01, 10.0))##c
        ##recording spawn point so lateral bounding logic acts to this
        self.spawn_xy[0] = x0
        self.spawn_xy[1] = y0
        self.spawn_z = float(z0)
        
        ##set all target z, soft and hard ceilings
        self.target_z_abs = self.spawn_z + self.target_z
        self.soft_ceiling = self.target_z_abs + self.soft_ceiling_margin
        self.hard_ceiling = self.target_z_abs + self.hard_ceiling_margin
        ##change spawn if it is above the hard ceiling
        if self.spawn_z > self.hard_ceiling - 0.05:
            self.spawn_z = self.hard_ceiling - 0.05
            z0 = self.spawn_z
        ##set mj default starting state and zero out velocoties and control actions
        self.data.qpos[:] = np.array([x0, y0, z0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        self.u_cmd = self.HOVER_THRUST
        self.last_du = 0.0
        self.hover_count = 0
        self.du_hist.clear()
        self.vz_hist.clear()
        self.step_idx = 0
        self.ground_steps = 0
        self.prev_dz = 0.0
        self.prev_cmd[:] = 0.0
        ##reseting landing controller states
        self.phase = "HOVER"
        self.landing_step_idx = 0
        self.landing_beta = 0.0
        self.landing_mode = "DESCEND"
        self.landing_catch_steps = 0
        self.pre_landing_reason = None

        self._end_noise_free_landing()##re-enable noise after landing disables it from prev episodes
        self._sample_episode_randomization()##randomize noise paramaters

        self.obs_stack.clear()##reset frame stack
        single_clean = self._get_single_obs()##get our first obs and then apply noise and create frame stack
        single = self._apply_obs_noise(single_clean)
        for _ in range(self.n_stack):
            self.obs_stack.append(single.copy())

        obs = single if self.n_stack == 1 else np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)## either return a single 13D vector or a list of the n_stck*13 dimensioned vector
        return obs, {}

 
    def _apply_disturbances(self, dt: float):
        """
   Applying external forces and torques to simulate noise,
   takes in dt for time duration for one physics step
        """
        nv = int(self.model.nv)##double check for 6 DOF on our mj model
        if nv < 6:
            return
        
        ##reset because mj does not tend to reset these external forces
        self.data.qfrc_applied[:] = 0.0##data.qfrc_applied is how to apply externa mj forces
        rng = getattr(self, "np_random", np.random)

        ##Generates gust torque using Ornstein-uhlenbeck process: 
        ###https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        ##OU noise changes over time and it doesnt jump wildly each step and decays back towards 0 naturally
        ##simulates random gusts then fading.
        if self.torque_gust_std > 0.0:##
            decay_t = np.exp(-dt / self.torque_gust_tau)
            noise_t = rng.normal(0.0, self.torque_gust_std, size=3).astype(np.float32)
            self.torque_ou = decay_t * self.torque_ou + np.sqrt(max(1e-9, 1.0 - decay_t * decay_t)) * noise_t
        else:
            self.torque_ou[:] = 0.0
        ##add the gust torque differences to the bias offset
        torque = self.torque_bias + self.torque_ou

        ##apply linear drag  and quadratic drag
        v = self.data.qvel[0:3].astype(np.float32)
        drag = -self.drag_lin * v - self.drag_quad * np.abs(v) * v

        self.data.qfrc_applied[0:3] += drag.astype(np.float64)##add drag values and torque to external forces
        self.data.qfrc_applied[3:6] += torque.astype(np.float64)

    # ------------------------------------------------------------------
    # Actuator application
    # ------------------------------------------------------------------
    def _apply_thrust(self, u_scalar: float, m_vec: np.ndarray):
        ####this applies the filtering to the thrust, u_scalar is the thrust aciton
        ## in this funciton we perform acutator shaping or thrust smoothing and moments
        ## u_cmd:current motor thrust
        ## u_scalar: current requested thrust for the drone
        ##du :how much the policy wants to change thrust -- (action_thrust_value - base_thrust_command ) clipped between the maximum thrust change
        du = np.clip(u_scalar - self.u_cmd, -self.max_du, self.max_du)
          
        ##u_slewed: new thrust
        ##calcualte new thrust by adding previous thrust to the new requested thrust value's clipped change, this is slew-rate
        u_slewed = self.u_cmd + du
        ##new_u is the filtered thrust
        ##smooth the new thrust with the low-pass filter = (1-alpha)*current_thrust + alpha*desired_thrust
        new_u = (1.0 - self.alpha) * self.u_cmd + self.alpha * u_slewed##this helps blending of thrust

         ##the magnitude of the change in thrust for the current step
        self.last_du = float(abs(new_u - self.u_cmd))##measure how much the thrust has changed and store it in last_du
        self.u_cmd = float(new_u)

        ##apply filtered thrust to actuator 0
        self.data.ctrl[0] = self.u_cmd

        ##apply moments to actuators 1:4 (roll, pitch, yaw)
        ##NOTE: this must match your MuJoCo actuator ordering
        m_vec = np.asarray(m_vec, dtype=np.float32)
        m_clipped = np.clip(m_vec, -1.0, 1.0)

        ##safety: only write if model has those actuators
        if int(self.model.nu) >= 4:
            self.data.ctrl[1:4] = m_clipped.astype(np.float64)
            ##if model has extra actuators, keep them at 0 so nothing weird happens
            if int(self.model.nu) > 4:
                self.data.ctrl[4:] = 0.0
        else:
            ##if your model only has 1 actuator, we cannot apply moments
            ##(this will make attitude control impossible)
            pass

        ##track moment smoothness for reward shaping
        dm = m_clipped - self.last_moments##difference from previous moments
        self.last_dm = float(np.linalg.norm(dm, ord=2))##magnitude of the change
        self.last_moments = m_clipped.copy()##store last applied moments


 
    
    def _quat_to_euler(self, qw: float, qx: float, qy: float, qz: float):
        """Converys quaternion coardinats(w,x,y,z) to euler angle for convenience """
        ##standard from quaternion positions to euler

        ##first get roll
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)##pitch computation
        if abs(sinp) >= 1.0:
            pitch = np.sign(sinp) * (np.pi / 2.0)
        else:
            pitch = np.arcsin(sinp)
        ##yaw computation
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return float(roll), float(pitch), float(yaw)

    @staticmethod##static method
    def _quat_to_rp(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float]:
        """
        takes in quaternion and only returns roll and pitch
        """
        ##calculate roll from quat
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        ##calculate pitch from quat
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = float(np.clip(sinp, -1.0, 1.0))
        pitch = np.arcsin(sinp)
        return float(roll), float(pitch)
  
  ##Helper funcitons to Bridge RL Outputs and Simulation inputs
    def _decode_commander(self, a_norm: np.ndarray):
        """turn normalized actions into real units, here it is rdians and vz in m/s
        a_norm: 4D vector where each element is of type [-1,1]"""
        
        a = np.asarray(a_norm, dtype=np.float32).squeeze()
        if a.shape != (4,):
            raise ValueError("Policy action must be shape (4,)")
        
        ##convertto radians and desired vz
        roll_cmd = float(a[0]) * self.max_roll_rad
        pitch_cmd = float(a[1]) * self.max_pitch_rad
        yawrate_cmd = float(a[2]) * self.max_yawrate
        vz_cmd = float(a[3]) * self.max_vz_cmd

        return roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd

    def _attitude_pd(self, roll_cmd: float, pitch_cmd: float, yawrate_cmd: float, state: np.ndarray) -> np.ndarray:
        """takes command of actual units and converts it to mujjoco units,
        roll_cmd: desired roll,
        pitch_cmd: desired pitch,
        yawrate_cmd: desired yaw rate,
        state: full 13D single state"""
        
        qw, qx, qy, qz = state[3:7]
        wx, wy, wz = state[10:13]

        roll, pitch, _ = self._quat_to_euler(qw, qx, qy, qz)##extract quaternion from state
        
        ##use proportional derivative to convert to mitigate error
        ##proportion corrects angle error
        ##derivate damps and prevents oscilation in rotation
        tau_roll = self.att_kp * (roll_cmd - roll) - self.att_kd * float(wx)
        tau_pitch = self.att_kp * (pitch_cmd - pitch) - self.att_kd * float(wy)
        tau_yaw = self.yaw_kp * (yawrate_cmd - float(wz)) - self.yaw_kd * float(wz)

        m_vec = np.array([tau_roll, tau_pitch, tau_yaw], dtype=np.float32)
        return np.clip(m_vec, -1.0, 1.0)

    def _vertical_pd(self, vz_cmd: float, state: np.ndarray) -> float:
        """ converts desired vertical speed into thrust"""
        
        vz = float(state[9])
        err_vz = vz - vz_cmd ##vertical velocity error
        u = self.HOVER_THRUST - self.vz_kp * err_vz
        return float(np.clip(u, self.tmin, self.tmax))
 
 ##UPRIGHT ASSIST
    def _apply_attitude_assist_and_scaling(
        self,
        roll_cmd: float,
        pitch_cmd: float,
        yawrate_cmd: float,
        state: np.ndarray,
        tilt_deg: float,
        r_rad: float,
        z: float,
    ) -> Tuple[float, float, float, float]:
        if self.enable_upright_assist:
            qw, qx, qy, qz = state[3:7]
            roll_est, pitch_est = self._quat_to_rp(qw, qx, qy, qz)
            roll_cmd -= self.k_att * roll_est
            pitch_cmd -= self.k_att * pitch_est

        att_scale = self.att_base_scale

        if z < self.spawn_z + self.near_ground_z:
            att_scale = min(att_scale, self.near_ground_scale)

        if tilt_deg > self.tilt_soft_deg:
            over = min(tilt_deg, self.tilt_hard_deg) - self.tilt_soft_deg
            span = max(self.tilt_hard_deg - self.tilt_soft_deg, 1e-6)
            frac = over / span
            att_scale = min(att_scale, max(self.att_min_scale, 1.0 - 0.7 * frac))

        if r_rad > self.lateral_soft_radius:
            over_r = min(r_rad, self.safety_radius) - self.lateral_soft_radius
            span_r = max(self.safety_radius - self.lateral_soft_radius, 1e-6)
            frac_r = over_r / span_r
            att_scale = min(att_scale, max(self.att_min_scale, 1.0 - 0.7 * frac_r))

        roll_cmd *= att_scale
        pitch_cmd *= att_scale
        yawrate_cmd *= np.sqrt(max(1e-6, att_scale))

        return roll_cmd, pitch_cmd, yawrate_cmd, float(att_scale)

    def _commander_action_to_thrust_moments(
        self,
        a_norm: np.ndarray,
        state: np.ndarray,
        tilt_deg: float,
        r_rad: float,
        z: float,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """takes in the aciton from the RL agent that is normalized,
        takes in the current state,
        takes current tilt angle,
        takes in radius distance
        current z
        and outputs the thrust command in mujjoco newtons,
        normalized moment vector for pitcj,roll,yaw,
        debug info """
        
        roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd = self._decode_commander(a_norm)

        roll_cmd, pitch_cmd, yawrate_cmd, att_scale = self._apply_attitude_assist_and_scaling(
            roll_cmd, pitch_cmd, yawrate_cmd, state, tilt_deg, r_rad, z
        )

        m_vec = self._attitude_pd(roll_cmd, pitch_cmd, yawrate_cmd, state)
        u_req = self._vertical_pd(vz_cmd, state)

        debug = {
            "vz_cmd": float(vz_cmd),
            "att_scale": float(att_scale),
            "roll_cmd": float(roll_cmd),
            "pitch_cmd": float(pitch_cmd),
        }
        return u_req, m_vec, debug

        # ------------------------------------------------------------------
        # step()
        # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """
        Takes in an action, converts the normalized action space into physical actuator signals,
        advances MuJoCo for frame_skip physics steps, builds the new observation (with noise + frame stack),
        computes a SCALED reward (so episode returns don't blow up), and applies termination checks.
        """

        # --- Auto-landing override (if active) ---
        a = np.asarray(action, dtype=np.float32).squeeze()
        if self.auto_landing and (self.phase == "LANDING"):
            return self._step_landing(action)

        # --- Validate action shape ---
        if a.shape != (4,):
            raise ValueError("Action must be shape (4,): [roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd]")

        # --- Optional action noise (domain randomization) ---
        if self.action_noise_std > 0.0:
            rng = getattr(self, "np_random", np.random)
            a = a + rng.normal(0.0, self.action_noise_std, size=a.shape).astype(np.float32)

        # --- Clip action to [-1, 1] bounds ---
        a_clipped = np.clip(a, self.action_space.low, self.action_space.high)

        # --- State before applying action (clean) ---
        state_now = self._get_single_obs()
        x, y, z = float(state_now[0]), float(state_now[1]), float(state_now[2])
        qw, qx, qy, qz = state_now[3:7]
        vx, vy, vz = float(state_now[7]), float(state_now[8]), float(state_now[9])

        # --- Lateral radius from spawn ---
        x_rel = x - float(self.spawn_xy[0])
        y_rel = y - float(self.spawn_xy[1])
        r_rad = float(np.sqrt(x_rel * x_rel + y_rel * y_rel))

        # --- Tilt estimate from quaternion (approx) ---
        tilt_sin = float(np.clip(np.sqrt(qx * qx + qy * qy), 0.0, 1.0))
        tilt_angle = float(2.0 * np.arcsin(tilt_sin))
        tilt_deg = float(np.rad2deg(tilt_angle))

        # --- Convert normalized commander action -> thrust (N) and moments via your mapping ---
        u_req, m_req, dbg = self._commander_action_to_thrust_moments(
            a_clipped, state_now, tilt_deg=tilt_deg, r_rad=r_rad, z=z
        )

        # --- Apply per-episode motor scaling DR and clip to MuJoCo thrust bounds ---
        u_req = float(np.clip(u_req * self.motor_scale, self.tmin, self.tmax))

        # --- Apply thrust + moments (includes smoothing inside _apply_thrust) ---
        self._apply_thrust(u_req, m_req)

        # --- Step MuJoCo forward frame_skip times ---
        dt = float(self.model.opt.timestep)
        for _ in range(self.frame_skip):
            self._apply_disturbances(dt)
            mj.mj_step(self.model, self.data)

        # --- Advance env step counter ---
        self.step_idx += 1

        # --- Build next observation (clean -> noisy -> frame stack) ---
        single_clean = self._get_single_obs()
        single_noisy = self._apply_obs_noise(single_clean)

        if self.n_stack == 1:
            obs = single_noisy
        else:
            if len(self.obs_stack) == 0:
                for _ in range(self.n_stack):
                    self.obs_stack.append(single_noisy.copy())
            else:
                self.obs_stack.append(single_noisy.copy())
            obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)

        # --- Unpack next clean state for reward/termination ---
        x2, y2, z2 = float(single_clean[0]), float(single_clean[1]), float(single_clean[2])
        qw2, qx2, qy2, qz2 = single_clean[3:7]
        vx2, vy2, vz2 = float(single_clean[7]), float(single_clean[8]), float(single_clean[9])
        wx2, wy2, wz2 = float(single_clean[10]), float(single_clean[11]), float(single_clean[12])

        # --- Lateral drift radius from spawn ---
        x_rel2 = x2 - float(self.spawn_xy[0])
        y_rel2 = y2 - float(self.spawn_xy[1])
        r_rad2 = float(np.sqrt(x_rel2 * x_rel2 + y_rel2 * y_rel2))

        # --- Tilt from quaternion (same approx) ---
        tilt_sin2 = float(np.clip(np.sqrt(qx2 * qx2 + qy2 * qy2), 0.0, 1.0))
        tilt_angle2 = float(2.0 * np.arcsin(tilt_sin2))
        tilt_deg2 = float(np.rad2deg(tilt_angle2))

        # --- Track smoothness histories (optional debugging) ---
        self.du_hist.append(self.last_du)
        self.vz_hist.append(abs(vz2))

        # ============================================================
        # Ground detection + ground stall tracking
        # ============================================================
        on_ground = (z2 < (self.spawn_z + self.ground_z_threshold)) and (abs(vz2) < 0.05)
        if on_ground:
            self.ground_steps += 1
        else:
            self.ground_steps = 0

        # ============================================================
        # NORMALIZED COST -> DENSE REWARD (then SCALE by max_steps)
        # ============================================================
        # Core errors
        dz = z2 - self.target_z_abs                 # height error (m)
        r = r_rad2                                  # lateral drift radius (m)
        vxy2 = vx2 * vx2 + vy2 * vy2                # lateral speed squared
        tilt = tilt_angle2                          # tilt magnitude (rad)
        omega2 = wx2 * wx2 + wy2 * wy2 + wz2 * wz2  # angular rate squared

        # Normalized squared errors (dimensionless)
        ez2 = (dz / max(self.z_scale, 1e-6)) ** 2
        evz2 = (vz2 / max(self.vz_scale, 1e-6)) ** 2
        er2 = (r / max(self.r_scale, 1e-6)) ** 2
        evxy2 = vxy2 / max(self.vxy_scale * self.vxy_scale, 1e-6)
        etilt2 = (tilt / max(self.tilt_scale, 1e-6)) ** 2
        eomega2 = omega2 / max(self.omega_scale * self.omega_scale, 1e-6)

        # Smoothness penalties (important for sim2real)
        eu2 = (self.last_du / max(self.du_scale, 1e-6)) ** 2
        em2 = (self.last_dm / max(self.dm_scale, 1e-6)) ** 2

        # Total cost
        cost = (
            self.w_z * ez2
            + self.w_vz * evz2
            + self.w_r * er2
            + self.w_vxy * evxy2
            + self.w_tilt * etilt2
            + self.w_omega * eomega2
            + self.w_smooth_u * eu2
            + self.w_smooth_m * em2
        )

        # Dense step reward (clipped for PPO stability)
        dense = 1.0 - float(cost)
        dense = float(np.clip(dense, -2.0, 2.0))

        # IMPORTANT: scale so episode returns don't become huge (SB3 sums step rewards)
        reward = dense * (1.0 / max(1, self.max_steps))

        # Small "wasting time on the ground" penalty, also scaled
        if on_ground:
            reward -= (0.1 / max(1, self.max_steps))

        # ============================================================
        # STABILITY TRACKING + SUCCESS BONUS
        # ============================================================
        stable = (
            abs(dz) <= self.band
            and abs(vz2) < 0.05
            and tilt < np.deg2rad(10.0)
            and r < 0.25
            and np.sqrt(vxy2) < 0.10
            and not on_ground
        )

        if stable:
            self.hover_count += 1
            # small dense stability bonus (scaled)
            reward += (0.2 / max(1, self.max_steps))
        else:
            self.hover_count = 0

        # Success termination (episode-level terminal bonus)
        if self.hover_count >= self.hover_required:
            reward += 1.0  # NOT scaled (episode-level)

            info = {
                "success": True,
                "hover_steps": int(self.hover_count),
                "tilt_deg": float(tilt_deg2),
                "radius": float(r_rad2),
                "vz": float(vz2),
                "att_scale": float(dbg.get("att_scale", 1.0)),
            }

            if self.auto_landing:
                self._start_landing_phase("success")
                info["phase"] = "landing_start"
                # not terminated yet, entering landing
                return obs, float(np.clip(reward, -2.0, 2.0)), False, False, info

            return obs, float(np.clip(reward, -2.0, 2.0)), True, False, info

        # ============================================================
        # TERMINATIONS / TRUNCATIONS (use episode-level penalties)
        # ============================================================

        # Stalled on ground (episode-level penalty)
        if self.ground_steps >= self.max_ground_steps:
            info = {
                "crash": True,
                "reason": "stalled_on_ground",
                "ground_steps": int(self.ground_steps),
            }
            return obs, -1.0, True, False, info

        # NaN / inf / below ground (episode-level penalty)
        if z2 < 0.01 or np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            return obs, -1.0, True, False, {"crash": True, "reason": "nan_or_below_ground"}

        # Flipped (episode-level penalty)
        if tilt_angle2 > np.deg2rad(70.0):
            return obs, -1.0, True, False, {"crash": True, "reason": "flipped"}

        # Hard ceiling (episode-level penalty)
        if z2 > self.hard_ceiling:
            return obs, -1.0, True, False, {"ceiling": True, "reason": "hard_ceiling"}

        # Out of bounds (episode-level penalty)
        if r_rad2 > self.safety_radius:
            return obs, -1.0, True, False, {"crash": True, "reason": "out_of_bounds"}

        # Timeout truncation (keep whatever scaled reward you had)
        timeout = self.step_idx >= self.max_steps
        if timeout:
            info = {"hover_steps": int(self.hover_count), "timeout": True}
            if self.auto_landing:
                self._start_landing_phase("timeout")
                info["phase"] = "landing_start"
                return obs, float(np.clip(reward, -2.0, 2.0)), False, False, info
            return obs, float(np.clip(reward, -2.0, 2.0)), False, True, info

        # ============================================================
        # Normal step return
        # ============================================================
        info = {
            "hover_steps": int(self.hover_count),
            "tilt_deg": float(tilt_deg2),
            "radius": float(r_rad2),
            "vz": float(vz2),
            "att_scale": float(dbg.get("att_scale", 1.0)),
        }

        return obs, float(np.clip(reward, -2.0, 2.0)), False, False, info


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_single_obs(self) -> np.ndarray:
        pos = self.data.qpos[0:3]
        quat = self.data.qpos[3:7]
        linv = self.data.qvel[0:3]
        angv = self.data.qvel[3:6]
        return np.concatenate([pos, quat, linv, angv]).astype(np.float32)

    def _obs(self) -> np.ndarray:
        single_clean = self._get_single_obs()
        single = self._apply_obs_noise(single_clean)
        if self.n_stack == 1:
            return single
        if len(self.obs_stack) == 0:
            for _ in range(self.n_stack):
                self.obs_stack.append(single.copy())
        else:
            self.obs_stack.append(single.copy())
        return np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)

    def get_altitude(self) -> float:
        return float(self.data.qpos[2])

    def safe_ground_height(self) -> float:
        return 0.03

    def cut_motors(self) -> None:
        self.u_cmd = 0.0
        self.data.ctrl[:] = 0.0

    def apply_dr_params(self, params: dict) -> None:
        """
        Update domain-randomisation parameters at runtime.
        Changes take effect from the next episode (reset) because DR values
        are sampled inside _sample_episode_randomization().

        Called by the noise-curriculum callback in TrainVelocity2.py so that
        noise can be ramped up progressively without restarting environments.

        Example:
            env.apply_dr_params({"torque_bias_std": 3e-6, "torque_gust_std": 5e-7})
        """
        for key, val in params.items():
            if hasattr(self, key):
                current = getattr(self, key)
                try:
                    setattr(self, key, type(current)(val))
                except (TypeError, ValueError):
                    setattr(self, key, val)

    def _tilt_and_radius(self):
        x, y, _ = self.data.qpos[0:3]
        _, qx, qy, _ = self.data.qpos[3:7]

        tilt_sin = np.sqrt(qx * qx + qy * qy)
        tilt_sin = np.clip(tilt_sin, 0.0, 1.0)
        tilt_angle = 2.0 * np.arcsin(tilt_sin)

        x_rel = x - float(self.spawn_xy[0])
        y_rel = y - float(self.spawn_xy[1])
        r = float(np.sqrt(x_rel * x_rel + y_rel * y_rel))
        return float(tilt_angle), r

 
    ## Auto-landing 
    
    def _start_landing_phase(self, reason: str):
        self.phase = "LANDING"
        self.landing_step_idx = 0
        self.landing_beta = 0.0
        self.landing_mode = "DESCEND"
        self.landing_catch_steps = 0
        self.pre_landing_reason = reason
        self._begin_noise_free_landing()

    def _begin_noise_free_landing(self):
        self._landing_noise_backup = {
            "action_noise_std": self.action_noise_std,
            "motor_scale": self.motor_scale,
        }
        self.action_noise_std = 0.0
        self.motor_scale = 1.0

    def _end_noise_free_landing(self):
        if hasattr(self, "_landing_noise_backup"):
            self.action_noise_std = self._landing_noise_backup["action_noise_std"]
            self.motor_scale = self._landing_noise_backup["motor_scale"]
            del self._landing_noise_backup

    def _step_landing(self, action: np.ndarray):
        a_pol = np.asarray(action, dtype=np.float32).squeeze()
        if a_pol.shape != (4,):
            raise ValueError("Landing expects 4D commander action [roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd].")

        a_pol = np.clip(a_pol, self.action_space.low, self.action_space.high)

        single_clean = self._get_single_obs()
        x, y, z = single_clean[0:3]
        vx, vy, vz = single_clean[7:10]

        roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd_pol = self._decode_commander(a_pol)
        m_pol = self._attitude_pd(roll_cmd, pitch_cmd, yawrate_cmd, single_clean)

        err_vz_pol = vz - vz_cmd_pol
        u_pol = self.HOVER_THRUST - self.vz_kp * err_vz_pol
        u_pol = float(np.clip(u_pol, self.tmin, self.tmax))

        tilt_angle, r = self._tilt_and_radius()
        tilt_deg = float(np.rad2deg(tilt_angle))

        if self.landing_step_idx == 0:
            self.landing_beta = 0.0
            self.landing_mode = "DESCEND"
            self.landing_catch_steps = 0

        mode = self.landing_mode
        if (tilt_deg > self.landing_tilt_abort_deg) or (r > self.landing_max_radius):
            mode = "CATCH"
            self.landing_mode = "CATCH"
            self.landing_catch_steps = 0

        if mode == "CATCH":
            v_des = 0.0
            self.landing_beta = max(0.0, self.landing_beta - 0.05)

            if (tilt_deg < self.landing_tilt_ok_deg and r < self.landing_safe_radius and
                    abs(vx) < 0.2 and abs(vy) < 0.2):
                self.landing_catch_steps += 1
            else:
                self.landing_catch_steps = 0

            if self.landing_catch_steps > 50:
                self.landing_mode = "DESCEND"
        else:
            h = z - self.safe_ground_height()
            h = max(0.0, float(h))
            if h > 0.8:
                v_des = self.landing_vz_fast
            elif h > 0.4:
                v_des = self.landing_vz_med
            elif h > 0.2:
                v_des = self.landing_vz_mid
            else:
                v_des = self.landing_vz_slow

            step_idx = self.landing_step_idx
            self.landing_beta = min(1.0, step_idx / max(1, self.landing_beta_ramp_steps))

        beta = float(self.landing_beta)

        err_v = vz - v_des
        u_land = self.HOVER_THRUST - self.landing_k_vz * err_v
        u_min = 0.12
        u_max = float(self.tmax)
        u_land = float(np.clip(u_land, u_min, u_max))

        u = (1.0 - beta) * u_pol + beta * u_land
        u = float(np.clip(u, u_min, u_max))

        self._apply_thrust(u, m_pol)

        dt = float(self.model.opt.timestep)
        for _ in range(self.frame_skip):
            self._apply_disturbances(dt)##apply each physics step (consistent with hover)
            mj.mj_step(self.model, self.data)

        self.step_idx += 1
        self.landing_step_idx += 1

        single_clean_next = self._get_single_obs()
        single_next = self._apply_obs_noise(single_clean_next)

        if self.n_stack == 1:
            obs = single_next
        else:
            if len(self.obs_stack) == 0:
                for _ in range(self.n_stack):
                    self.obs_stack.append(single_next.copy())
            else:
                self.obs_stack.append(single_next.copy())
            obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)

        x2, y2, z2 = single_clean_next[0:3]
        vx2, vy2, vz2 = single_clean_next[7:10]
        tilt_angle2, r2 = self._tilt_and_radius()
        tilt_deg2 = float(np.rad2deg(tilt_angle2))

        low_enough = (z2 <= self.safe_ground_height())
        slow_enough = (abs(vz2) < 0.10)
        upright_enough = (tilt_deg2 < 10.0)
        inside_radius = (r2 < self.landing_safe_radius)

        landed = low_enough and slow_enough and upright_enough and inside_radius
        timeout = (self.landing_step_idx >= self.landing_max_steps)

        reward = 0.0
        terminated = False
        truncated = False

        info = {
            "phase": "landing",
            "landing_mode": self.landing_mode,
            "landing_beta": beta,
            "landing_landed": landed,
            "landing_timeout": timeout,
            "tilt_deg": tilt_deg2,
            "radius": r2,
            "vz": float(vz2),
            "pre_landing_reason": self.pre_landing_reason,
        }

        if landed or timeout:
            self._end_noise_free_landing()
            self.phase = "HOVER"
            terminated = True

        return obs, reward, terminated, truncated, info