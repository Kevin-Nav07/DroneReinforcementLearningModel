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
        tmin, tmax = 0.0, 0.35
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        ##hover-thrust in mujjoco based on weight of drone multipled by gravity
        ##we use this as base hover for stable learning earlier
        self.HOVER_THRUST = float(np.clip(0.32373, tmin, tmax))

       #Smoothing
        ##alpha and max_du are properties we have for the slew rate and lowpass filter
        self.alpha = float(thrust_lowpass_alpha)
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
        self.ground_steps = 0.1

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

        self.data.ctrl[0] = self.u_cmd
        # use the full physical actuator range from MuJoCo
        m_vec = np.asarray(m_vec, dtype=np.float32)
        m_clipped = np.clip(m_vec, -1.0, 1.0)
        ##delete this safety check commented out?
        # if self.model.nu >= 4:
        #     self.data.ctrl[1:4] = m_clipped
        #     if self.model.nu > 4:
        #         self.data.ctrl[4:] = 0.0

        dm = m_clipped - self.last_moments ###difference from previous moments
        self.last_dm = float(np.linalg.norm(dm, ord=2)) # magnitude of the change for smoothness penalty
        self.last_moments = m_clipped.copy()  # store last applied moments for next step

 
    
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
        """takes in an action, converts the normalized aciton space into physical actuator signals.
        advance mj steps with the action
        then build the new observation and add sensor noise and append to frame stack
        compute reward and termination checks
        """
        

        a = np.asarray(action, dtype=np.float32).squeeze()

        if self.auto_landing and (self.phase == "LANDING"):##auto landing feautre
            return self._step_landing(action)

        if a.shape == ():##check action space dimensionality
            raise ValueError("Action must be 4D: [roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd]")

        if self.action_noise_std > 0.0:
            rng = getattr(self, "np_random", np.random)
            a = a + rng.normal(0.0, self.action_noise_std, size=a.shape).astype(np.float32)##add noise then clip

        a_clipped = np.clip(a, self.action_space.low, self.action_space.high)

        state_now = self._get_single_obs()##retrieve the current clean_state before applying the actions
        x, y, z = float(state_now[0]), float(state_now[1]), float(state_now[2])
        qw, qx, qy, qz = state_now[3:7]
        vx, vy, vz = state_now[7:10]

        x_rel = x - float(self.spawn_xy[0])
        y_rel = y - float(self.spawn_xy[1])
        r_rad = float(np.sqrt(x_rel * x_rel + y_rel * y_rel))##compute lateralr radius
            
        tilt_sin = float(np.clip(np.sqrt(qx * qx + qy * qy), 0.0, 1.0))##
        tilt_angle = float(2.0 * np.arcsin(tilt_sin))
        tilt_deg = float(np.rad2deg(tilt_angle))##current tilt
        ##comute mujjoco requested thrust, requested moments, and get dbg(debug info)
        ##u_req and m_req are direct mujjoco commands in the mujjoco range
        u_req, m_req, dbg = self._commander_action_to_thrust_moments(
            a_clipped, state_now, tilt_deg=tilt_deg, r_rad=r_rad, z=z
        )
        ##apply motor thrust scaling and then send it to apply_thrust which 
        ##slews it then sends it into a lowpass filter and applies it on mjCtrl
        u_req = float(np.clip(u_req * self.motor_scale, self.tmin, self.tmax))
        self._apply_thrust(u_req, m_req)
        
        dt = float(self.model.opt.timestep)
        for _ in range(self.frame_skip):##advances fame_skip times physics steps
            self._apply_disturbances(dt)##applies disturbances on the timestep
            mj.mj_step(self.model, self.data)##advances physics step
        self.step_idx += 1##increment the overall episode counter
        
        ##retrieve our new observation and apply noise to it
        single_clean = self._get_single_obs()
        single = self._apply_obs_noise(single_clean)
        

        ##add to frame stack or create new frame stack item
        if self.n_stack == 1:
            obs = single
        else:
            if len(self.obs_stack) == 0:
                for _ in range(self.n_stack):
                    self.obs_stack.append(single.copy())
            else:
                self.obs_stack.append(single.copy())
            obs = np.concatenate(list(self.obs_stack), axis=0).astype(np.float32)
        ##unpack state and recompute lateral radius and tilts
        x2, y2, z2 = float(single_clean[0]), float(single_clean[1]), float(single_clean[2])
        qw2, qx2, qy2, qz2 = single_clean[3:7]
        vx2, vy2, vz2 = single_clean[7:10]
        wx2, wy2, wz2 = single_clean[10:13]

        x_rel2 = x2 - float(self.spawn_xy[0])
        y_rel2 = y2 - float(self.spawn_xy[1])
        r_rad2 = float(np.sqrt(x_rel2 * x_rel2 + y_rel2 * y_rel2))

        tilt_sin2 = float(np.clip(np.sqrt(qx2 * qx2 + qy2 * qy2), 0.0, 1.0))
        tilt_angle2 = float(2.0 * np.arcsin(tilt_sin2))
        tilt_deg2 = float(np.rad2deg(tilt_angle2))
        ##add to histories
        self.du_hist.append(self.last_du)
        self.vz_hist.append(abs(vz2))

        # REWARD SHAPING
        ##ground stall penalties: every step on the ground penalizes
        ground_penalty = 0.0##if z pos is less than a certain val and z vel is less then  add a penalty
        if (z2 < self.spawn_z + 0.015) and (abs(vz2) < 0.05):
            self.ground_steps += 1
            ground_penalty -= 3
        else:
            self.ground_steps = 0
        
        ##HEIGHT ERROR REWARD: reward based on negative quadratic formula
        ##dz = how far the current z is from the target height(error margin)
        dz = z2 - self.target_z_abs
        h_scale = max(self.target_z, 1e-3)
        dz_rel = dz / h_scale##normalize the eror
        k_z = 2.0##quadratic coeffecient
        r_z = -k_z * (dz_rel ** 2)##farther away more negtive, closer, more positive
        
        ##PROGRESS SHAPING: encourages getting closer by measuring height_error difference from last step
        ##so if you are moving in the right direction , then good
        prev_err_rel = abs(self.prev_dz) / h_scale
        curr_err_rel = abs(dz) / h_scale
        k_prog = 3.0##coeffecient
        r_progress = k_prog * np.clip(prev_err_rel - curr_err_rel, -0.2, 0.2)
        self.prev_dz = dz

        ##Near target factor, this makes penalties stronger closer to hover, value between [0,1]
        ##We want more stability around hover so the weights of rewards get stronger
        near = float(np.exp(- (dz_rel / 0.12) ** 2))
        
        ##VERTICAL SPEED PENALTY: penalize vertical velocity closer to target
        k_vz_far = 0.15
        k_vz_near = 2.5
        r_vz = -(k_vz_far + k_vz_near * near) * (vz2 ** 2)
        ###MOVING AWAY PENALTY:punishes going the wrong way when near the target
        ##checks if your current vertical velocity makes the height herror worse
        moving_away = 1.0 if (dz * vz2) > 0.0 else 0.0
        k_away = 6.0
        r_away = -k_away * moving_away * near * (abs(dz_rel) * (vz2 ** 2))
        
        ##LATERAL POSITION PENALTY: penalizes sideways drift from spawn centre
        k_xy_far = 1
        k_xy_near = 3
        r_xy_base = -(k_xy_far + k_xy_near * near) * (x_rel2 * x_rel2 + y_rel2 * y_rel2)
        
        ##SOFT LATERAL BOUNDARY PENALTY: extra punishment when drone exceeds the soft radius and begins quadratic negtive function#
        ##strongly discourages going towards the safety bounds
        r_xy_soft = 0.0
        if r_rad2 > self.lateral_soft_radius:
            over = r_rad2 - self.lateral_soft_radius
            k_xy = 4.0
            r_xy_soft = -k_xy * (over ** 2)
        
        ##LATERAL VELOCITY PENALTY: penalize sideways speed and more specifically closer to hover height. 
        ##negative quadrtic function where coefficient increases in magnitute closer to hover height
        k_vxy_far = 0.25
        k_vxy_near = 1.00
        r_vxy = -(k_vxy_far + k_vxy_near * near) * (vx2 ** 2 + vy2 ** 2)
        ######TILT PENALTY: penalize tilt and encourage uprightness
        ##negative quadrtic function that punishes big tilts a lot and tilting closer to target is heavily discouraged
        k_tilt_far = 4
        k_tilt_near = 8
        r_tilt = -(k_tilt_far + k_tilt_near * near) * (tilt_angle2 ** 2)
        
        ##ANGULAR VELOCITY PENALTY
        ##this, on top of tilt penalty, will encourge uprightness further as higher angular rates make it unstable
        k_omega_rp = 0.1
        k_omega_y = 0.05
        r_omega = -k_omega_rp * (wx2 ** 2 + wy2 ** 2) - k_omega_y * (wz2 ** 2)
        ##UPRIGHT BONUS
        ##this extra bonus attempts to encourage this exact behaviour
        upright_scale = 0.5
        upright_width = np.deg2rad(8.0)
        upright_bonus = upright_scale * np.exp(- (tilt_angle2 / upright_width) ** 2)
        ##SMOOTHNESS PENALTY for Moments+ thrust,
        ##prevents sudden jerky jumps
        m_mag2 = float(np.dot(self.last_moments, self.last_moments))
        k_moment_abs = 0.02
        k_moment_jump = 0.05
        k_thrust_jump = 0.2
        ##penalize big moment magnittudes, jumps and thrust jumps
        r_moment_abs = -k_moment_abs * m_mag2
        r_moment_jump = -k_moment_jump * self.last_dm
        r_thrust_smooth = -k_thrust_jump * self.last_du
        
        ##COMMAND RATE PENALTY:
        ##discourages twichy policies(remove maybe)
        roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd = self._decode_commander(a_clipped)
        cmd_now = np.array([roll_cmd, pitch_cmd, yawrate_cmd, vz_cmd], dtype=np.float32)
        dcmd = cmd_now - self.prev_cmd
        self.prev_cmd = cmd_now
        k_cmd_rate = 0.10
        r_cmd_rate = -k_cmd_rate * float(np.dot(dcmd, dcmd))
        ##PENALIZE LARGE VERTICAL VELOCITIES: discourages large velocity requests closer to target
        k_vz_cmd_far = 0.03
        k_vz_cmd_near = 0.25
        r_vz_cmd = -(k_vz_cmd_far + k_vz_cmd_near * near) * float(vz_cmd * vz_cmd)

        # r_takeoff = 0.0
        # if z2 < self.spawn_z + 0.02:
        #     r_takeoff -= 0.05
        # elif z2 < self.target_z_abs - 0.10:
        #     frac = np.clip((z2 - self.spawn_z) / h_scale, 0.0, 1.0)
        #     r_takeoff += 0.1 * frac
        on_ground = z2 < 0.03  # or use contact flag if you have one

        ##TAKEOFF BONUS
        just_took_off = self.on_ground_prev and (not on_ground)
        self.on_ground_prev = on_ground
        r_takeoff = 0
        if just_took_off:
            r_takeoff += 40

        alive_air_reward = 0.0 # per step

        if on_ground:
            # small per-step penalty just for wasting time
            alive_air_reward -= 0.5
        else:
            alive_air_reward +=0.01
        
        ##SOFT-CEILING PENALTY: when drone wanders past soft ceiling, start using a negative quadratic to punish it
        ##discourages wandering too close to hard ceiling value
        if z2 > self.soft_ceiling:
            k_ceiling = 5.0
            r_ceiling = -k_ceiling * (z2 - self.soft_ceiling) ** 2
        else:
            r_ceiling = 0.0
        ##TOTAL rward of all summed terms
        reward = (
            r_z
            + r_progress
            + r_vz
            + r_away
            + r_xy_base
            + r_xy_soft
            + r_vxy
            + r_tilt
            + r_omega
            + r_moment_abs
            + r_moment_jump
            + r_thrust_smooth
            + r_cmd_rate
            + r_vz_cmd
            + r_takeoff
            + r_ceiling
            + upright_bonus
            + ground_penalty
            + alive_air_reward
        )
        

        ##Hover band success conditions
        tilt_ok = tilt_angle2 < np.deg2rad(10.0)
        dz_band_rel = self.band / h_scale
        in_band = (abs(dz_rel) <= dz_band_rel) and (abs(vz2) < 0.05) and tilt_ok and (r_rad2 < 0.25)

        if in_band:##if the drone is under a vertical velocity, has a goo tilt and is in the band it's a good hover step
            self.hover_count += 1
            reward += 1.0

            if self.hover_count >= self.hover_required:##if we get to the hover count needed check histories for extra reward and then terminate as a success
                mean_vz = float(np.mean(self.vz_hist)) if len(self.vz_hist) else 999.0
                mean_du = float(np.mean(self.du_hist)) if len(self.du_hist) else 999.0
                info = {
                    "success": True,
                    "hover_steps": self.hover_count,
                    "mean_vz": mean_vz,
                    "mean_du": mean_du,
                }
                if mean_vz < 0.04 and mean_du < 0.010:
                    reward += 50.0
                    if self.auto_landing:
                        self._start_landing_phase("success")
                        info["phase"] = "landing_start"
                        return obs, reward, False, False, info
                    return obs, reward, True, False, info
                return obs, reward, True, False, info
        else:
            self.hover_count = 0

        ##hover band

        #Stalled on ground termination
        if self.ground_steps >= self.max_ground_steps:
            reward -= 100.0
            info = {
                "crash": True,
                "reason": "stalled_on_ground",
                "ground_steps": self.ground_steps,
            }
            return obs, reward, True, False, info
        ##crash/below ground termination
        if z2 < 0.01 or np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            reward -= 100.0
            return obs, reward, True, False, {"crash": True, "reason": "nan_or_below_ground"}
        ##flipped termination
        if tilt_angle2 > np.deg2rad(70.0):
            reward -= 150.0
            return obs, reward, True, False, {"crash": True, "reason": "flipped"}
        ##hitting hard ceiling termintion
        if z2 > self.hard_ceiling:
            reward -= 100.0
            return obs, reward, True, False, {"ceiling": True}
        ##exeeding lateral radius termination
        if r_rad2 > self.safety_radius:
            reward -= 100.0
            return obs, reward, True, False, {"crash": True, "reason": "out_of_bounds"}
        ##timeout truncation
        timeout = self.step_idx >= self.max_steps
        if timeout:
            info = {"hover_steps": self.hover_count, "timeout": True}
            if self.auto_landing:
                self._start_landing_phase("timeout")
                info["phase"] = "landing_start"
                return obs, reward, False, False, info
            return obs, reward, False, True, info

        return obs, float(reward), False, False, {
            "hover_steps": self.hover_count,
            "tilt_deg": float(tilt_deg2),
            "radius": float(r_rad2),
            "vz": float(vz2),
            "att_scale": float(dbg.get("att_scale", 1.0)),
        }

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
        self._apply_disturbances(dt)
        for _ in range(self.frame_skip):
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
