import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from CrazyFlieEnvVelocity2 import CrazyFlieEnvVelocity
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList

class DebugBehaviorCallback(BaseCallback):
    def __init__(self, eval_env, every_n_steps=100_000, n_episodes=5, verbose=0):
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
            ep_len = 0
            ep_rew = 0.0
            z_min, z_max = 1e9, -1e9
            max_tilt = 0.0
            reason = "timeout"

            done = [False]

            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, infos = self.eval_env.step(action)

                r0 = float(reward[0])
                info0 = infos[0]
                ep_rew += r0
                z = float(self.eval_env.get_attr("get_altitude")[0]())
                tilt = float(info0.get("tilt_deg", 0.0))

                z_min = min(z_min, z)
                z_max = max(z_max, z)
                max_tilt = max(max_tilt, tilt)

                if "reason" in info0:
                    reason = info0["reason"]

            print(
                f"[DBG ep {i}] len={ep_len:4d} R={ep_rew:8.1f} "
                f"z_min={z_min:6.3f} z_max={z_max:6.3f} "
                f"max_tilt={max_tilt:6.2f} deg  reason={reason}  info={info0}"
            )
        print("=" * 48 + "\n")
        return True



###factory for env making
def make_env(xml_path: str, target_z: float, max_steps: int = 1000, rank: int = 0):
    """
    Factory that creates one CrazyFlieEnvVelocity wrapped in a Monitor.
    Used by DummyVecEnv for parallel training.
    """
    def _f():
        env = CrazyFlieEnvVelocity(
            xml_path=xml_path,
            target_z=1,
            max_steps=1000,
            n_stack=4,
            hover_required_steps=300,
            auto_landing=False,

            # Stage-1: mild / almost no DR
            **DR_PARAMS,
        )
        env = Monitor(env)
        env.reset(seed=rank)
        return env

    return _f


if __name__ == "__main__":
    here = os.path.dirname(__file__)


    ##Domain randomization parameters
    DR_PARAMS = dict(
     
        start_z_min=0.0,
        start_z_max=0.0,
        random_start=True,     
        obs_noise_std   = 0.0,
        obs_bias_std    = 0.00,
        action_noise_std= 0.0,
        motor_scale_std = 0.0,

        start_xy_range  = 1.0 ,  
        torque_bias_std = 0,
        torque_gust_std = 0,

        drag_lin_max    = 0.0,
        drag_quad_max   = 0.0,
    

frame_skip_jitter= 0,     

    )

    ###Paths
    xml_path = os.path.abspath(
        os.path.join(here, "..", "Assets", "bitcraze_crazyflie_2", "scene.xml")
    )
    models_dir = os.path.abspath(os.path.join(here, "..", "models", "VelocityNew_Stage1_Clean"))
    logs_dir = os.path.abspath(os.path.join(here, "..", "logsVelocityNew_Stage1_Clean"))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Env constants
    TARGET_Z = 1.0
    MAX_STEPS = 1000
    N_ENVS = 16

        ##Building hte envs
    env_fns = [make_env(xml_path, TARGET_Z, MAX_STEPS, rank=i) for i in range(N_ENVS)]
    train_venv = DummyVecEnv(env_fns=env_fns)

    # Normalize OBS only (not rewards)
    train_venv = VecNormalize(
        train_venv,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    train_venv.training = True
    train_venv.norm_reward = False
    eval_env_fns = [make_env(xml_path, TARGET_Z, MAX_STEPS, rank=1000 + i) for i in range(N_ENVS)]
    eval_venv = DummyVecEnv(env_fns=eval_env_fns)
    eval_venv = VecNormalize(
        eval_venv,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )

    # Share obs normalization stats
    eval_venv.obs_rms = train_venv.obs_rms

    ###create ppo model
    model = PPO(
        "MlpPolicy",
        env=train_venv,
         learning_rate=1e-4,##lower learning rate for slow but stable learning
    clip_range=0.15,##clip range for PPO clipping method
    target_kl=0.02,       # let SB3 early-stop minibatches
        n_steps=2048, ## how many steps of experience to collect before each training update. Runs the environment for n_steps and then stores all the info(obs,action,reward) and do gradient updates (large leads to mroe stable gradients but more memory use)
        batch_size=64, ##minibatch size for each epoch, splits the large n_steps batch into mini-batches and traisn the neural network on that
        n_epochs=10,##the training doesn't scan the batch once, it goes over multiple times to train data, each iteration is an epoch. too many can lead to overfitting(10 is a good value)
        gamma=0.99,##discount factor, how much agent values future rewards over immediete
        gae_lambda=0.95,##generalized advantage estimation(GAE) it reduces noise when estimating how good an action was. 
       
        ent_coef=0.001, ###entropy coefficient, entropy is the randomness which encourages exploraiton in the loss funciton, higher means more exploraiton, lower means exploit what is already known
        vf_coef=0.5,##how much value funciton loss(critic) contributes in compairson wiht policy loss and entropy bonus
        policy_kwargs=dict(net_arch=[64, 64]),##architecture for neural network, two hidden layers of 64 neurons,
        tensorboard_log=logs_dir,##logs tensorboard log files into the logs director
        verbose=1##prints training progress into the console(1 is minimal)
    )

    ###create oru evaluationc callback for frequenty evaluation during training
    eval_callback = EvalCallback(
        eval_env=eval_venv,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=100_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    
    debug_env_fns = [make_env(xml_path, TARGET_Z, MAX_STEPS, rank=9999)]
    debug_venv = DummyVecEnv(env_fns=debug_env_fns)
    debug_venv = VecNormalize(
        debug_venv,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )

   
    debug_venv.obs_rms = train_venv.obs_rms

    debug_cb = DebugBehaviorCallback(debug_venv, every_n_steps=200_000, n_episodes=5)
    callback = CallbackList([eval_callback, debug_cb])

    model.learn(
        total_timesteps=2_000_000,
        progress_bar=True,
        reset_num_timesteps=True,  
        callback=callback,
    )
    stage1_model_path = os.path.join(models_dir, "complex.zip")
    stage1_vecnorm_path = os.path.join(models_dir, "vecnormalize.pkl")

    model.save(stage1_model_path)
    train_venv.save(stage1_vecnorm_path)

    print(f"Stage-1 model saved to: {stage1_model_path}")
    print(f"Stage-1 VecNormalize stats saved to: {stage1_vecnorm_path}")
    print("Use these as old_model_path and old_vecnorm_path in your Stage-2 (noisy) script.")
