# Crazyflie 2.1 Reinforcement Learning Project (MuJoCo → Real)
### **Reproducible training, evaluation, sim-to-real deployment, and firmware workflow**

## In Progress
**What this repo does**
- **Trains** a PPO policy (Stable-Baselines3) in **MuJoCo** to take off and hold a stable hover at 1 m.
- **Evaluates** the trained policy in simulation, headless or in the MuJoCo viewer, under many kinds of noise.
- **Deploys** the same policy to a **real Crazyflie 2.1** using **cflib** setpoints (roll, pitch, yaw rate, thrust).
- **Calibrates** sim-to-real thrust using hover thrust logging (counts per Newton).

---

## **1. Quick Overview**
### **Core idea**
The policy uses a **commander-style action space**. Each action is 4 numbers in [-1, 1]:
- **Roll** and **pitch** target angles (scaled by `max_roll_deg` / `max_pitch_deg`)
- **Yaw rate** (scaled by `max_yawrate_deg`)
- **Vertical speed command** (turned into thrust by a vertical PD controller)

The observation is the last 4 frames of a 13-number state, 52 numbers in total:
`[x_rel, y_rel, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]`.

```
policy action [roll, pitch, yawrate, vz] in [-1,1]
   SIM:  -> angle limits -> PD controller -> torque/thrust -> MuJoCo
   REAL: -> angle limits -> cf.commander.send_setpoint(roll, pitch, yawrate, thrust counts)
```

Every trained model is a pair of files that must stay together:
- `best_model.zip`: the policy network
- `vecnormalize_best.pkl`: the running mean and variance used to scale the 52 inputs. A model only works with the `.pkl` saved alongside it.

---

## **2. Repository Structure**

```
Assets/bitcraze_crazyflie_2/   MuJoCo model of the drone (cf2.xml) and world (scene.xml)
models/                        trained policies, one folder per run
logs/                          TensorBoard curves, evaluations.npz, noise sweep CSVs
src/
  common/paths.py              project paths and the drone's radio URI, in one place
  envs/
    sim_env.py                 MuJoCo training environment (reward, terminations, noise)
    real_env.py                real-drone twin of sim_env (same actions/observations)
    real_env_safe.py           real-drone env with an extra safety filter on actions
    state_observer.py          reads the drone's state over the radio (cflib logging)
  training/
    train_from_scratch.py      stage 1: new policy, 20M steps (RECONSTRUCTION, see file)
    train_warmstart.py         stage 2: continue from a model with hard noise, 40M steps
    callbacks.py               entropy schedule, noise curriculum, eval, debug callbacks
  evaluation/
    noise_sweep.py             headless robustness table for every noise type + viewer mode
    eval_viewer.py             one episode in the MuJoCo viewer under hard noise
    torque_bias_test.py        shows why torque noise is excluded from training
    real_flight.py             fly the policy on the real drone (real_env.py)
    real_flight_safe.py        fly the policy on the real drone (real_env_safe.py)
  tools/
    scan_radios.py             find nearby Crazyflies and print their URIs
    stream_state.py            connect and print the 13-number state live
    hover_sanity_check.py      non-RL takeoff/hover/land with cflib's MotionCommander
    hover_thrust_logger.py     measure hover thrust counts (writes logs/HoverThrust/)
    thrust_newton_check.py     thrust counts to Newtons conversion check
```

### **Which models are kept**
| Folder | What it is |
|---|---|
| `models/Velocity_Final` | Trained from scratch for 20M steps (best at 12.8M). Best clean hover. |
| `models/Velocity_WarmStart_HardDR_Long` | `Velocity_Final` trained 40M more steps with hard noise. Most recent and most noise-robust. |

Older runs were removed from the working tree. Their TensorBoard curves are still in `logs/`, and their weights are in git history (commit `bff2b0e`).

---

## **3. Requirements and Dependencies**
### **3.1 System requirements**
- **OS:** Windows 10/11 or Ubuntu 22.04+
- **Python:** **3.10+** (recommended)
- **Hardware (training):** CPU is fine; GPU optional (PyTorch will use it if available)
- **Hardware (real):**
  - **Crazyflie 2.1**
  - **Crazyradio PA** (recommended)
  - A stable estimator/positioning pipeline that produces valid `stateEstimate.*`


## **Software Installation Guide**
### **Python, Simulation, Reinforcement Learning, and Crazyflie Libraries**

This section describes **exactly how to install all required software libraries** needed to:
- run MuJoCo simulations,
- train PPO models using Stable-Baselines3,
- evaluate trained models in simulation,
- prepare for real-flight execution using Crazyflie's Python API.

> **Scope:**
> This section covers **software dependencies only** (Python + libraries).
> **Firmware flashing and embedded toolchains are covered separately.**

---

## **1. System Prerequisites**

### **1.1 Operating System**
- **Windows 10 / 11** *(recommended for Crazyflie tooling)*
- **Ubuntu 22.04+** *(recommended for MuJoCo stability)*

### **1.2 Python Version**
- **Python 3.10 or newer** is required
  (Stable-Baselines3 and MuJoCo Python bindings assume modern Python versions)

Verify your Python version:
```bash
python --version
```

### **Install prerequisites**
Ensure pip and python are installed. To check, run
```bash
python --version
pip --version
```
If either of these throws an error you must install both. It is better to do this in a Command Prompt separate from an IDE, as some IDE terminals have had trouble finding the python and pip paths. For the rest of this guide, open a fresh Command Prompt.

Ensure Git is installed as well.

### **2.0 Installation Instructions**
This repository includes a requirements.txt file, and we will use it to install all supporting libraries into a virtual environment.
Virtual environments keep a project's Python libraries separate from your system-wide Python and are highly recommended.

From the root of the repository run
```bash
python -m venv venv
venv\Scripts\activate
```
After activating the virtual environment you should see `(venv)` at the start of your command prompt line.

Afterwards upgrade pip
```bash
python -m pip install --upgrade pip
```

Now install the requirements stored in requirements.txt
```bash
pip install -r requirements.txt
```
To exit the virtual environment run
```bash
deactivate
```
To re-activate the virtual environment run
```bash
venv\Scripts\activate
```

## **Firmware Installation Guide**

NOTE: this guide assumes Windows as the base OS. Check other OS guides online for more detail.
You can find details in the Bitcraze documentation at: https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/install/

We follow those instructions in order. Ensure the Python and pip prerequisites are installed.

### 1. Install Zadig
Zadig installs the Crazyradio drivers on your laptop so the dongle can connect properly.

Follow these instructions: https://www.bitcraze.io/documentation/repository/crazyradio-firmware/master/building/usbwindows/

### 2. Install CfClient
The Crazyflie client is an interface that connects to the drone and is very helpful for tuning.

We install it from source:
```bash
git clone https://github.com/bitcraze/crazyflie-lib-python.git
cd crazyflie-lib-python
```

### 3. Install crazyflie-lib-python from source
Clone the repository and move into it
```bash
git clone https://github.com/bitcraze/crazyflie-lib-python.git
cd crazyflie-lib-python
```
Install the package
```bash
pip install -e .
```

### 4. Verifying CfClient
You should now be able to run cfclient from a command prompt:
```bash
cfclient
```

### 5. Installing WSL and toolchain
The firmware toolchain runs inside WSL (Ubuntu).

First install WSL
```bash
wsl --install
```

This installs Ubuntu 22.04. Inside a WSL shell, install the toolchain (works on Ubuntu 20.04 or 22.04):
```bash
sudo apt-get install make gcc-arm-none-eabi
```

Then switch back to a normal Command Prompt and run
```bash
pip.exe install cfclient
```

### 6. Installing firmware from source
You must be inside a WSL shell for this step. Open a fresh Command Prompt and run
```bash
wsl
```
The prompt text turns green when WSL is active.

Then clone the firmware and its dependencies recursively
```bash
git clone --recursive https://github.com/bitcraze/crazyflie-firmware.git
```
This takes a long time and may appear to pause. Do not type or interrupt it.

### 7. Compile firmware
In a WSL shell, move into the folder and configure
```bash
cd crazyflie-firmware
make cf2_defconfig
```
Build the firmware
```bash
make -j 12
```
This also takes a while.

This concludes the firmware installation.

---

## **4. Training and Evaluating Models**

**Every command below is run from the `src/` folder with the virtual environment active.**
Scripts are run as modules with `python -m package.module`. This puts `src/` on Python's import path, so every file can import `envs.sim_env`, `common.paths`, and so on the same way no matter where it lives.

```bash
cd src
..\venv\Scripts\activate
```

### **4.1 Training**
Training has two stages. Each script refuses to overwrite a folder that already has a `best_model.zip` unless you pass `--overwrite`.

```bash
# Stage 1: from scratch, 20M steps (reconstruction of the script that made Velocity_Final)
python -m training.train_from_scratch --run-name MyScratch

# Stage 2: warm-start from an existing model with hard noise, 40M steps
python -m training.train_warmstart --run-name MyWarmStart --source Velocity_Final

# Quick test that everything runs (a few seconds)
python -m training.train_from_scratch --run-name Smoke --total-steps 8192 --n-envs 2
```

Watch training curves with TensorBoard (from the project root):
```bash
tensorboard --logdir logs
```

### **4.2 Evaluating in simulation**
```bash
# Robustness table: every noise type and spawn condition, 20 episodes each, both models
python -m evaluation.noise_sweep
python -m evaluation.noise_sweep --episodes 5            # quicker

# Watch one condition in the MuJoCo viewer: MODEL NOISE_TYPE [LEVEL]
python -m evaluation.noise_sweep --watch HardDR_Long CLEAN
python -m evaluation.noise_sweep --watch HardDR_Long obs_noise 0.06
python -m evaluation.noise_sweep --watch Final ALL 2x
python -m evaluation.noise_sweep --watch HardDR_Long spawn_tilt 25

# Single viewer episode with custom noise (edit the env block in the file)
python -m evaluation.eval_viewer
```
The noise sweep saves its table to `logs/NoiseSweep/`.

### **4.3 Real drone**
1. Turn the Crazyflie on.
2. Find its address with `python -m tools.scan_radios` (or cfclient's scan button).
3. If it differs from `DEFAULT_URI` in `src/common/paths.py`, change it there. Every script reads it from that one place.
4. Check the link and state estimate: `python -m tools.stream_state`
5. Check that it can hover without RL: `python -m tools.hover_sanity_check`
6. Fly the policy:

```bash
python -m evaluation.real_flight            # sim-identical limits (70 deg roll/pitch)
python -m evaluation.real_flight --scaled   # conservative 4 deg limits
python -m evaluation.real_flight_safe       # safety-filtered env
```

All real-flight scripts stop the motors and power-cycle the drone on exit, including on Ctrl+C.

---

## **5. Known sim-to-real issues**
- **Saturated commands flip the real drone.** In the sim, roll/pitch commands go through a PD controller whose torque is clipped and scaled by a very small actuator gear (`gear=1e-5` in `cf2.xml`). The policy learned to push its actions to the limit. On hardware, the same action becomes a 70 degree target angle and flips the drone. `--scaled` (4 degrees) avoids the flip but drifts.
- **Hover thrust is about 6% high on hardware.** Sim mass is 0.035 kg (hover 0.34335 N), real mass is 0.033 kg. `real_env.py` multiplies Newtons by counts per Newton without normalising, so sim hover maps to about 44,500 counts versus a real hover of 41,940.
- **Yaw rate limit differs.** `real_env.py` uses 70 deg/s by default; the sim uses 45 deg/s.
- **Little thrust headroom in the sim.** Max thrust 0.4 N vs hover 0.343 N, so a motor scale below about 0.86 cannot take off. The noise sweep shows this as `stalled_on_ground` failures under `motor_scale` noise.
