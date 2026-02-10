# Crazyflie 2.1 Reinforcement Learning Project (MuJoCo → Real)
### **Reproducible training, evaluation, sim-to-real deployment, and firmware workflow**

## In Progress
**What this repo does**
- **Trains** a PPO policy in **MuJoCo** for Crazyflie 2.1 hover stabilization at a target height
- **Evaluates** the trained policy in simulation with a viewer.
- **Deploys** the same policy to a **real Crazyflie 2.1** using **cflib** setpoints (RPYT).
- **Calibrates** sim-to-real thrust using **hover thrust logging** (counts-per-Newton).



## **1. Quick Overview**
### **Core idea**
The policy uses a **commander-style action space**:
- **Roll** *(deg)*
- **Pitch** *(deg)*
- **Yaw rate** *(deg/s)*
- **Vertical command** *(mapped into thrust / vertical behavior)*

**Simulation**
- The RL environment converts policy actions into MuJoCo control signals.
- Observations include state needed for stabilization (position/velocity/orientation/gyro etc., depending on env implementation).

**Real flight**
- The policy’s outputs are mapped to **Crazyflie RPYT setpoints** using `cflib`.
- A state logger (`CrazyFlieStateObserver.py`) collects firmware logs (e.g., `stateEstimate.*`, `gyro.*`) and exposes them to the control loop.
- A hover thrust logger (`hover_thrust_logger.py`) measures **hover thrust counts** to compute **counts-per-Newton**.

---

## **2. Repository Structure**
### **Key files (Velocity2 stack)**
- **Simulation env**
  - `CrazyFlieEnvVelocity2.py` — main MuJoCo RL environment (reward, terminations, DR hooks, obs/action scaling)
- **Training**
  - `TrainVelocity2.py` — PPO training pipeline (VecNormalize, callbacks, checkpointing)
- **Simulation evaluation**
  - `EvaluationVelocity2.py` — loads model + VecNormalize stats and runs in MuJoCo viewer
- **Real env**
  - `CrazyFlieVelocity2RealEnv.py` — real-world env wrapper (connect, log, apply policy → setpoints)
- **Real evaluation**
  - `EvaluationVelocityReal.py` — runs the trained policy on hardware using the real env
- **State logging**
  - `CrazyFlieStateObserver.py` — Crazyflie firmware logger → observation vector
- **Connectivity sanity checks**
  - `test_connect.py` — scan for URIs
  - `test_state.py` — connect + print streamed state

### **Other useful utilities**
- `view.py` — open MuJoCo viewer for a given scene
- `hover_test.py` — quick sim hover sanity check
- `test_newtons.py` — thrust/newton conversion checks (if included in your workflow)

### **MuJoCo XML assets**
- `scene.xml`, `cf2.xml`


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
- prepare for real-flight execution using Crazyflie’s Python API.

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

### **Install prerequisites
ensure pip and python are installed . To check run
```bash
python --version
pip --version
```
if either of these throw an error you must install both. Note that it is better to do this in command prompt seperate from an IDE as certain IDE's command line interfaces have given us issues with recognizing python and pip paths. For the duration of this guide open up a fresh instance of command prompt

Ensure Git is installed as well

### ** 2.0 Intallation Instructions** 
This repository includes a requirement.txt file and we will use that to install all our supporting libraries with a virtual environment.
Virtual environments allow you to compartemanalize python libraries and versions to your system-wide python versions and are highly reccomended.

from the root of the repository run the command

```bash
python -m venv .venv
.venv\Scripts\activate
```
after actating the virtual env you should see a (venv) on your command prompt line

Afterwards install and upgrade pip

```bash
python -m pip install --upgrade pip
```

now we install our requirements which are stored in requirements.txt

```bash
pip install -r requirements.txt
```
In order to exit the virtual environment you can run
```bash
deactivate
```
in order to re-activate the virutal environment run
```bash
.venv\Scripts\activate
```
## **Firmware Installation Guide**

NOTE: this guide is intended for windows OS as the base OS. We encourage you to check out other OS guides online to find better details
you can find details on the bitcraze documentation at: https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/install/

We will be following these instructions in our repository. Ensure the prerequisites with python and pip are installed
We will go in a sequential order


 ### 1. Install Zadig
 Zadig will allow you to download the crazyradio drivers to your laptop so the dongle can properly talk to and connect

 follow these instrucitons: https://www.bitcraze.io/documentation/repository/crazyradio-firmware/master/building/usbwindows/

 ### 2. Install CfClient

 CrazyFlie client is an interface that allows you to connect with the drone and it is very helpful for tuning

 We will be installing from the source

 run

 ```bash
 git clone https://github.com/bitcraze/crazyflie-lib-python.git
 ```
 then
 ```bash
 cd crazyflie-lib-python
 ```
### 3. Install CrazyFlie-lib-python from source
 we will also be downlaod crazyflie lib directly from the source. Clone hte directory and move into it

```bash
git clone https://github.com/bitcraze/crazyflie-lib-python.git
cd crazyflie-lib-python
```
run installaiton packages
```bash
pip install -e .
```
### 4.Verifying Cfclient
At this point you should be able ot run cfclient from your command prompt by running
```bash
cfclient
```
in an open command prompt instance

### 5. installing WSL and toolchain

We will be installing a toolchain in this which includes WSL(ubuntu) for firmware

first and foremost install wsl
```bash
wsl --install
```

The aforementioned command will install Ubuntu 22.04, and so we will use the following command which is compatible with weither Ubuntu 20.04 or 22.04 to install the toolchain. Note you must be in a wsl instance which should be the case after installing it
```bash
$ sudo apt-get install make gcc-arm-none-eabi
```

now switch back to a command prompt instance and run

```bash
pip.exe install cfclient
```

### 6. Installing firmware from source
most important note is that you must be in a wsl instance on your command prompt. So to start fresh, open an instance of the command prompt.
Then, run
```bash
wsl
```
in your command prompt which should activate your wsl environment
the text will be in green and you will know.

then you may run the following command to recursively install the firmware dependancies

```bash
git clone --recursive https://github.com/bitcraze/crazyflie-firmware.git
```
Note that this installation will take a while and it will pause for a long time. Do not type any keys or interrupt the installation process

## 7. Compile firmware

now we must compile and build the firmware after installation

run the following in a wsl instance to first move into the directory then compile
```bash
cd crazyflie-firmware
make cf2_defconfig
```

build the framework with the following command
```bash
make -j 12
```
the above process will take a while

This concludes the installation for the firmware and now we have all we need to run our scripts and RL models



## Training and Evaluating Models
NOTE: Ensure you activate the virtual environment
The process and workflow is to train a model in simulation, evaluate the model in simulation, then evaluate the model on the real drone

to train a model you run 

```bash
python src\Training\TrainVelocity2.py
```
in the TrainVelocity2.py file, you can name the model how you want

in evaluation, go to EvaluationVelocit2.py, and then adjust the path/directory to point to the newly trained model's name

then run
```bash
python src\Evaluation\EvaluationVelocity2.py
```

Once a successful episode is performed without crashing, we evaluate it on the real drone by running


but this requires a proper connection to be set up

to set up  conneciton turn the crazyflie drone on by clicking the button

then in a command prompt instance open up cfclient
```bash
cfclient
```
Once cfclient is open, click scan to look for an open crazyflie connection

When a connection is found, take the URL for the crazyflie, open up EvaluationVelocityReal.py, and adjust the URL or test_connect.py or test_state.py

once that is done, run the EvaluationVelocityReal.py file
Also adjust the directory for the model and the name


```bash
python src\Evaluation\EvaluationVelocityReal.py
```


