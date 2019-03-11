# RL_Lab: NPG
The code is based on the paper https://arxiv.org/pdf/1703.02660.pdf. <br>
Quanser platforms has been used as simulation and control interface, you can learn and evaluate the algorithm in simulation, or you can learn the algorithm in simulation and evaluate on the physical system, or you can also learn and evaluate the algorithm on the physical system.
There are 2 platforms, on which algorithms can be learnded and evaluated, i.e. `Inverted Pendulum` and `Ball in a Plate`. There are four different environments for testing in inverted pendulum, the simulation names of the environments are: `CartpoleStabShort-v0`, `CartpoleStabLong-v0`,`CartpoleSwingShort-v0`, `CartpoleSwingLong-v0`. The environment name of Ball in a Plate is `BallBalancerSim-v0`. You can change the environment name during initialization.

## Initialization
* Write the platform name which you want to test to `env_name` in `train_sim.py` for simulation and in `main_real.py` for real physical system..<br>
* Setup hyperparameters in `train_sim.py` for simulation and in `main_real.py` for real physical system.<br>

|hyperparameter| simulation| real system     |
| ----------| :-----------:  | :-----------: |
| number of iterations for plotting| num_iters| num_iters|
| learning rate     | adelta     | delta     |
| number of sampled trajectories per iteration | traj_samples_list     | traj_samples     |

## Content
* `main_sim.py` High level script for training a NPG agent on different simulated environments with varying hyperparameters.
The plots and learned policies are saved.<br>
* `main_real.py` Script for training a model on the real system after having pretrained in simulation. Plot and policy are saved.<br>
* `npg.py` represents the NPG algorithm. <br>
* `linear_policy.py` represents a linear policy, which is described in https://arxiv.org/pdf/1703.02660.pdf . <br>
* `rbf_policy.py` represents an RBF policy, which is described in https://arxiv.org/pdf/1703.02660.pdf . <br>
* `mlp_value_function.py` represents a neural network with three layers and an approximated value function.<br>
* `evaluate.py` defines a class for evaluating a learned policy. <br>
* `train_sim.py` returns different hyperparameters(number of iterations for plotting, delta, number of sampled trajectories per iteration) based on current environment.<br>
* `test_models.py` evaluates Model after learning with 100 rollouts and prints out total reward<br>
* `test_models_real.py` evaluates Model after learning with 100 rollouts, plots rewards together with iterations and saves the plot in a pdf file named after the environment name and saves it in figures_sim_real folder.<br>


## Examples
### 1. Learning & Evaluation in Simulation
You can train either with delta（learning rate）or with number of trajectories as hyperparameter. You can initialize it in `train_sim.py`.
* Choose the Hyperparameter that you want to train with.
  ```python
  # train_sim.py
  # Train with delta (learning rate) as hyperparam
  train(env_names, True)
  # Train with num of trajs as hyperparam
  train(env_names)
  ```
#### Environment: CartpoleStabShort-v0
* Hyperparameters:
  ```python
  # train_sim.py
  num_iters = range(0, 151, 15)
  adelta = np.linspace(5e-4, 2e-3, 3) if delta else 0.001
  traj_samples_list = 20 if delta else [5, 10, 20]
  ```
##### Train with delta (learning rate) as hyperparam
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures_delta/CartpoleStabShort-v0.png" width="400" height="300" div align=center> <br>
##### Train with number of trajectories as hyperparam
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures/CartpoleStabShort-v0.png" width="400" height="300" div align=center> <br>

#### Environment: CartpoleStabLong-v0
* Hyperparameter:
  ```python
  # train_sim.py
  num_iters = range(0, 151, 15)
  adelta = np.linspace(5e-4, 2e-3, 3) if delta else 0.001
  traj_samples_list = 20 if delta else [5, 10, 20]
  ```

##### Train with delta (learning rate) as hyperparam
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures_delta/CartpoleStabLong-v0.png" width="400" height="300" div align=center> <br>
##### Train with number of trajectories as hyperparam
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures/CartpoleStabLong-v0.png" width="400" height="300" div align=center> <br>


#### Environment: CartpoleSwingShort-v0
* Hyperparameter:
  ```python
  # train_sim.py
  num_iters = range(0, 151, 15)
  adelta = np.linspace(0.001, 0.01, 3) if delta else 0.0055
  traj_samples_list = 20 if delta else [10, 20, 30]
  ```
##### Train with delta (learning rate) as hyperparam
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures_delta/CartpoleSwingShort-v0.png" width="400" height="300" div align=center> <br>
##### Train with number of trajectories as hyperparam
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures/CartpoleSwingShort-v0.png" width="400" height="300" div align=center> <br>

#### Environment: CartpoleSwingLong-v0
* Hyperparameter:
  ```python
  # train_sim.py
  num_iters = range(0, 151, 15)
  adelta = np.linspace(0.001, 0.01, 3) if delta else 0.0055
  traj_samples_list = 20 if delta else [10, 20, 30]
  ```

##### Train with delta (learning rate) as hyperparam
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures_delta/CartpoleSwingLong-v0.png" width="400" height="300" div align=center> <br>
##### Train with number of trajectories as hyperparam
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures/CartpoleSwingLong-v0.png" width="400" height="300" div align=center> <br>

#### Environment: BallBalancerSim-v0
* Hyperparameter:
  ```python
  # train_sim.py
  num_iters = range(0, 201, 20)
  adelta = np.linspace(0.001, 0.01, 3) if delta else 0.01
  traj_samples_list = 50 if delta else [50, 100, 200]
  ```
##### Train with delta (learning rate) as hyperparam
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures_delta/BallBalancerSim-v0.png" width="400" height="300" div align=center> <br>
##### Train with number of trajectories as hyperparam
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures/BallBalancerSim-v0.png" width="400" height="300" div align=center> <br>




### 2. Learning in Simulation & Evaluation on the Physical System
#### Environment: CartpoleStabRR-v0
* Hyperparameter:
  ```python
  # test_models_real.py
  iters = [100]  # Number of iterations of training
  traj_samps = 20  # Number of sampled trajs per iteration during training
  ```
  
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures_sim_real/CartpoleStabRR-v0.png" width="400" height="300" div align=center> <br>
### 3. Learning & Evaluation on the Physical System

#### Environment: CartpoleStabRR-v0
* Hyperparameter:
  ```python
  # main_real.py
  num_iters, delta, traj_samples = range(10), 0.0055, 5
  ```
  
* Plot:        
<img src="https://github.com/k4d3v/RL_Lab/raw/NPG/images/figures_real_real/CartpoleStabRR-v0.png" width="400" height="300" div align=center> <br>
