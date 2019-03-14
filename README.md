# RL_Lab: PILCO
The code is based on the paper http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf  . <br>
Quanser platforms has been used as simulation and control interface, you can learn and evaluate the algorithm in simulation, or you can learn the algorithm in simulation and evaluate on the physical system, or you can also learn and evaluate the algorithm on the physical system.
There are 2 platforms, on which algorithms can be learnded and evaluated, i.e. `Inverted Pendulum` and `Ball in a Plate`. There are four different environments for testing in inverted pendulum, the simulation names of the environments are: `CartpoleStabShort-v0`, `CartpoleStabLong-v0`,`CartpoleSwingShort-v0`, `CartpoleSwingLong-v0`. The environment name of Ball in a Plate is `BallBalancerSim-v0`. You can change the environment name during initialization.

## Initialization
* Write the platform name you want to test to `env_name` in `main.py`.<br>
* Setup the number of rollouts `J` and the number of iterations `N` in `pilco.py`.<br>

## Content
According to the paper, PILCO consists of three layers, the bottom layer learns the Transition Dynamics; the intermediate layer approximates Inference for Long-Term Predictions; the top layer uses gradient-based method to update the parameters of policy.
<br>
<br>
* `main.py` Main file for testing the PILCO implementation. You can implement the  PILCO algorithm either by using an RBF policy in `pilco.py` or by using a linear policy in `policy_lin.py`. YOu can also change the environment name in the file.<br>
* `gp_policy.py` Gaussian Process with a squared exponential kernel for learning a policy. More details see https://publikationen.bibliothek.kit.edu/1000019799 <br>
* `linear_policy.py` Linear policy with weights Psi and an offset v. More details see https://publikationen.bibliothek.kit.edu/1000019799<br>
* `dyn_model.py` Gaussian Process with a squared exponential kernel for learning the system dynamics. <br>
* `pilco.py` contains the implementation of the PILCO algorithm by using an RBF policy without predicting the long-term state distribution.<br>
* `pilco_lin.py` contains the implementation of the PILCO algorithm by using a linear policy.<br>
* `test_models.py` For testing learnt policies in simulation. <br>

## Examples
### 1. Learning & Evaluation in Simulation
You can train either with a linear or an RBF policy. You can initialize a training agent in `main.py`.
* Choose the policy that you want to train by commenting one import and uncommenting the other. Note: Moment matching works only for the linear policy.
  ```python
  from pilco_lin import PILCO
    #from pilco import PILCO

    np.random.seed(12)

    env_names = ['CartpoleStabShort-v0', 'CartpoleStabLong-v0',
                 'CartpoleSwingShort-v0', 'CartpoleSwingLong-v0', 'BallBalancerSim-v0']

    for env_name in env_names:
        # Train agent
        agent = PILCO(env_name)
        optimal_policy = agent.train()
  ```