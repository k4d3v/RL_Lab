# RL_Lab: PILCO
The code is based on the paper http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf  . <br>
## Installation
* Write the platform name you want to test to `env_name` in `main.py`.<br>
* Setup the number of rollouts `J` and the number of iterations `N` in `pilco.py`.<br>
## Content
According to the paper, PILCO consists of three layers, the bottom layer learns the Transition Dynamics; the intermediate layer approximates Inference for Long-Term Predictions; the top layer uses gradient-based method to update the parameters of policy.
<br>
<br>
* `dyn_model.py` builds the dynamics model, which is implemented as a GP. <br>
* `policy.py` represents a RBF policy. <br>
* `pilco.py` contains the implementation of the PILCO algorithm, i.e. analytic approximate policy evaluation, and gradient based policy improvement.<br>

## Examples
