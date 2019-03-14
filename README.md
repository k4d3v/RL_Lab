# RL_Lab
This repository contains the implementations for two reinforcement learning(RL) algorithms, i.e. `NPG`(natural policy gradient) and `PILCO`(Probabilistic Inference for Learning Control) with a focus on topics covered by the TU Darmstadt IAS Reinforcement Learning Classes, more details see https://www.ias.informatik.tu-darmstadt.de/Teaching/ReinforcementLearning . The implementations are based on recent papers and written in Python. The algorithms are tested on `Quanser-robots` platform, which has been used as simulation and control interface. You can learn and evaluate the algorithms in simulation, or you can learn the algorithms in simulation and evaluate on the physical system, or you can also learn and evaluate the algorithms on the physical system. 
## Getting Started
All the code in this repository has been tested in `ubuntu 16.04` and `18.04` with `Python 3.6`. Before running the code, make sure you have already installed `PyTorch` and `OpenAI Gym` on your computer.<br>
<br>
`Quanser platforms` has been used as simulator and real robot control interface, you can find the detailed installation instructions at https://git.ias.informatik.tu-darmstadt.de/quanser/clients.<br>
You can run these codes to get start.<br>
  ```python
  # Clone this repository into some folder
  cd ~; mkdir tmp; cd tmp
  git clone https://git.ias.informatik.tu-darmstadt.de/quanser/clients.git
  # Make sure you have Python >= 3.5.3 on your system. If that is not the case, install Python3.6
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install python3.6
  sudo apt-get install python3.6-venv
  # Create a virtual environment, activate it, and update it. You can also use an Anaconda virtual environment.
  python3.6 -m venv venv3
  source venv3/bin/activate
  pip3 install -U pip setuptools
  # Install the `quanser_robots` package
  cd clients
  pip3 install -e
  ```
