import math

import gym
import numpy as np

from policy import Policy


class Plant:
    def __init__(self):
        pass

def create_settings(env_name):
    env = gym.make(env_name)
    s_dim = env.reset().shape[0]

    if env_name == 'CartpoleStabShort-v0':
        # TODO: Maybe use some gym-specific stuff??

        # Define state and important indices

        odei = [1, 2, 3, 4] # variables for the ode solver
        augi = [] # variables to be augmented
        dyno = [1, 2, 3, 4] # variables to be predicted( and known to loss)
        angi = [4] # angle variables
        dyni = [1, 2, 3, 5, 6] # variables that serve as inputs to the dynamics GP
        poli = [1, 2, 3, 5, 6] # variables that serve as inputs to the policy
        difi = [1, 2, 3, 4] # variables that are learned via differences


        # Set up scenario

        dt = 0.10 # [s] sampling time
        T = 4.0 # [s] initial prediction horizon time
        H = math.ceil(T / dt) # prediction steps(optimization horizon)
        mu0 = [0]*s_dim # initial state mean
        S0 = np.diag([0.1**2]*s_dim) # initial state covariance
        N = 15 # number controller optimizations
        J = 1 # initial J trajectories of lengthH
        K = 1 # no.of initial states for which we optimize
        nc = 10 # number of controller basis functions

        # Set up the plant structure

        plant = Plant()
        plant.dynamics = 0 # dynamics ode function, TODO: Dynamics from gym
        plant.noise = np.diag([0.01**2]*s_dim) # measurement noise
        plant.dt = dt
        plant.ctrl = 0 # TODO: What is zero order hold?
        plant.odei = odei
        plant.augi = augi
        plant.angi = angi
        plant.poli = poli
        plant.dyno = dyno
        plant.dyni = dyni
        plant.difi = difi
        plant.prop = 0 # TODO: What is @propagated?

        # Set up the policy structure

        policy = Policy()
        #policy.fcn = @(policy, m, s)conCat( @ congp, @gSat, policy, m, s) # controller representation # TODO: WTF is this!?
        policy.maxU = 10 # max.amplitude of control
        """
        [mm, ss, cc] = gTrig(mu0, S0, plant.angi) # represent angles in complex pane
        mm = [mu0;mm]
        cc = S0 * cc
        ss = [S0 cc; cc' ss]
        policy.p.inputs = gaussian(mm(poli), ss(poli, poli), nc) # init. location of basis functions
        """ # TODO: WTF dude
        policy.p.targets = 0.1 * np.random.normal(size = nc) # init.policy targets (close to zero)
        policy.p.hyp = np.log([1, 1, 1, 0.7, 0.7, 1, 0.01]) # initialize policy hyper-parameters

        # Set up the cost structure
        # Set up the GP dynamics model structure
        # Parameters for policy optimization
        # Plotting verbosity

        # Some array initializations
