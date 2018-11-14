""" An implementation of the natural policy gradient algorithm as introduced in https://arxiv.org/abs/1703.02660"""

import numpy as np




def npg_pc(K, N, T, gamma):
    """ Implementation of policy search with NPG
    K -- Number of iterations
    N -- umber of trajs
    T -- Number of time steps (?)
    gamma -- Discount factor"""

    # Init. policy params
    Theta = 0

    for k in range(K):
        # Collect trajs by rolling out policy with Theta
        Taus = rollout(Theta, N)

        # Compute log gradient for ech state-action pair
        states, actions = [], []
        Nabla_Theta = np.zeros((len(states), len(actions)))
        for s in states:
            for a in actions:
                Nabla_Theta[s][a] = 0

        # Compute advantages and approx. value function
        A = 0
        V_k_old = 0

        # Compute policy gradient (2)
        # TODO: What is T? Maybe number of time steps?
        pg = pol_grad(Nabla_Theta, A, T)

        # Compute Fisher matrix (4)
        F = fish(Nabla_Theta, T)

        # Perform gradient ascent (5)
        # TODO: Normalize step size
        delta = 0
        Theta = grad_asc(Theta, pg, F, delta)

        # Update params of value function in order to approx. V(s_t^n)
        R = empi_re(states, N, T, gamma)
        v_params = update_v_params(R)

        # Return params of optimal policy
        return Theta


def rollout(Theta, N):
    """ Returns sampled trajs based on the stochastic policy
    Theta -- Parameters of the current policy
    N -- Number of trajectories"""
    return 0


def pol_grad(Nabla_Theta, A, T):
    """ Computes the policy gradient.
    Nabla_Theta -- Contains the log gradient for each state-action pair along trajs
    A -- Advantages based on trajs in current iteration
    T -- Number of time steps (?)"""
    return 0


def fish(Nabla_Theta, T):
    """ Computes the Fisher matrix.
    Nabla_Theta -- Log gragdient for each (s,a) pair
    T -- Number of time steps (?)"""
    return 0


def grad_asc(Theta, pg, F, delta):
    """Performs gradient ascent on the parameter function
    Theta -- Params of the current policy
    pg -- Policy gradient
    F -- Fisher information matrix
    delta -- Normalized step size"""
    return 0


def empi_re(states, N, T, gamma):
    """ Computes the empirical return for each state in each time step and every trajectory
    states -- The states in the current rollout
    N -- Number of trajs
    T -- Number of time steps (?)
    gamma -- Discount factor"""
    return 0


def update_v_params(R):
    """ Updates the params of the value function in order to approximate it according to the empirical reward
    R -- Empirical reward"""
    return 0
