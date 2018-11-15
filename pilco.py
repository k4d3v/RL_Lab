""" An implementation of the PILCO algorithm as shown in http://www.doc.ic.ac.uk/~mpd37/publications/pami_final_w_appendix.pdf"""

import numpy as np


class PILCO:
    def __init__(self, J=1, N=1):
        """
        :param J: Number of rollouts
        :param N: Number of iterations
        """
        self.J = J
        self.N = N

    def train(self):
        # Init.

        for j in range(self.J):
            # Sample controller params
            Theta = np.random.normal(size=(self.N, self.N))

            # Apply random control signals and record data

        for n in range(self.N):
            # Learn GP dynamics model using all data (Sec. 3.1)


            # Approx. inference for policy evaluation (Sec. 3.2)

            # Get J^pi(Theta) (9), (10), (11)

            # Policy improvement based on the gradient (Sec. 3.3)
            # Get the gradient of J (12)-(16)

            # Update Theta (CG or L-BFGS)

            # Convergence check


            # Set new optimal policy

            # Apply policy to system and record

            # Convergence check
            break

        return 0
