""" An implementation of the PILCO algorithm as shown in http://www.doc.ic.ac.uk/~mpd37/publications/pami_final_w_appendix.pdf"""

import numpy as np

def train(N=1):
    # Init.

    # Sample controller params
    Theta =  np.random.normal(size=(N, N))

    # Apply random control signals and record data

    while True:
        # Learn GP dynamics model using all data (Sec. 3.1)

        while True:
            # Approx. inference for policy evaluation (Sec. 3.2)

            # Get J^pi(Theta) (9), (10), (11)

            # Policy improvement based on the gradient (Sec. 3.3)
            # Get the gradient of J (12)-(16)

            # Update Theta (CG or L-BFGS)

            # Convergence check
            break

        # Set new optimal policy

        # Apply policy to system and record

        # Convergence check
        break

    return 0