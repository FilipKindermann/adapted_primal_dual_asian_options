######################################################################################
######################################################################################
#
# Description: simulation of Black-Scholes model
#
######################################################################################
######################################################################################

import numpy as np

def simulate_BS_paths(M, N, T, X0, r, sigma):
    r"""
    Simulate underlying's price paths under the Black-Scholes model
    Parameters:
    - M ... number of samples
    - N ... number of time steps (finer grid)
    - T ... maturity
    - X0 ... starting level price
    - r ... constant risk-free interest rate
    - sigma ... constant volatility 
    """    
    # matrix to store the log price 
    X = np.zeros((M,N+1)) + np.log(X0)
    # drift of log price
    mu_log = r - sigma**2/2
    # time step 
    dt = T/N
    # Brownian motion increments per sample and time step
    dW = np.random.normal(0,np.sqrt(dt),(M,N))
    # for all time steps
    for j in range(N):
        # get increment of log price
        increment = mu_log * dt + sigma * dW[:,j]
        # add to log price matrix
        X[:,j+1] = X[:,j] + increment
    # get level price
    X = np.exp(X)
    # ensure shapes of level price and Brownian increments
    return np.reshape(X,(M,N+1)), np.reshape(dW,(M,N,1))