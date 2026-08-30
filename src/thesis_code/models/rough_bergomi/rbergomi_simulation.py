######################################################################################
######################################################################################
#
# Description: simulation of rough Bergomi model with drift
#
######################################################################################
######################################################################################

# This script used the rough bergomi simulator of Ryan McCrickerd and adjusts the output
# to the need of the thesis. Specifically:
# - it is scaled by the starting value
# - a drift is added
# - the Hurst parameter is used directly

import numpy as np
from thesis_code.models.rough_bergomi.rbergomi_modified import rBergomi

def simulate_rbergomi_paths(M, N, T, X0, H, xi, eta, rho, r):
   r"""
   Simulate underlying's price and volatility paths under the rough Bergomi (rBergomi) model
   Parameters:
   - M ... number of paths (samples)
   - N ... number of time steps
   - T ... maturity
   - X0 ... starting level price
   - H ... Hurst parameter (roughness)
   - xi ... inital forward variance (assumption of constant curve)
   - eta ... vol-of-vol parameter
   - rho ... correlation between the Brownian motions of the variance and the price
   - r ... risk-free interest rate
   
   Returns:
   - X ... level price paths
   - v ... volatility paths
   - dW1 ... first Brownian motion - noise driver of variance
   - dW2 ... second Brownian motion - independent to the first
   - dB ... correlated Brownian motion - noise driver of level price
   - Y ... fractional Volterra process
   """

   # time grid
   tt = np.linspace(0, T, N + 1)

   # initializes simulator class using a = H - 0.5
   rB = rBergomi(n=N, N=M, T=T, a=H - 0.5)

   # get independent Brownian motions
   dW1 = rB.dW1()
   dW2 = rB.dW2()

   # get fractional Volterra process
   Y = rB.Y(dW1)

   # Variance process
   V = rB.V(Y, xi, eta)

   # Correlated Brownian motion
   dB = rB.dB(dW1, dW2, rho)
   
   # price of the simulator is a pure martingale
   # no drift yet
   martingale_X = rB.S(V, dB) 
   
   # scale by starting value and add drift
   X = X0 * martingale_X * np.exp(r * tt)

   # get volatility process
   v = np.sqrt(V)
   
   return X, v, dW1, dW2, dB, Y
   