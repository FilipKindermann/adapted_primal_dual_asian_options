######################################################################################
######################################################################################
#
# Description: price using linear primal and dual with OLS and LP
#
######################################################################################
######################################################################################

from thesis_code.configs.baseline_config import *

import numpy as np
import pandas as pd
import time
from itertools import product
from pathlib import Path
import gurobipy as gp

from thesis_code.deep_pricer import Pricer

pricer = Pricer(N_exercise=N_exercise, T=T, r=r, payoff=None, dim_W=2)

# override vars here
M_train_primal = 2**17
M_test_primal = 2**17
M_train_dual = 10**3
M_test_dual = 10**5

primal_polynomials=5
dual_polynomials=5

signature_spec = "standard signature"
signature_lift = ["baseline","payoff"]
K_trunc = 5

# log used parameters
params_model = {
    "name": (Path(__file__).name),
    "version": version,
    "T": T,
    "N_exercise": N_exercise,
    "r": r,
    "eta": eta,
    "xi": xi,
    "rho": rho,
    "X0": X0,
    "N_grid": N_grid,
    "M_train_primal": M_train_primal,
    "M_test_primal": M_test_primal,
    "M_train_dual": M_train_dual,
    "M_test_dual": M_test_dual,
    "primal_polynomials": primal_polynomials,
    "primal_state_spec": primal_state_spec,
    "dual_polynomials": dual_polynomials,
    "dual_state_spec": dual_state_spec,
}

# enter your gurobipy credentials
params_env = {
# "WLSACCESSID": 'XXX',
# "WLSSECRET": 'XXX',
# "LICENSEID": XXX,
}
env = gp.Env(params=params_env)

# define test that is run per parameter combination
def run_test(paths, K_trunc, signature_spec, signature_lift_primal, signature_lift_dual, strike):

    # define asian payoff function with average on finer grid
    def asian_put_payoff(paths):
        avg_price = np.cumsum(paths, axis=1) / np.arange(1, np.shape(paths)[1]+1)
        return np.maximum(strike - avg_price, 0.0)

    pricer.payoff = asian_put_payoff

    start_time = time.time()
    print("Start generating samples")
    # creates signatures and other input objects
    pricer.create_samples(*paths,
                        signature_spec, signature_lift_primal, signature_spec, signature_lift_dual, K_trunc, strike, log_prices)

    print(f"Finished generating samples in {(time.time()-start_time)} sec")

    start_time = time.time()
    print("Start training networks")
    # train primal and dual with its linear implementation
    pricer.train_linear(primal_polynomials, primal_state_spec, dual_polynomials, dual_state_spec, env)

    print(f"Finished training networks in {(time.time()-start_time)}  sec")

    start_time = time.time()
    print("Start pricing")
    # test primal and dual with its linear implementation
    lower_bound, lower_std, upper_bound, upper_std, gap = pricer.test_linear()

    print(f"Finish pricing in {(time.time()-start_time)} sec")
    print(f"Results for: strike={strike}, K_trunc={K_trunc}, signature_spec={signature_spec}, signature_lift_primal={signature_lift_primal}, signature_lift_dual={signature_lift_dual}")

    print(f"Lower Bound (Primal) : {lower_bound:.4f}  ±  {lower_std:.4f}")
    print(f"Upper Bound (Dual)   : {upper_bound:.4f}  ±  {upper_std:.4f}")
    print(f"Duality Gap          : {gap * 100:.2f}%")

    return lower_bound, lower_std, upper_bound, upper_std, gap

# run all test cases
results = []

paths = pricer.create_paths(M_train_primal, M_test_primal, M_train_dual, M_test_dual, N_grid, X0, H, xi, eta, rho)

for strike in strikes:
    lower_bound, lower_std, upper_bound, upper_std, gap = run_test(
        paths,
        K_trunc,
        signature_spec,
        signature_lift[0],
        signature_lift[1],
        strike,
    )

    # store results
    results.append(
        {
            "Hurst parameter": H,
            "Signature Specification": signature_spec,
            "Signature Lift Primal": signature_lift[0],
            "Signature Lift Dual": signature_lift[1],
            "Signature Truncation Level": K_trunc,
            "Strike": strike,
            "Lower Bound": lower_bound,
            "Lower Bound Std": lower_std,
            "Upper Bound": upper_bound,
            "Upper Bound Std": upper_std,
            "Duality Gap": gap,
        }
    )

    df = pd.DataFrame(results)

    df.to_csv(csv_path, index=False)

print(f"Saved to {csv_path}")

# store log file
with open(log_path, "w") as f:
    f.write(f"Version: {version}\n")
    f.write(f"CSV file: {csv_path}\n")
    f.write("\nParameters:\n")

    for key, value in params_model.items():
        f.write(f"{key}: {value}\n")

print(f"Saved log to: {log_path}")

env.dispose()