######################################################################################
######################################################################################
#
# Description: estimate lower bound using linear primal with OLS with larger training sample
#
######################################################################################
######################################################################################

from thesis_code.configs.baseline_config import *

import numpy as np
import pandas as pd
import time
from itertools import product
from pathlib import Path

from thesis_code.deep_pricer import Pricer

# path to store coefficients
coef_path = results_dir / f"coef_v{version}_{suffix}.csv"
coef = []

pricer = Pricer(N_exercise=N_exercise, T=T, r=r, payoff=None, dim_W=2)

# override vars here
primal_polynomials=5

signature_spec = "standard signature"
signature_lift = ["baseline","baseline"]
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

# define test that is run per parameter combination
def run_test(paths, K_trunc, signature_spec, signature_lift_primal, strike):

    # define asian payoff function with average on finer grid
    def asian_put_payoff(paths):
        avg_price = np.cumsum(paths, axis=1) / np.arange(1, np.shape(paths)[1]+1)
        return np.maximum(strike - avg_price, 0.0)

    pricer.payoff = asian_put_payoff

    start_time = time.time()
    print("Start generating samples")
    # creates signatures and other input objects
    pricer.create_samples(*paths,
                        signature_spec, signature_lift_primal, signature_spec, signature_lift_primal, K_trunc, strike, log_prices)

    print(f"Finished generating samples in {(time.time()-start_time)} sec")

    start_time = time.time()
    print("Start training networks")
    # train primal with OLS
    pricer._train_primal_linear(primal_polynomials, primal_state_spec)

    print(f"Finished training networks in {(time.time()-start_time)}  sec")

    start_time = time.time()
    print("Start pricing")
    # test primal with OLS
    lower_bound, lower_std = pricer._test_primal_linear()
    
    print(f"Finish pricing in {(time.time()-start_time)} sec")
    print(f"Results for: strike={strike}, K_trunc={K_trunc}, signature_spec={signature_spec}, signature_lift_primal={signature_lift_primal}")

    print(f"Lower Bound (Primal) : {lower_bound:.4f}  ±  {lower_std:.4f}")

    # store coefficents
    for nn_id, model in enumerate(pricer.primal_models, start=1):
        if model is None:
            continue
        weights, bias = model.layers[-1].get_weights()

        weights = weights.flatten()

        coef.append({
            "nn_id": nn_id,
            "Strike": strike,
            **{f"coef_{j}": value for j, value in enumerate(weights)},
            "intercept": bias[0],
        })

    df_coef = pd.DataFrame(coef)
    df_coef.to_csv(coef_path, index=False)

    return lower_bound, lower_std

# run all test cases
results = []

paths = pricer.create_paths(M_train_primal, M_test_primal, M_train_dual, M_test_dual, N_grid, X0, H, xi, eta, rho)

for strike in strikes:
    lower_bound, lower_std = run_test(
        paths,
        K_trunc,
        signature_spec,
        signature_lift[0],
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