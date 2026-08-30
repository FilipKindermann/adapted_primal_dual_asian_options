######################################################################################
######################################################################################
#
# Description: runtime comparison of signature representations
#
######################################################################################
######################################################################################

from thesis_code.configs.baseline_config import *

import numpy as np
import pandas as pd
import time
from itertools import product
from pathlib import Path
import os

from thesis_code.deep_pricer import Pricer
from thesis_code.signature_computations import SignatureComputer

pricer = Pricer(N_exercise=N_exercise, T=T, r=r, payoff=None, dim_W=2)

# override vars here
strike = 1.0

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
}

# define test that measures how long it takes to compute the truncated signatures for each representation
def run_test(paths, K_trunc, signature_spec, signature_lift_primal, signature_lift_dual, strike):
    # create one signature
    def _create_single_sample(X, V, signature_spec, signature_lift, K_sig, strike=None, log_price=True):
            if log_prices:
                X = np.log(X)
            sig_comp = SignatureComputer(T, N_grid, K_sig, signature_spec, signature_lift, strike)
            input = sig_comp.compute_signature(X, V, log_price).astype(np.float32, copy=False)

    X_train, V_train, _, _, X_test, V_test, _, _ = paths    

    print("Start generating samples")
    start_time = time.time()
    # primal
    _create_single_sample(X_train, V_train, signature_spec, signature_lift_primal, K_trunc, strike, log_prices)
    _create_single_sample(X_test, V_test, signature_spec, signature_lift_primal, K_trunc, strike, log_prices)
    # dual
    _create_single_sample(X_train, V_train, signature_spec, signature_lift_dual, K_trunc, strike, log_prices)
    _create_single_sample(X_test, V_test, signature_spec, signature_lift_dual, K_trunc, strike, log_prices)

    run_time = (time.time()-start_time)
    print(f"Finished generating samples in {run_time} sec")

    return run_time

# tested parameters
signature_spec_params = ["standard signature", "basis words signature", "log signature"]
K_trunc_params = [2, 3, 4, 5]

# run all test cases
results = []

# repeat 20 times for stability
for i in range(1,21):
    paths = pricer.create_paths(M_train_primal, M_test_primal, M_train_dual, M_test_dual, N_grid, X0, H, xi, eta, rho)

    for signature_spec, K_trunc in product(
        signature_spec_params,
        K_trunc_params
    ):        
        runtime = run_test(
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
                "Run": i,
                "Signature Specification": signature_spec,
                "Signature Truncation Level": K_trunc,
                "Runtime": runtime,
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