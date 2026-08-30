######################################################################################
######################################################################################
#
# Description: runs variable importance analysis
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

# paths to store importance results
i_primal_path = results_dir / f"importance_primal_v{version}_{suffix}.csv"
i_dual_path = results_dir / f"importance_dual_v{version}_{suffix}.csv"
strike = 1

pricer = Pricer(N_exercise=N_exercise, T=T, r=r, payoff=None, dim_W=2)

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
    "primal_layers": primal_layers,
    "primal_nodes": primal_nodes,
    "primal_epochs": primal_epochs,
    "primal_batch_size": primal_batch_size,
    "primal_learning_rate": primal_learning_rate,
    "primal_activation": primal_activation,
    "primal_dropout": primal_dropout,
    "primal_polynomials": primal_polynomials,
    "primal_state_spec": primal_state_spec,
    "primal_regularizer": primal_regularizer,
    "primal_regularizer_alpha": primal_regularizer_alpha,

    "dual_layers": dual_layers,
    "dual_nodes": dual_nodes,
    "dual_epochs": dual_epochs,
    "dual_batch_size": dual_batch_size,
    "dual_learning_rate": dual_learning_rate,
    "dual_activation": dual_activation,
    "dual_dropout": dual_dropout,
    "dual_polynomials": dual_polynomials,
    "dual_state_spec": dual_state_spec,
    "dual_regularizer": dual_regularizer,
    "dual_regularizer_alpha": dual_regularizer_alpha,
}

# define test that performs the variable importance analysis
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
    # train primal and dual
    pricer.train(primal_layers, primal_nodes, primal_epochs, primal_batch_size, primal_learning_rate, primal_activation, primal_dropout, primal_polynomials, primal_state_spec, primal_regularizer, primal_regularizer_alpha,
                dual_layers, dual_nodes, dual_epochs, dual_batch_size, dual_learning_rate, dual_activation, dual_dropout, dual_polynomials, dual_state_spec, dual_regularizer, dual_regularizer_alpha)

    print(f"Finished training networks in {(time.time()-start_time)}  sec")

    # variable importance primal
    print("Start the variable importance for primal")
    start_time = time.time()
    importance_primal = pricer.feature_importance("primal", 10, True)
    print(f"Variable importance for primal finished in {(time.time()-start_time)} sec")
    df_i_primal = pd.DataFrame(importance_primal)
    df_i_primal.to_csv(i_primal_path, index=False)

    # variable importance dual
    print("#######################################################")
    print("Start the variable importance for dual")
    start_time = time.time()
    importance_dual = pricer.feature_importance("dual", 10, True)
    print(f"Variable importance for dual finished  in {(time.time()-start_time)} sec")
    df_i_dual = pd.DataFrame(importance_dual)
    df_i_dual.to_csv(i_dual_path, index=False)

results = []

paths = pricer.create_paths(M_train_primal, M_test_primal, M_train_dual, M_test_dual, N_grid, X0, H, xi, eta, rho)

run_test(
    paths,
    K_trunc,
    signature_spec,
    signature_lift[0],
    signature_lift[1],
    strike,
)

results.append(
    {
        "Hurst parameter": H,
        "Signature Specification": signature_spec,
        "Signature Lift Primal": signature_lift[0],
        "Signature Lift Dual": signature_lift[1],
        "Signature Truncation Level": K_trunc,
        "Strike": strike,
    }
)

df = pd.DataFrame(results)

df.to_csv(csv_path, index=False)

print(f"Saved to {csv_path}, {i_primal_path}, {i_dual_path}")

# store log file
with open(log_path, "w") as f:
    f.write(f"Version: {version}\n")
    f.write(f"CSV file: {csv_path}\n")
    f.write("\nParameters:\n")

    for key, value in params_model.items():
        f.write(f"{key}: {value}\n")

print(f"Saved log to: {log_path}")