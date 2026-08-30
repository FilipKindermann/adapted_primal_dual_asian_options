######################################################################################
######################################################################################
#
# Description: runs finer grid size J sensitivity test
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

pricer = Pricer(N_exercise=N_exercise, T=T, r=r, payoff=None, dim_W=2)

# override vars here
N_grid = None

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
    # "N_grid": N_grid,
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

# define test that is run per parameter combination
def run_test(paths, K_trunc, signature_spec, signature_lift_primal, signature_lift_dual, strike):

    # asian put average def - average over 24 step grid for consistency
    def asian_put_average(paths, average_dates=24):
            n_paths, n_steps = paths.shape
            n_grid = n_steps - 1
    
            step_size = n_grid // average_dates
            average_indices = [0] + [step_size * i for i in range(1, average_dates)] + [n_grid]
            average_indices = np.array(average_indices)
    
            average_paths = paths[:, average_indices]
            avg = np.cumsum(average_paths, axis=1) / np.arange(1, average_paths.shape[1] + 1)
            
            interval_lengths = np.diff(average_indices, append=n_steps)
            average = np.repeat(avg, interval_lengths, axis=1)
    
            return average

    # corresponding payoff function
    def asian_put_payoff(paths, average_dates=24):
        avg = asian_put_average(paths, average_dates)
        payoff = np.maximum(strike - avg, 0.0)

        return payoff

    pricer.payoff = asian_put_payoff
    pricer.average_fn = asian_put_average

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

    start_time = time.time()
    print("Start pricing")
    # test primal and dual
    lower_bound, lower_std, upper_bound, upper_std, gap = pricer.test()

    print(f"Finish pricing in {(time.time()-start_time)} sec")
    print(f"Results for: strike={strike}, K_trunc={K_trunc}, signature_spec={signature_spec}, signature_lift_primal={signature_lift_primal}, signature_lift_dual={signature_lift_dual}")

    print(f"Lower Bound (Primal) : {lower_bound:.4f}  ±  {lower_std:.4f}")
    print(f"Upper Bound (Dual)   : {upper_bound:.4f}  ±  {upper_std:.4f}")
    print(f"Duality Gap          : {gap * 100:.2f}%")

    return lower_bound, lower_std, upper_bound, upper_std, gap

# tested parameters
N_grid_params = [24,48,96,120,240,360]

# run all test cases
results = []
for N_grid in N_grid_params:
    print("###########################################################################################################")
    print(f"Results for N_grid={N_grid}")
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
                "N Grid": N_grid,
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