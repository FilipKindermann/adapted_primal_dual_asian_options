######################################################################################
######################################################################################
#
# Description: runs Black-Scholes test
#
######################################################################################
######################################################################################


import numpy as np
import pandas as pd
import time
from itertools import product
from pathlib import Path
import os

from thesis_code.deep_pricer import Pricer

from thesis_code.configs.baseline_config import *

# baseline config overrides
S0 = 1
r = 0.06

N_exercise = 50

N_grid = 100

primal_polynomials=5
dual_polynomials=5

# log used parameters
params_model = {
    "name": (Path(__file__).name),
    "version": version,
    "N_exercise": N_exercise,
    "r": r,
    "S0": S0,
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

# define test that is run per parameter combination
def run_test(paths, K_trunc, signature_spec, signature_lift_primal, signature_lift_dual, strike):

    # asian put average def - average over N grid as in benchmark
    def asian_put_average(paths):
        step_size = int(N_grid / N_exercise)

        exercise_indices = ([0] + [step_size * i for i in range(1, N_exercise)] + [N_grid])
        exercise_indices = np.array(exercise_indices)

        exercise_paths = paths[:, exercise_indices]

        exercise_avg = (
            np.cumsum(exercise_paths[:, 1:], axis=1)
            / np.arange(1, exercise_paths.shape[1])
        )

        exercise_avg = np.concatenate(
            [np.zeros((paths.shape[0], 1), dtype=paths.dtype), exercise_avg],
            axis=1
        )

        interval_lengths = np.diff(
            exercise_indices,
            append=N_grid + 1
        )

        avg = np.repeat(
            exercise_avg,
            interval_lengths,
            axis=1
        )

        return avg

    # define asian put payoff function with this average
    def asian_put_payoff(paths):
        avg = asian_put_average(paths)

        payoff = np.zeros_like(avg)
        first_exercise_index = int(N_grid / N_exercise)
        payoff[:, first_exercise_index:] = np.maximum(strike - avg[:, first_exercise_index:],0.0)

        return payoff


    pricer.payoff = asian_put_payoff
    pricer.average_fn = asian_put_average

    start_time = time.time()
    print("Start generating samples")
    # creates signatures and other input objects
    pricer.create_samples_BS(*paths,
                            signature_spec, signature_lift_primal, signature_spec, signature_lift_dual,
                            K_trunc, strike, log_prices)

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

# benchmark cases tested
benchmark_cases = [
    {"original_S0": 36, "strike": 40/36, "T": 1.0, "sigma": 0.20, "reference_lower": 4.6135/36, "reference_upper": 4.6215/36},
    {"original_S0": 36, "strike": 40/36, "T": 1.0, "sigma": 0.40, "reference_lower": 5.9770/36, "reference_upper": 5.9979/36},
    {"original_S0": 38, "strike": 40/38, "T": 1.0, "sigma": 0.20, "reference_lower": 2.8179/38, "reference_upper": 2.8321/38},
    {"original_S0": 38, "strike": 40/38, "T": 1.0, "sigma": 0.40, "reference_lower": 4.5784/38, "reference_upper": 4.5993/38},
    {"original_S0": 40, "strike": 1.0,   "T": 1.0, "sigma": 0.20, "reference_lower": 1.5346/40, "reference_upper": 1.5434/40},
    {"original_S0": 40, "strike": 1.0,   "T": 1.0, "sigma": 0.40, "reference_lower": 3.4517/40, "reference_upper": 3.4694/40},
    {"original_S0": 42, "strike": 40/42, "T": 1.0, "sigma": 0.20, "reference_lower": 0.7767/42, "reference_upper": 0.7815/42},
    {"original_S0": 42, "strike": 40/42, "T": 1.0, "sigma": 0.40, "reference_lower": 2.5790/42, "reference_upper": 2.5933/42},
    {"original_S0": 44, "strike": 40/44, "T": 1.0, "sigma": 0.20, "reference_lower": 0.3592/44, "reference_upper": 0.3611/44},
    {"original_S0": 44, "strike": 40/44, "T": 1.0, "sigma": 0.40, "reference_lower": 1.8863/44, "reference_upper": 1.8977/44},
]

# overrides of normal tested cases
signature_lift = ["baseline","baseline"]
K_trunc = 5

results = []

for case in benchmark_cases:
    T = case["T"]
    sigma = case["sigma"]
    strike = case["strike"]
    reference_lower = case["reference_lower"]
    reference_upper = case["reference_upper"]
    pricer = Pricer(N_exercise=N_exercise, T=T, r=r, payoff=None, dim_W=1)

    paths = pricer.create_paths_BS(M_train_primal, M_test_primal, M_train_dual, M_test_dual, N_grid, S0, r, sigma)

    lower_bound, lower_std, upper_bound, upper_std, gap = run_test(
        paths,
        K_trunc,
        signature_spec,
        signature_lift[0],
        signature_lift[1],
        strike
    )

    # store results
    results.append(
        {
            "Benchmark": "Rasmussen",
            "T": T,
            "N_exercise": N_exercise,
            "N_grid": N_grid,
            "r": r,
            "S0": S0,
            "Sigma parameter": sigma,
            "Strike": strike,
            "Signature Specification": signature_spec,
            "Signature Lift Primal": signature_lift[0],
            "Signature Lift Dual": signature_lift[1],
            "Signature Truncation Level": K_trunc,
            "Lower Bound": lower_bound,
            "Lower Bound Std": lower_std,
            "Upper Bound": upper_bound,
            "Upper Bound Std": upper_std,
            "Duality Gap": gap,
            "Reference Lower": reference_lower,
            "Reference Upper": reference_upper,
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