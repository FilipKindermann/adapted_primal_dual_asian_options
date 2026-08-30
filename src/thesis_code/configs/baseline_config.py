######################################################################################
######################################################################################
#
# Description: sets paths for log and results and defines baseline parameters
#
######################################################################################
######################################################################################


from pathlib import Path
import os

path = os.environ.get(
    "RESULTS_DIR",
    "results"
)
suffix = os.environ.get(
    "RESULTS_SUFFIX",
    "local"
)

results_dir = Path(path)
results_dir.mkdir(parents=True, exist_ok=True)

existing_csv_files = list(results_dir.glob(f"results_v*_{suffix}.csv"))
version = len(existing_csv_files) + 1

csv_path = results_dir / f"results_v{version}_{suffix}.csv"
log_path = results_dir / f"log_v{version}_{suffix}.txt"

print(f"will be saved as v{version}_{suffix}")

T = 1.0
N_exercise = 12
r = 0.05

eta = 1.9
xi = 0.09
rho = -0.9
X0 = 1
H = 0.07
N_grid = 48
M_train_primal = 2**18
M_test_primal = 2**18
M_train_dual = 2**18
M_test_dual = 2**18
log_prices = True

primal_layers=3
primal_nodes=90
primal_epochs=15
primal_batch_size=2**8
primal_learning_rate=0.001
primal_activation='tanh'
primal_dropout=0
primal_polynomials=0
primal_state_spec=["average"]
primal_regularizer = "Ridge"
primal_regularizer_alpha = 0.0

dual_layers=6
dual_nodes=90
dual_epochs=15
dual_batch_size=2**8
dual_learning_rate=0.001
dual_activation='relu'
dual_dropout=0
dual_polynomials=0
dual_state_spec=["average"]
dual_regularizer = "Ridge"
dual_regularizer_alpha = 0.0

# signature_spec can be ["standard signature", "basis words signature", "log signature"]
signature_spec = "basis words signature"
# signature_lift can be ["baseline","volatility","qv_volatility","payoff"]
signature_lift = ["qv_volatility","volatility"]
# K_trunc can be [2, 3, 4, 5],
K_trunc = 3
strikes = [0.70, 0.8, 0.9, 1.00, 1.10, 1.20]