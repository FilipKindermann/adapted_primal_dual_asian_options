######################################################################################
######################################################################################
#
# Description: pricing implementation for neural networks and linear models
#
######################################################################################
######################################################################################

import numpy as np
import time
import scipy.special as sc

import tensorflow as tf
from keras.layers import Input, Dense, Subtract, TimeDistributed, Multiply, Lambda, Normalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import L1, L2

from thesis_code.models.black_scholes import simulate_BS_paths
from thesis_code.signature_computations import SignatureComputer
from thesis_code.models.rough_bergomi.rbergomi_simulation import simulate_rbergomi_paths

import gurobipy as gp
from gurobipy import GRB

class Pricer:
    def __init__(self, N_exercise, T, r, payoff, dim_W):
        """
        Initialize parameters relevant for pricing the instrument
        Parameters:
        - N_exerice ... number of exercise dates until maturity excluding t=0
        - T ... maturity in years
        - r ... interest rate p.a.
        - payoff ... function to compute the payoff of the American path dependent option (will be Asian in this project)
        - dim_W ... dimension of Brownian motion used in the dual model
        """
        self.N_exercise = N_exercise
        self.T = T
        self.r = r
        self.payoff = payoff
        self.dim_W = dim_W

        self.primal_models = None
        self.dual_model = None

    def _compute_payoff_matrix(self, paths):
        """
            Compute the payoff matrix in one place
            Parameters:
            - paths ... price from the underlying model
        """
        payoff_matrix = self.payoff(paths)
        return payoff_matrix

    def create_paths(self, M_train_primal, M_test_primal, M_train_dual, M_test_dual, N_grid, X0, H, xi, eta, rho):
        r"""
        Generates train and test paths under rough Bergomi
        Parameters:
        - M_train_primal ... number of training paths for primal
        - M_test_primal ... number of testing paths for primal
        - M_train_dual ... number of training paths for dual
        - M_test_dual ... number of testing paths for dual
        - N_grid ... number of time steps for signature and running integral
        - X0 ... initial asset spot price (S_0)
        - H ... Hurst parameter (H in (0, 0.5) for rough volatility)
        - xi ... Initial forward variance (\xi_0)
        - eta ... Volatility of volatility parameter (\eta)
        - rho ... Correlation coefficient between the asset and variance driving noises
        """
        # store as instance variables
        self.N_grid = N_grid
        self.M_train_primal = M_train_primal
        self.M_test_primal = M_test_primal
        self.M_train_dual = M_train_dual
        self.M_test_dual = M_test_dual

        M_train = max(M_train_primal, M_train_dual)
        M_test = max(M_test_primal, M_test_dual)

        # generate paths under rough Bergomi
        X_train, V_train, dW1_train, dW2_train, _, _ = simulate_rbergomi_paths(M_train, N_grid, self.T, X0, H, xi, eta, rho, self.r)
        X_test, V_test, dW1_test, dW2_test, _, _ = simulate_rbergomi_paths(M_test, N_grid, self.T, X0, H, xi, eta, rho, self.r)

        return [X_train, V_train, dW1_train, dW2_train, X_test, V_test, dW1_test, dW2_test]
    
    def _create_single_sample(self, X, V, dW1, dW2, signature_spec, signature_lift, K_sig, strike=None, log_price=True, dW=False):
        r"""
        Generates payoff matrix, average matrix, signature array and dW array for the given parameters
        One function executed for several paths and specifications
        Parameters:
        - X ... price paths
        - V ... volaility paths
        - dW1 ... first independent Brownian motion
        - dW2 ... second independent Brownian motion
        - signature_spec ... signature representation specification
        - signature_lift ... input process specification
        - K_sig ... truncation level
        - strike ... strike price
        - log_price ... whether to use log prices (True) or level prices (False)
        - dW ... whether dW array should be created
        
        """
        # create payoff matrix
        payoff_matrix = self._compute_payoff_matrix(X).astype(np.float32, copy=False)

        # create average matrix
        if hasattr(self, "average_fn") and self.average_fn is not None:
            # if the average is not over the finer grid
            average_matrix = self.average_fn(X).astype(np.float32, copy=False)
        else:
            # if the average is over the finer gird
            average_matrix = (np.cumsum(X, axis=1) / np.arange(1, np.shape(X)[1] + 1)).astype(np.float32, copy=False)

        if log_price:
            # work with log or level prices for the signature
            X = np.log(X)

        # generate the signature with the specified input process and representation
        sig_comp = SignatureComputer(self.T, self.N_grid, K_sig, signature_spec, signature_lift, strike)
        input = sig_comp.compute_signature(X, V, log_price).astype(np.float32, copy=False)

        # store price paths (log or level) and volatility paths
        X = X.astype(np.float32, copy=False)
        V = V.astype(np.float32, copy=False) if V is not None else None

        # create the dW array, if needed
        if dW:
            if self.dim_W == 1:
                # black scholes
                dW = np.expand_dims(dW1[:, :, 0], axis=-1).astype(np.float32, copy=False)
            elif self.dim_W == 2:
                # rough Bergomi
                dW = np.stack([dW1[:, :, 0], dW2], axis=-1).astype(np.float32, copy=False)
            return [payoff_matrix, average_matrix, X, V, input, dW]
        else:
            return [payoff_matrix, average_matrix, X, V, input, None]


    def create_samples(self, 
                       X_train, V_train, dW1_train, dW2_train, X_test, V_test, dW1_test, dW2_test,
                       signature_spec_primal, signature_lift_primal, signature_spec_dual, signature_lift_dual,
                       K_sig, strike=None, log_price=True):
        r"""
        Generates train and test paths' signature arrays, payoff matrices, average matrices, and dW arrays for primal and dual for rough Bergomi
        It calls _create_single_sample 4 times and stores the results as instance variables
        Parameters:
        - X_train ... price paths
        - V_train ... volaility paths
        - dW1_train ... first independent Brownian motion
        - dW2_train ... second independent Brownian motion
        - X_test ... price paths
        - V_test ... volaility paths
        - dW1_test ... first independent Brownian motion
        - dW2_test ... second independent Brownian motion
        - signature_spec_primal ... one of {"standard signature", "log signature", "basis words signature"} for the primal algorithm
        - signature_lift_primal ... one of {"baseline", "payoff", "volatility"} for the primal algorithm
        - signature_spec_dual ... one of {"standard signature", "log signature", "basis words signature"} for the dual algorithm
        - signature_lift_dual ... one of {"baseline", "payoff", "volatility"} for the dual algorithm
        - K_sig ... truncation level for the signature
        - strike ... strike price K used in the payoff lift
        - log_price ... whether to use log prices (True) or level prices (False)
        """
        values = dict()

        reusable = (signature_spec_primal == signature_spec_dual and signature_lift_primal == signature_lift_dual)

        # spec and lift are the same for primcal and dual
        if reusable:
            signature_spec = signature_spec_dual
            signature_lift = signature_lift_dual

            M_train_min = min(self.M_train_primal, self.M_train_dual)
            if M_train_min == self.M_train_dual:
                approach_train = "train_primal"
                approach_train_copy = "train_dual"
            else:
                approach_train = "train_dual"
                approach_train_copy = "train_primal"

            M_test_min = min(self.M_test_primal, self.M_test_dual)
            if M_test_min == self.M_test_dual:
                approach_test = "test_primal"
                approach_test_copy = "test_dual"
            else:
                approach_test = "test_dual"
                approach_test_copy = "test_primal"

            values[approach_train] = self._create_single_sample(X_train, V_train, dW1_train, dW2_train, signature_spec, signature_lift, K_sig, strike, log_price, dW=True)
            values[approach_test] = self._create_single_sample(X_test, V_test, dW1_test, dW2_test, signature_spec_dual, signature_lift_dual, K_sig, strike, log_price, dW=True)
            values[approach_train_copy] = [(value[:M_train_min] if value is not None else None) for value in values[approach_train]]
            values[approach_test_copy] = [(value[:M_test_min] if value is not None else None) for value in values[approach_test]]

            values["train_primal"][5] = None
            values["test_primal"][5] = None

        # spec and lift differ for primcal and dual
        else:
            values["train_primal"] = self._create_single_sample(X_train[:self.M_train_primal], V_train[:self.M_train_primal], None, None, signature_spec_primal, signature_lift_primal, K_sig, strike, log_price, dW=False)
            values["test_primal"] = self._create_single_sample(X_test[:self.M_test_primal], V_test[:self.M_test_primal], None, None, signature_spec_primal, signature_lift_primal, K_sig, strike, log_price, dW=False)

            values["train_dual"] = self._create_single_sample(X_train[:self.M_train_dual], V_train[:self.M_train_dual], dW1_train[:self.M_train_dual], dW2_train[:self.M_train_dual], signature_spec_dual, signature_lift_dual, K_sig, strike, log_price, dW=True)
            values["test_dual"] = self._create_single_sample(X_test[:self.M_test_dual], V_test[:self.M_test_dual], dW1_test[:self.M_test_dual], dW2_test[:self.M_test_dual], signature_spec_dual, signature_lift_dual, K_sig, strike, log_price, dW=True)

        # stores variables as instance variables
        self_name = ["payoff_matrix", "average_matrix", "X", "V", "input", "dW"]
        for key in values.keys():
            for i, value in enumerate(values[key]):
                setattr(self, self_name[i] + "_" + key, value)

        print("Signatures computed")

    def create_paths_BS(self, M_train_primal, M_test_primal, M_train_dual, M_test_dual, N_grid, X0, r, sigma):
        r"""
        Generates train and test paths under Black-Scholes
        Parameters:
        - M_train_primal ... number of training paths for primal
        - M_test_primal ... number of testing paths for primal
        - M_train_dual ... number of training paths for dual
        - M_test_dual ... number of testing paths for dual
        - N_grid ... number of time steps for signature and running integral
        - X0 ... initial asset spot price (S_0)
        - r ... interest rate p.a.
        - sigma ... volatility of the asset price
        """
        # stores relevant inputs as instance variables
        self.N_grid = N_grid
        self.M_train_primal = M_train_primal
        self.M_test_primal = M_test_primal
        self.M_train_dual = M_train_dual
        self.M_test_dual = M_test_dual

        M_train = max(M_train_primal, M_train_dual)
        M_test = max(M_test_primal, M_test_dual)

        # generates discrete Black-Scholes paths
        X_train, dW1_train = simulate_BS_paths(M_train, N_grid, self.T, X0, r, sigma)
        X_test, dW1_test = simulate_BS_paths(M_test, N_grid, self.T, X0, r, sigma)

        return [X_train, dW1_train, X_test, dW1_test]

    def create_samples_BS(self, 
                            X_train, dW1_train, X_test, dW1_test,
                            signature_spec_primal, signature_lift_primal, signature_spec_dual, signature_lift_dual,
                            K_sig, strike=None, log_price=True):
        r"""
        Generates train and test paths' signature arrays, payoff matrices, average matrices, and dW arrays for primal and dual for Black-Scholes
        It calls _create_single_sample 4 times and stores the results as instance variables
        Parameters:
        - X_train ... price paths
        - V_train ... volaility paths
        - dW1_train ... independent Brownian motion (just one for Black-Scholes)
        - X_test ... price paths
        - V_test ... volaility paths
        - dW1_test ... independent Brownian motion (just one for Black-Scholes)
        - signature_spec_primal ... one of {"standard signature", "log signature", "basis words signature"} for the primal algorithm
        - signature_lift_primal ... one of {"baseline", "payoff", "volatility"} for the primal algorithm
        - signature_spec_dual ... one of {"standard signature", "log signature", "basis words signature"} for the dual algorithm
        - signature_lift_dual ... one of {"baseline", "payoff", "volatility"} for the dual algorithm
        - K_sig ... truncation level for the signature
        - strike ... strike price K used in the payoff lift
        - log_price ... whether to use log prices (True) or level prices (False)        """

        # Black-Scholes has one Brownian driver
        self.dim_W = 1
        values = dict()

        reusable = (signature_spec_primal == signature_spec_dual and signature_lift_primal == signature_lift_dual)

        # spec and lift are the same for primcal and dual
        if reusable:
            signature_spec = signature_spec_dual
            signature_lift = signature_lift_dual

            M_train_min = min(self.M_train_primal, self.M_train_dual)
            if M_train_min == self.M_train_dual:
                approach_train = "train_primal"
                approach_train_copy = "train_dual"
            else:
                approach_train = "train_dual"
                approach_train_copy = "train_primal"


            M_test_min = min(self.M_test_primal, self.M_test_dual)
            if M_test_min == self.M_test_dual:
                approach_test = "test_primal"
                approach_test_copy = "test_dual"
            else:
                approach_test = "test_dual"
                approach_test_copy = "test_primal"

            values[approach_train] = self._create_single_sample(X_train, None, dW1_train, None, signature_spec, signature_lift, K_sig, strike, log_price, dW=True)
            values[approach_test] = self._create_single_sample(X_test, None, dW1_test, None, signature_spec_dual, signature_lift_dual, K_sig, strike, log_price, dW=True)
            values[approach_train_copy] = [(value[:M_train_min] if value is not None else None) for value in values[approach_train]]
            values[approach_test_copy] = [(value[:M_test_min] if value is not None else None) for value in values[approach_test]]

            values["train_primal"][5] = None
            values["test_primal"][5] = None

        # spec and lift differ for primcal and dual
        else:
            values["train_primal"] = self._create_single_sample(X_train[:self.M_train_primal], None, None, None, signature_spec_primal, signature_lift_primal, K_sig, strike, log_price, dW=False)
            values["test_primal"] = self._create_single_sample(X_test[:self.M_test_primal], None, None, None, signature_spec_primal, signature_lift_primal, K_sig, strike, log_price, dW=False)

            values["train_dual"] = self._create_single_sample(X_train[:self.M_train_dual], None, dW1_train[:self.M_train_dual], None, signature_spec_dual, signature_lift_dual, K_sig, strike, log_price, dW=True)
            values["test_dual"] = self._create_single_sample(X_test[:self.M_test_dual], None, dW1_test[:self.M_test_dual], None, signature_spec_dual, signature_lift_dual, K_sig, strike, log_price, dW=True)

        # stores variables as instance variables
        self_name = ["payoff_matrix", "average_matrix", "X", "V", "input", "dW"]
        for key in values.keys():
            for i, value in enumerate(values[key]):
                setattr(self, self_name[i] + "_" + key, value)

        print("Signatures computed")

    def train(self, 
              primal_layers=2, primal_nodes=64, primal_epochs=100, primal_batch_size=32, primal_learning_rate=0.001, primal_activation='leaky_relu', primal_dropout = 0.1, primal_degree_poly = 3, primal_state_spec = None, primal_regularizer = None, primal_regularizer_alpha = 0,
              dual_layers=3, dual_nodes=128, dual_epochs=200, dual_batch_size=32, dual_learning_rate=0.001, dual_activation='leaky_relu', dual_dropout = 0.1, dual_degree_poly = 5, dual_state_spec = None, dual_regularizer = None, dual_regularizer_alpha = 0):
        """
        Trains both primal and dual models with the given specifications by calling the respective training function
        Parameters:
        - primal_layers ... number of hidden layers for primal
        - primal_nodes ... number of nodes per hidden layers for primal
        - primal_epochs ... number of epochs for the optimization of primal
        - primal_batch_size ... batch size for the optimization of primal
        - primal_learning_rate ... learning rate for the optimization of primal
        - primal_activation ... activation function for each hidden layers for primal
        - primal_dropout ... dropout percentage for each hidden layers for primal
        - primal_degree_poly ... how many Laguerre polynomials of the state variables (log) price and volatility are added for primal
        - primal_state_spec ... further additional variables for primal as an array with elements "payoff" and/or "average"
        - primal_regularizer ... L1 or L2 regularizer for each hidden layers for primal - is either "Lasso" or "Ridge"
        - primal_regularizer_alpha ... penalty of regularizer for each hidden layers for primal
        - dual_layers ... number of hidden layers for dual
        - dual_nodes ... number of nodes per hidden layers for dual
        - dual_epochs ... number of epochs for the optimization of dual
        - dual_batch_size ... batch size for the optimization of dual
        - dual_learning_rate ... learning rate for the optimization of dual
        - dual_activation ... activation function for each hidden layers for dual
        - dual_dropout ... dropout percentage for each hidden layers for dual
        - dual_degree_poly ... how many Laguerre polynomials of the state variables (log) price and volatility are added for dual
        - dual_state_spec ... further additional variables for dual as an array with elements "payoff" and/or "average"
        - dual_regularizer ... L1 or L2 regularizer for each hidden layers for dual - is either "Lasso" or "Ridge"
        - dual_regularizer_alpha ... penalty of regularizer for each hidden layers for dual
        """
        if primal_state_spec is None:
            primal_state_spec = []
        if dual_state_spec is None:
            dual_state_spec = []

        # train primal given the parameters
        print("Starting primal")
        start_time = time.time()
        self._train_primal(primal_layers, primal_nodes, primal_epochs, primal_batch_size, primal_learning_rate, primal_activation, primal_dropout, primal_degree_poly, primal_state_spec, primal_regularizer, primal_regularizer_alpha)
        print(f"Finished training primal networks in {(time.time()-start_time)}  sec")

        # train dual given the parameters
        print("Starting dual")
        start_time = time.time()
        self._train_dual(dual_layers, dual_nodes, dual_epochs, dual_batch_size, dual_learning_rate, dual_activation, dual_dropout, dual_degree_poly, dual_state_spec, dual_regularizer, dual_regularizer_alpha)
        print(f"Finished training dual networks in {(time.time()-start_time)}  sec")

    def test(self):
        """
        Estimates lower and upper bounds with the trained primal and dual models by calling the respective test function and computes the duality gap
        """
        # price primal
        price_primal, price_primal_std = self._test_primal()
        # price dual
        price_dual, price_dual_std = self._test_dual()

        # compute duality gap
        duality_gap = (price_dual - price_primal) / price_dual
        return price_primal, price_primal_std, price_dual, price_dual_std, duality_gap

    def _add_polynomials(self, input_data, X, V, degree):
        """
        Adds Laguerre polynomials of the state (log) price and volatility to the input for the neural network, without cross terms
        Parameters:
        - input_data ... current signature input for neural network
        - X ... (log) price paths
        - V ... volatility paths
        - degree ... defines up to which degree the Laguerre polynomials are added
        """
        if degree is None or degree == 0:
            return input_data

        # add Laguerre polynomials of the state (log) price
        poly_list_X = [sc.eval_laguerre(k, X) for k in range(1,degree+1)]
        poly_array_X = np.stack(poly_list_X, axis=-1)
        input_added = np.concatenate([input_data,poly_array_X], axis=-1)

        # add Laguerre polynomials of the volatility
        # not happening for Black-Scholes - then V is None
        if V is not None:
            poly_list_V = [sc.eval_laguerre(k, V) for k in range(1,degree+1)]
            poly_array_V = np.stack(poly_list_V, axis=-1)
            input_added = np.concatenate([input_added,poly_array_V], axis=-1)
        
        return input_added

    def _add_state_variables(self, input_data, X, payoff_matrix, average_matrix, state_specs):
        """
        Adds additional state variables like the state average with "average", the state (log) price with "price", and the state payoff with "payoff" to the signature input
        Parameters:
        - input_data ... current signature input for neural network
        - X ... (log) price paths
        - payoff_matrix ... payoff matrix with the payoff for each time step and sample
        - average_matrix ... average matrix with the running average for each time step and sample
        - state_specs ... defines which additional variable is added - including "price", "payoff", "average"
        """
        input_added = input_data
        M, N, D = np.shape(input_added)
        # add (log) price for for each time step and sample
        if "price" in state_specs:
            input_added = np.concatenate([input_added, np.reshape(X, (M,N,1))], axis=-1)
        # add state payoff for for each time step and sample
        if "payoff" in state_specs:
            input_added = np.concatenate([input_added, np.reshape(payoff_matrix, (M,N,1))], axis=-1)
        # add state average for for each time step and sample
        if "average" in state_specs:
            input_added = np.concatenate([input_added, np.reshape(average_matrix, (M,N,1))], axis=-1)
        return input_added

    def _train_primal(self, layers, nodes, epochs, batch_size, learning_rate, activation, dropout, degree_poly, state_specs, regularizer, regularizer_alpha):
        """
        Trains primal neural networks with the given specifications - based on the paper "Pricing American Options under Rough Volatility Using Deep-Signatures and Signature-Kernels" of Bayer, Pelizzari, and Zhu (2025 preprint)
        Parameters:
        - layers ... number of hidden layers for primal
        - nodes ... number of nodes per hidden layers for primal
        - epochs ... number of epochs for the optimization of primal
        - batch_size ... batch size for the optimization of primal
        - learning_rate ... learning rate for the optimization of primal
        - activation ... activation function for each hidden layers for primal
        - dropout ... dropout percentage for each hidden layers for primal
        - degree_poly ... how many Laguerre polynomials of the state variables (log) price and volatility are added for primal
        - state_spec ... further additional variables for primal as an array with elements "payoff" and/or "average"
        - regularizer ... L1 or L2 regularizer for each hidden layers for primal - is either "Lasso" or "Ridge"
        - regularizer_alpha ... penalty of regularizer for each hidden layers for primal
        """
        input_train_primal = self.input_train_primal
        X_train_primal = self.X_train_primal
        V_train_primal = self.V_train_primal
        payoff_matrix_train_primal = self.payoff_matrix_train_primal
        average_matrix_train_primal = self.average_matrix_train_primal
        self.degree_poly_primal = degree_poly
        self.state_specs_primal = state_specs

        # add additional variables
        input_train_primal = self._add_polynomials(input_train_primal, X_train_primal, V_train_primal, self.degree_poly_primal)
        input_train_primal = self._add_state_variables(input_train_primal, X_train_primal, payoff_matrix_train_primal, average_matrix_train_primal, state_specs)

        # prepare regularization for each layer
        reg_nn = None
        if regularizer:
            if regularizer == "Ridge":
                reg_nn = L2(regularizer_alpha)
            elif regularizer == "Lasso":
                reg_nn = L1(regularizer_alpha)
            else:
                print(f"Unvalid regularizer {regularizer}: choose Ridge or Lasso")

        M, N_grid_plus_one, D = np.shape(input_train_primal)
        N_grid = N_grid_plus_one - 1
        TimePoints_grid = np.linspace(0,self.T, N_grid+1)
        # get indices of exercise dates - to get the value of X at that moment from the finer price path 
        step_size = int(N_grid / self.N_exercise)
        exercise_indices = [0] + [step_size * i for i in range(1, self.N_exercise)] + [N_grid]
        # compute the payoff at each exercise date if it is exercised at corresponding exercise date
        early_payoff_matrix = payoff_matrix_train_primal[:,exercise_indices]
        
        # will store NN models per exercise date
        models = [None] * (self.N_exercise-1)
        # true continuation values 
        ## will be filled in backwards induction
        realized_discounted_payoffs = np.zeros((M,self.N_exercise+1))
        # starting with the last exercise date where the continuation value is just the payoff at maturity
        realized_discounted_payoffs[:, -1] = early_payoff_matrix[:, -1]

        # trains one neural netowrk per time step (excluding start and end) with backward recursion
        for i in reversed(range(1,self.N_exercise)):
            # correct output at this time step
            time_step = TimePoints_grid[exercise_indices[i+1]] - TimePoints_grid[exercise_indices[i]]
            realized_discounted_payoff_t = (realized_discounted_payoffs[:,i+1] * np.exp(-self.r*time_step)).astype(np.float32, copy=False)
            # input at this time step
            input_NN_t = input_train_primal[:,exercise_indices[i],:].astype(np.float32, copy=False)

            # only ITM for training
            mask = early_payoff_matrix[:,i] > 0
            if np.sum(mask) < 2:
                realized_discounted_payoffs[:, i] = realized_discounted_payoff_t
                continue

            input_NN_t_ITM = input_NN_t[mask,:]
            realized_discounted_payoff_t_ITM = realized_discounted_payoff_t[mask]

            # design neural network per time step
            normalizer = Normalization()

            input = Input(shape=(D,))
            output = normalizer(input)
            for _ in range(layers):
                output = Dense(nodes, activation=activation, kernel_regularizer=reg_nn)(output)
                output = Dropout(dropout)(output)
            output = Dense(1, activation='linear')(output)

            model = Model(inputs=input, outputs=output)
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
            normalizer.adapt(input_NN_t_ITM)

            # train network extensively for in the first iteration (last time step before maturity)
            if i == self.N_exercise-1 or models[i] is None:
                epochs_t = epochs
                early_callback = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)
                model.fit(input_NN_t_ITM, realized_discounted_payoff_t_ITM, epochs=epochs_t, batch_size=batch_size, verbose=0, validation_split=0.2, callbacks=[early_callback])
            # trains network less for other iterations - weights get initialized by weights from last iteration
            else:
                model.set_weights(models[i].get_weights())
                epochs_t = 3
                model.fit(input_NN_t_ITM, realized_discounted_payoff_t_ITM, epochs=epochs_t, batch_size=batch_size, verbose=0)

            # predict continuation value
            continuation_value_next = model.predict(input_NN_t_ITM, verbose=0).flatten()
            # update ex post payoff value for the next NN model
            # current payoff if continuation value < immediate exercise payoff, otherwise set to dependent variable
            realized_discounted_payoffs[:, i] = realized_discounted_payoff_t
            realized_discounted_payoffs[mask, i] = np.where((early_payoff_matrix[mask, i] > continuation_value_next), early_payoff_matrix[mask, i], realized_discounted_payoff_t_ITM)
            # store trained neural network
            models[i-1] = model

        # store trained neural network list for pricing in an instance variable
        self.primal_models = models


    def _test_primal(self):
        """
        Estimates lower bound with the trained primal models and also return the standard error of the continuation value estimate
        """
        input_test_primal = self.input_test_primal
        X_test_primal = self.X_test_primal
        V_test_primal = self.V_test_primal
        payoff_matrix_test_primal = self.payoff_matrix_test_primal
        average_matrix_test_primal = self.average_matrix_test_primal

        # add additional variables to input
        input_test_primal = self._add_polynomials(input_test_primal, X_test_primal, V_test_primal, self.degree_poly_primal)
        input_test_primal = self._add_state_variables(input_test_primal, X_test_primal, payoff_matrix_test_primal, average_matrix_test_primal, self.state_specs_primal)

        M, N_grid_plus_one, D = np.shape(input_test_primal)
        N_grid = N_grid_plus_one - 1

        # only inputs at exercise dates needed
        TimePoints_grid = np.linspace(0,self.T, N_grid+1)
        step_size = int(N_grid / self.N_exercise)
        exercise_indices = [0] + [step_size * i for i in range(1, self.N_exercise)] + [N_grid]

        early_payoff_matrix_test = payoff_matrix_test_primal[:,exercise_indices]
        
        continuation_values_test = np.zeros((M,self.N_exercise+1))
        continuation_values_test[:, -1] = early_payoff_matrix_test[:, -1]

        # predict all continuation values
        for i in range(1,self.N_exercise):
            if self.primal_models[i-1] is None:
                continuation_values_test[:, i] = np.inf
                continue
            input_NN_t = input_test_primal[:,exercise_indices[i],:].astype(np.float32, copy=False)
            continuation_value_t = self.primal_models[i-1].predict(input_NN_t, verbose=0).flatten()
            continuation_values_test[:,i] = continuation_value_t

        # stop the first time the continuation value < immediate payoff for each sample
        stop_cond = (early_payoff_matrix_test > 0) & (early_payoff_matrix_test >= continuation_values_test)
        stop_cond[:, 0] = False
        stop_cond[:, -1] = True
        stopping_indices = np.argmax(stop_cond, axis=1)

        # find the associated stopping times and payoffs
        stopping_times = TimePoints_grid[exercise_indices][stopping_indices]
        stopping_payoffs = early_payoff_matrix_test[np.arange(M), stopping_indices]

        # discount payoffs according to this stopping rule
        discounted_stopping_payoff = stopping_payoffs * np.exp(-self.r*stopping_times)
        # get continuation value at t=0 estimate
        C0 = np.mean(discounted_stopping_payoff)
        # get lower bound
        lower_bound = max(early_payoff_matrix_test[0,0], C0)
        # get standard error of continuation value at t=0 estimate
        C0_std = np.std(discounted_stopping_payoff, ddof=1) / np.sqrt(M)
        return lower_bound, C0_std

    def _train_dual(self, layers, nodes, epochs, batch_size, learning_rate, activation, dropout, degree_poly, state_specs, regularizer, regularizer_alpha):
        """
        Trains dual neural network with the given specifications - based on the paper "Pricing American Options under Rough Volatility Using Deep-Signatures and Signature-Kernels" of Bayer, Pelizzari, and Zhu (2025 preprint)
        Parameters:
        - layers ... number of hidden layers for dual
        - nodes ... number of nodes per hidden layers for dual
        - epochs ... number of epochs for the optimization of dual
        - batch_size ... batch size for the optimization of dual
        - learning_rate ... learning rate for the optimization of dual
        - activation ... activation function for each hidden layers for dual
        - dropout ... dropout percentage for each hidden layers for dual
        - degree_poly ... how many Laguerre polynomials of the state variables (log) price and volatility are added for dual
        - state_spec ... further additional variables for dual as an array with elements "payoff" and/or "average"
        - regularizer ... L1 or L2 regularizer for each hidden layers for dual - is either "Lasso" or "Ridge"
        - regularizer_alpha ... penalty of regularizer for each hidden layers for dual
        """
        input_train_dual = self.input_train_dual
        X_train_dual = self.X_train_dual
        V_train_dual = self.V_train_dual
        dW_train_dual = self.dW_train_dual
        payoff_matrix_train_dual = self.payoff_matrix_train_dual
        average_matrix_train_dual = self.average_matrix_train_dual

        self.degree_poly_dual = degree_poly
        self.state_specs_dual = state_specs

        # add additional variables to signature input
        input_train_dual = self._add_polynomials(input_train_dual, X_train_dual, V_train_dual, self.degree_poly_dual)
        input_train_dual = self._add_state_variables(input_train_dual, X_train_dual, payoff_matrix_train_dual, average_matrix_train_dual, self.state_specs_dual)

        # set up regularizer for each hidden layer
        reg_nn = None
        if regularizer:
            if regularizer == "Ridge":
                reg_nn = L2(regularizer_alpha)
            elif regularizer == "Lasso":
                reg_nn = L1(regularizer_alpha)
            else:
                print(f"Unvalid regularizer {regularizer}: choose Ridge or Lasso")

        M, N_grid_plus_one, D = np.shape(input_train_dual)
        N_grid = N_grid_plus_one - 1

        # compute payoff matrix for minimization problem of the dual approach
        TimePoints_grid = np.linspace(0,self.T, N_grid+1)
        ## get indices of exercise dates - to get the value of X at that moment from the finer price path 
        step_size = int(N_grid / self.N_exercise)
        exercise_indices = [0] + [step_size * i for i in range(1, self.N_exercise)] + [N_grid]
        ## compute the payoff at each exercise date if it is exercised at corresponding exercise date
        payoff_matrix = payoff_matrix_train_dual[:,exercise_indices]

        discounted_payoff = (payoff_matrix * np.exp(-self.r*TimePoints_grid[exercise_indices])).astype(np.float32, copy=False)

        normalizer = Normalization()
        normalizer.adapt(input_train_dual[:, :-1, :].reshape(-1, D))

        # NN archtecture for the integrand
        input_integrand = Input(shape=(D,))
        output_integrand = normalizer(input_integrand)
        for _ in range(layers):
            output_integrand = Dense(nodes, activation=activation, kernel_regularizer=reg_nn)(output_integrand)
            output_integrand = Dropout(dropout)(output_integrand)
        output_integrand = Dense(self.dim_W, activation='linear')(output_integrand)
        model_integrand = Model(inputs=input_integrand, outputs=output_integrand)
        
        # include in larger NN for each time step
        input_NN = Input(shape=(N_grid,D))
        integrand_all = TimeDistributed(model_integrand)(input_NN)

        # compute martingale from the integrand
        input_dW = Input(shape=(N_grid,self.dim_W))
        integrand_times_dW = Multiply()([integrand_all, input_dW])        
        
        def _compute_objective(args):
            integrand_times_dW, payoff = args
            dM = tf.reduce_sum(integrand_times_dW, axis=-1)
            M_t = tf.cumsum(dM, axis=-1)
            m0 = tf.zeros((tf.shape(M_t)[0], 1), dtype=M_t.dtype)
            M_full = tf.concat([m0, M_t], axis=1) 
            m_exercise = tf.gather(M_full, exercise_indices, axis=1)
            return tf.reduce_max(payoff - m_exercise, axis=1)
        
        # compute the dual objective
        payoff_input = Input(shape=(self.N_exercise+1,))
        output = Lambda(_compute_objective)([integrand_times_dW, payoff_input])

        model_dual = Model(inputs=[input_NN, input_dW, payoff_input], outputs=[output])

        # custom loss function to minimize the dual objective
        def dual_loss(y_true, y_pred):
            return tf.reduce_mean(y_pred)
            
        model_dual.compile(optimizer=Adam(learning_rate=learning_rate), loss=dual_loss)
        
        # train model
        dummy_y = np.zeros((M, 1))
        early_callback = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)
        model_dual.fit(x=[input_train_dual[:,:-1,:], dW_train_dual, discounted_payoff], y=dummy_y,epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2, callbacks=[early_callback])

        # store the neural network for pricing in an instance variable
        self.dual_model = model_dual
        
    def _test_dual(self):
        """
        Estimates upper bound with the trained dual models and also return the standard error of the estimate
        """
        input_test_dual = self.input_test_dual
        X_test_dual = self.X_test_dual
        V_test_dual = self.V_test_dual
        dW_test = self.dW_test_dual
        payoff_matrix_test_dual = self.payoff_matrix_test_dual
        average_matrix_test_dual = self.average_matrix_test_dual

        # add additional variables to signature input
        input_test_dual = self._add_polynomials(input_test_dual, X_test_dual, V_test_dual, self.degree_poly_dual)
        input_test_dual = self._add_state_variables(input_test_dual, X_test_dual, payoff_matrix_test_dual, average_matrix_test_dual, self.state_specs_dual)

        M, N_grid_plus_one, D = np.shape(input_test_dual)
        N_grid = N_grid_plus_one - 1

        # payoff matrix is only needed at exercise dates
        TimePoints_grid = np.linspace(0,self.T, N_grid+1)
        step_size = int(N_grid / self.N_exercise)
        exercise_indices = [0] + [step_size * i for i in range(1, self.N_exercise)] + [N_grid]
        payoff_matrix = payoff_matrix_test_dual[:,exercise_indices]
        discounted_payoff = (payoff_matrix * np.exp(-self.r*TimePoints_grid[exercise_indices])).astype(np.float32, copy=False)

        # use neural network to predict the max differences per sample
        input_NN_test = input_test_dual[:,:-1,:].astype(np.float32, copy=False)
        dual_objective_test = self.dual_model.predict([input_NN_test, dW_test, discounted_payoff], verbose=0).flatten()

        # average of max differences per sample is equal to the upper bound estimate
        upper_bound = np.mean(dual_objective_test)
        # standard error of estimate
        Upper_bound_std = np.std(dual_objective_test) / np.sqrt(M)

        return upper_bound, Upper_bound_std

    ####################################################################################################################
    # Variable Importance Analysis
    ####################################################################################################################

    def _prepare_primal_input_importance(self):
        """
        Get signature input with additional variables for the variable importance analysis for primal
        """
        input_test_primal = self.input_test_primal
        X_test_primal = self.X_test_primal
        V_test_primal = self.V_test_primal
        payoff_matrix_test_primal = self.payoff_matrix_test_primal
        average_matrix_test_primal = self.average_matrix_test_primal

        # add addtional variables to signature input array for test of primal
        input_test_primal = self._add_polynomials(input_test_primal, X_test_primal, V_test_primal, self.degree_poly_primal)
        input_test_primal = self._add_state_variables(input_test_primal, X_test_primal, payoff_matrix_test_primal, average_matrix_test_primal, self.state_specs_primal)

        return input_test_primal

    def _test_primal_importance(self, input_test_primal):
        """
        Function is equal to the primal test function, but the additional variables are already included in the signature array
        Parameter:
        - input_test_primal ... signature array including additional variables
        """
        # this version is not futher commented for compactness - explanatory comments are in _test_primal
        payoff_matrix_test_primal = self.payoff_matrix_test_primal

        M, N_grid_plus_one, D = np.shape(input_test_primal)
        N_grid = N_grid_plus_one - 1

        TimePoints_grid = np.linspace(0,self.T, N_grid+1)
        step_size = int(N_grid / self.N_exercise)
        exercise_indices = [0] + [step_size * i for i in range(1, self.N_exercise)] + [N_grid]

        early_payoff_matrix_test = payoff_matrix_test_primal[:,exercise_indices]
        
        continuation_values_test = np.zeros((M,self.N_exercise+1))
        continuation_values_test[:, -1] = early_payoff_matrix_test[:, -1]
        
        for i in range(1,self.N_exercise):
            if self.primal_models[i-1] is None:
                continuation_values_test[:, i] = np.inf
                continue
            input_NN_t = input_test_primal[:,exercise_indices[i],:].astype(np.float32, copy=False)
            continuation_value_t = self.primal_models[i-1].predict(input_NN_t, verbose=0).flatten()
            continuation_values_test[:,i] = continuation_value_t
        
        stop_cond = (early_payoff_matrix_test > 0) & (early_payoff_matrix_test >= continuation_values_test)
        stop_cond[:, 0] = False
        stop_cond[:, -1] = True
        stopping_indices = np.argmax(stop_cond, axis=1)

        stopping_times = TimePoints_grid[exercise_indices][stopping_indices]
        stopping_payoffs = early_payoff_matrix_test[np.arange(M), stopping_indices]
        
        discounted_stopping_payoff = stopping_payoffs * np.exp(-self.r*stopping_times)
        C0 = np.mean(discounted_stopping_payoff)
        lower_bound = max(early_payoff_matrix_test[0,0], C0)
        C0_std = np.std(discounted_stopping_payoff, ddof=1) / np.sqrt(M)
        return lower_bound, C0_std

    def _prepare_dual_input_importance(self):
        """
        Get signature input with additional variables for the variable importance analysis for dual
        """
        input_test_dual = self.input_test_dual
        X_test_dual = self.X_test_dual
        V_test_dual = self.V_test_dual
        payoff_matrix_test_dual = self.payoff_matrix_test_dual
        average_matrix_test_dual = self.average_matrix_test_dual

        # add addtional variables to signature input array for test of dual
        input_test_dual = self._add_polynomials(input_test_dual, X_test_dual, V_test_dual, self.degree_poly_dual)
        input_test_dual = self._add_state_variables(input_test_dual, X_test_dual, payoff_matrix_test_dual, average_matrix_test_dual, self.state_specs_dual)

        return input_test_dual

    def _test_dual_importance(self, input_test_dual):
        """
        Function is equal to the dual test function, but the additional variables are already included in the signature array
        Parameter:
        - input_test_dual ... signature array including additional variables
        """
        # this version is not futher commented for compactness - explanatory comments are in _test_dual
        dW_test = self.dW_test_dual
        payoff_matrix_test_dual = self.payoff_matrix_test_dual

        M, N_grid_plus_one, D = np.shape(input_test_dual)
        N_grid = N_grid_plus_one - 1

        TimePoints_grid = np.linspace(0,self.T, N_grid+1)
        step_size = int(N_grid / self.N_exercise)
        exercise_indices = [0] + [step_size * i for i in range(1, self.N_exercise)] + [N_grid]
        payoff_matrix = payoff_matrix_test_dual[:,exercise_indices]
        discounted_payoff = (payoff_matrix * np.exp(-self.r*TimePoints_grid[exercise_indices])).astype(np.float32, copy=False)
        
        input_NN_test = input_test_dual[:,:-1,:].astype(np.float32, copy=False)
        dual_objective_test = self.dual_model.predict([input_NN_test, dW_test, discounted_payoff], verbose=0).flatten()

        upper_bound = np.mean(dual_objective_test)
        Upper_bound_std = np.std(dual_objective_test) / np.sqrt(M)

        return upper_bound, Upper_bound_std
    

    def feature_importance(self, approach, num_permutations, verbose=False):
        """
        The variable importance analysis measures the importance of input variables by the change in the bound estimate after permutation
        Stores the change directly as Error and as a percentage of the standard bound estimate as Error pct
        Parameters:
        - approach ... "primal" or "dual" for the respective variable importance analysis
        - num_permutations ... how often should the procedure be repeated for an input variable
        - verbose ... whether progress should be printed to the console
        """
        # determine the correct test function and input
        if approach == "primal":
            if self.primal_models is None:
                raise ValueError(f"{approach} was not trained yet")
            test = self._test_primal_importance
            input_test = self._prepare_primal_input_importance()
            target_attr = "input_test_primal"
            multiplier = -1
        elif approach == "dual":
            if self.dual_model is None:
                raise ValueError(f"{approach} was not trained yet")
            test = self._test_dual_importance
            input_test = self._prepare_dual_input_importance()
            target_attr = "input_test_dual"
            multiplier = 1
        else:
            raise ValueError("approach must be either 'primal' or 'dual'")
        
        # Compute standard estimate
        baseline, _ = test(input_test)

        original_input = input_test
        _, _, N_features = np.shape(original_input)
        # first entry is the standard estimate
        feature_importances = [{
                        "Dimension": -1,
                        "Error": baseline,
                        "Error pct": 0,
        }]
        # loop for each feature
        for i in range(N_features):
            start_feature = time.time()
            feature_errors = np.zeros((num_permutations,))
            # loop num_permutations times for stability 
            for j in range(num_permutations):
                start_permuation = time.time()
                # shuffle input variable across samples for each time point
                permuted_input = original_input.copy()
                shuffled_indices = np.random.permutation(np.shape(input_test)[0])
                permuted_input[:, :, i] = permuted_input[shuffled_indices, :, i]
                # compute estimate with shuffled input
                permuted_score, _ = test(permuted_input)
                # compute difference
                feature_errors[j] = (permuted_score - baseline) * multiplier
                if verbose:
                    print(f"---Iteration {j+1}/{num_permutations} finished - {(time.time()-start_permuation)}---")

            # average differences across num_permutations repetitions
            feature_mean_error = np.mean(feature_errors)
            # compute average differences as percent of standard estimate
            feature_mean_error_pct = feature_mean_error/baseline
            # store results
            feature_importances.append(
                {
                    "Dimension": i,
                    "Error": feature_mean_error,
                    "Error pct": feature_mean_error_pct,
                }
            )
            if verbose:
                print(f"{i+1}/{N_features} finished - {(time.time()-start_feature)}")
    
        return feature_importances
    
    ####################################################################################################################
    # Linear implementations
    ####################################################################################################################

    def train_linear(self, 
              primal_degree_poly = 3, primal_state_spec = None,
              dual_degree_poly = 5, dual_state_spec = None, env = None):
        """
        Train primal and dual using linear models - OLS for primal and LP for dual - by calling the respective train_XXX_linear functions
        Parameters:
        - primal_degree_poly ... how many Laguerre polynomials of the state variables (log) price and volatility are added for primal
        - primal_state_spec ... further additional variables for primal as an array with elements "payoff" and/or "average"
        - dual_degree_poly ... how many Laguerre polynomials of the state variables (log) price and volatility are added for dual
        - dual_state_spec ... further additional variables for dual as an array with elements "payoff" and/or "average"
        """
        if primal_state_spec is None:
            primal_state_spec = []
        if dual_state_spec is None:
            dual_state_spec = []

        # train primal using OLS 
        print("Starting primal")
        start_time = time.time()
        self._train_primal_linear(primal_degree_poly, primal_state_spec)
        print(f"Finished training primal networks in {(time.time()-start_time)}  sec")

        # train dual using LP
        print("Starting dual")
        start_time = time.time()
        self._train_dual_linear(dual_degree_poly, dual_state_spec, env)
        print(f"Finished training dual networks in {(time.time()-start_time)}  sec")

    def test_linear(self):
        """
        Get lower and upper bound estiamtes using the trained linear models by calling the respective test_XXX_linear functions
        """
        # lower bound estimate using trained linear model of primal
        price_primal, price_primal_std = self._test_primal_linear()
        # upper bound estimate using trained linear model of dual
        price_dual, price_dual_std = self._test_dual_linear()

        # compute duality gap
        duality_gap = (price_dual - price_primal) / price_dual
        return price_primal, price_primal_std, price_dual, price_dual_std, duality_gap

    def _train_primal_linear(self, degree_poly, state_specs):
        """
        Trains primal linear model using OLS with the given specifications - based on the paper "Primal and dual optimal stopping with signatures" by Bayer, Pelizzari, and Schoenmakers (2025)
        Linear version of the function _train_primal
        Parameters:
        - degree_poly ... how many Laguerre polynomials of the state variables (log) price and volatility are added for primal
        - state_specs ... further additional variables for primal as an array with elements "payoff" and/or "average"
        """
        input_train_primal = self.input_train_primal
        X_train_primal = self.X_train_primal
        V_train_primal = self.V_train_primal
        payoff_matrix_train_primal = self.payoff_matrix_train_primal
        average_matrix_train_primal = self.average_matrix_train_primal
        self.degree_poly_primal = degree_poly
        self.state_specs_primal = state_specs

        # add additional variables to the signature input array
        input_train_primal = self._add_polynomials(input_train_primal, X_train_primal, V_train_primal, self.degree_poly_primal)
        input_train_primal = self._add_state_variables(input_train_primal, X_train_primal, payoff_matrix_train_primal, average_matrix_train_primal, state_specs)
        
        M, N_grid_plus_one, D = np.shape(input_train_primal)
        N_grid = N_grid_plus_one - 1
        TimePoints_grid = np.linspace(0,self.T, N_grid+1)
        # get indices of exercise dates - to get the value of X at that moment from the finer price path 
        step_size = int(N_grid / self.N_exercise)
        exercise_indices = [0] + [step_size * i for i in range(1, self.N_exercise)] + [N_grid]
        # compute the payoff at each exercise date if it is exercised at corresponding exercise date
        early_payoff_matrix = payoff_matrix_train_primal[:,exercise_indices]
        
        # will store NN models per exercise date
        models = [None] * (self.N_exercise-1)
        # true continuation values 
        ## will be filled in backwards induction
        continuation_values = np.zeros((M,self.N_exercise+1))
        # starting with the last exercise date where the continuation value is just the payoff at maturity
        continuation_values[:, -1] = early_payoff_matrix[:, -1]

        for i in reversed(range(1,self.N_exercise)):
            # correct output at this time step
            time_step = TimePoints_grid[exercise_indices[i+1]] - TimePoints_grid[exercise_indices[i]]
            continuation_value_t = (continuation_values[:,i+1] * np.exp(-self.r*time_step)).astype(np.float32, copy=False)
            # input at this time step
            input_NN_t = input_train_primal[:,exercise_indices[i],:].astype(np.float32, copy=False)

            # only ITM for training
            mask = early_payoff_matrix[:,i] > 0
            if np.sum(mask) < 2:
                continuation_values[:, i] = continuation_value_t
                continue

            input_NN_t_ITM = input_NN_t[mask,:]
            continuation_value_t_ITM = continuation_value_t[mask]

            # define neural network with just a linear readout
            input = Input(shape=(D,))
            output = Dense(1, activation='linear')(input)

            model = Model(inputs=input, outputs=output)

            X_reg = np.column_stack([
                input_NN_t_ITM,
                np.ones(input_NN_t_ITM.shape[0])
            ])

            # the weights of the neural network are not trained by ADAM
            # but we take the coefficients from the OLS estimator
            coef, _, _, _ = np.linalg.lstsq(X_reg, continuation_value_t_ITM, rcond=None)

            weights = coef[:-1].reshape(D, 1).astype(np.float32, copy=False)
            bias = np.array([coef[-1]], dtype=np.float32)

            model.set_weights([weights, bias])

            # predict continuation value with this model
            continuation_value_next = model.predict(input_NN_t_ITM, verbose=0).flatten()
            # update ex post payoff value for the next NN model
            # same as before (discounted by one step) if continuation value > immediate payoff
            # equal to immediate payoff if continuation value <= immediate payoff
            continuation_values[:, i] = continuation_value_t
            continuation_values[mask, i] = np.where((early_payoff_matrix[mask, i] > continuation_value_next), early_payoff_matrix[mask, i], continuation_value_t_ITM)

            # store neural network (note still they are just a shell for the OLS solution and a linear model)
            models[i-1] = model

        # store network list as instance variable
        self.primal_models = models

    def _test_primal_linear(self):
        """
        Estimates lower bound with the trained primal models and also return the standard error of the continuation value estimate
        Note that the linear model is stored in a neural network with 0 hidden layers and a linear readout with the weight = OLS estimate
        This is used to keep the test function very similar
        """
        input_test_primal = self.input_test_primal
        X_test_primal = self.X_test_primal
        V_test_primal = self.V_test_primal
        payoff_matrix_test_primal = self.payoff_matrix_test_primal
        average_matrix_test_primal = self.average_matrix_test_primal        

        # add additional variables to the input signature array
        input_test_primal = self._add_polynomials(input_test_primal, X_test_primal, V_test_primal, self.degree_poly_primal)
        input_test_primal = self._add_state_variables(input_test_primal, X_test_primal, payoff_matrix_test_primal, average_matrix_test_primal, self.state_specs_primal)

        M, N_grid_plus_one, D = np.shape(input_test_primal)
        N_grid = N_grid_plus_one - 1

        # only input and payoffs at exercise dates needed
        TimePoints_grid = np.linspace(0,self.T, N_grid+1)
        step_size = int(N_grid / self.N_exercise)
        exercise_indices = [0] + [step_size * i for i in range(1, self.N_exercise)] + [N_grid]

        early_payoff_matrix_test = payoff_matrix_test_primal[:,exercise_indices]

        # predict continuation values per exercise date and sample using the trained linear model
        continuation_values_test = np.zeros((M,self.N_exercise+1))
        continuation_values_test[:, -1] = early_payoff_matrix_test[:, -1]
        for i in reversed(range(1,self.N_exercise)):
            if self.primal_models[i-1] is None:
                continuation_values_test[:, i] = np.inf
                continue
            input_NN_t = input_test_primal[:,exercise_indices[i],:].astype(np.float32, copy=False)
            continuation_value_t = self.primal_models[i-1].predict(input_NN_t, verbose=0).flatten()
            continuation_values_test[:,i] = continuation_value_t

        # find the first time the continuation value > immediate payoff per sample, and its associated payoffs
        stop_cond = (early_payoff_matrix_test > 0) & (early_payoff_matrix_test >= continuation_values_test)
        stop_cond[:, 0] = False
        stop_cond[:, -1] = True
        stopping_indices = np.argmax(stop_cond, axis=1)

        stopping_times = TimePoints_grid[exercise_indices][stopping_indices]
        stopping_payoffs = early_payoff_matrix_test[np.arange(M), stopping_indices]

        # discount stopping payoff to t=0
        discounted_stopping_payoff = stopping_payoffs * np.exp(-self.r*stopping_times)

        # get continuation value at t=0 estimate
        C0 = np.mean(discounted_stopping_payoff)
        # get lower bound
        lower_bound = max(early_payoff_matrix_test[0,0], C0)
        # get standard error of continuation value at t=0 estimate
        C0_std = np.std(discounted_stopping_payoff, ddof=1) / np.sqrt(M)
        return lower_bound, C0_std

    def _compute_linear_dual_weights_lp(
        self,
        input_train_dual,
        dW_train_dual,
        discounted_payoff,
        exercise_indices,
        env
    ):
        """
        Solve the associated linear program for the dual - the derivation of the program can be seen in the paper "Primal and dual optimal stopping with signatures" by Bayer, Pelizzari, and Schoenmakers (2025)
        Parameters:
        - input_train_dual ... truncated signature and additional state variables as array
        - dW_train_dual ... increments of independent Brownian drivers (2 for rough Bergomi)
        - discounted_payoff ... discounted immediate payoffs for each finer time point and sample
        - exercise_indices ... set of indices that correspond to the exercise date
        - env ... gurobipy optimizer environment
        """
        M, N_grid_plus_one, D = np.shape(input_train_dual)

        N_exercise_plus_1 = len(exercise_indices)
        L = D * self.dim_W

        exercise_indices = np.asarray(exercise_indices, dtype=np.int32)

        # dY = features * dW
        # Y_exercise are the Y values at exercise dates
        Y_exercise = np.zeros((M, N_exercise_plus_1, L), dtype=np.float32)

        # for each Brownian motion
        for w in range(self.dim_W):
            # for each variable
            for d in range(D):
                # compute the increments of dY
                l = w * D + d
                increments = (
                    input_train_dual[:, :-1, d].astype(np.float32)
                    * dW_train_dual[:, :, w].astype(np.float32)
                )

                # compute Y
                cumsum = np.cumsum(increments, axis=1)
                # restrict Y to exercise dates
                Y_exercise[:, 1:, l] = cumsum[:,exercise_indices[1:]-1]

        # define gurobipy optimizer to solve the linear program of the dual
        model = gp.Model("linear_dual_lp", env=env)
        model.Params.OutputFlag = 1

        # define new variable - see derivation of linear program
        z = model.addMVar(shape=M, lb=0.0, name="z")
        # define beta - which are the weights we are looking for
        beta = model.addMVar(shape=L, lb=-GRB.INFINITY, name="beta")

        # set the objective of the linear program (minimize average z)
        model.setObjective((1.0 / M) * z.sum(), GRB.MINIMIZE)

        # add constraints
        for j in range(N_exercise_plus_1):
            model.addConstr(
                z + Y_exercise[:, j, :] @ beta
                >= discounted_payoff[:, j],
                name=f"exercise_{j}"
            )

        # optimize for z and beta
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            # Fully solved LP
            lp_coef = beta.X.astype(np.float32, copy=False)

        else:
            status = model.Status
            model.dispose()
            raise RuntimeError(
                f"Gurobi stopped with status {status}"
            )

        # return weights for dual model
        weights = lp_coef.reshape(self.dim_W, D).T.astype(np.float32, copy=False)

        model.dispose()

        return weights, lp_coef
        
    def _train_dual_linear(self, degree_poly, state_specs, env):
        """
        Trains dual linear model by solving the corresponding linear problem with the given specifications - based on the paper "Primal and dual optimal stopping with signatures" by Bayer, Pelizzari, and Schoenmakers (2025)
        Linear version of the function _train_dual
        Parameters:
        - degree_poly ... how many Laguerre polynomials of the state variables (log) price and volatility are added for dual
        - state_specs ... further additional variables for dual as an array with elements "payoff" and/or "average"
        """
        input_train_dual = self.input_train_dual
        X_train_dual = self.X_train_dual
        V_train_dual = self.V_train_dual
        dW_train_dual = self.dW_train_dual
        payoff_matrix_train_dual = self.payoff_matrix_train_dual
        average_matrix_train_dual = self.average_matrix_train_dual

        self.degree_poly_dual = degree_poly
        self.state_specs_dual = state_specs

        # add additional variables to input signature array
        input_train_dual = self._add_polynomials(input_train_dual, X_train_dual, V_train_dual, self.degree_poly_dual)
        input_train_dual = self._add_state_variables(input_train_dual, X_train_dual, payoff_matrix_train_dual, average_matrix_train_dual, self.state_specs_dual)

        M, N_grid_plus_one, D = np.shape(input_train_dual)
        N_grid = N_grid_plus_one - 1

        TimePoints_grid = np.linspace(0, self.T, N_grid + 1)

        # get exercise indices
        step_size = int(N_grid / self.N_exercise)
        exercise_indices = (
            [0]
            + [step_size * i for i in range(1, self.N_exercise)]
            + [N_grid]
        )

        # get corresponding immediate payoffs
        payoff_matrix = payoff_matrix_train_dual[:, exercise_indices]
        # discounted payoffs needed
        discounted_payoff = (
            payoff_matrix
            * np.exp(-self.r * TimePoints_grid[exercise_indices])
        ).astype(np.float32)

        # optimize weights by solving the linear program
        weights, _ = self._compute_linear_dual_weights_lp(
            input_train_dual=input_train_dual,
            dW_train_dual=dW_train_dual,
            discounted_payoff=discounted_payoff,
            exercise_indices=exercise_indices,
            env=env
        )

        # Readout-only linear model:
        # No bias is used, because the LP above did not include it
        input_integrand = Input(shape=(D,))
        output_integrand = Dense(
            self.dim_W,
            activation="linear",
            use_bias=False
        )(input_integrand)

        model_integrand = Model(
            inputs=input_integrand,
            outputs=output_integrand
        )

        model_integrand.layers[-1].set_weights([weights])

        # rest is as before for the architecture
        input_NN = Input(shape=(N_grid, D))
        integrand_all = TimeDistributed(model_integrand)(input_NN)

        input_dW = Input(shape=(N_grid, self.dim_W))
        integrand_times_dW = Multiply()([integrand_all, input_dW])

        def _compute_objective(args):
            integrand_times_dW, payoff = args

            dM = tf.reduce_sum(integrand_times_dW, axis=-1)
            M_t = tf.cumsum(dM, axis=-1)

            m0 = tf.zeros((tf.shape(M_t)[0], 1), dtype=M_t.dtype)
            M_full = tf.concat([m0, M_t], axis=1)

            m_exercise = tf.gather(M_full, exercise_indices, axis=1)

            return tf.reduce_max(payoff - m_exercise, axis=1)

        payoff_input = Input(shape=(self.N_exercise + 1,))

        output = Lambda(_compute_objective)(
            [integrand_times_dW, payoff_input]
        )

        self.dual_model = Model(
            inputs=[input_NN, input_dW, payoff_input],
            outputs=output
        )

        # not training as we have the weights from the linear program solution

        print("Finished training linear LP dual")

    def _test_dual_linear(self):
        """
        Estimates upper bound with the trained dual linear model and also return the standard error of the estimate
        """
        input_test_dual = self.input_test_dual
        X_test_dual = self.X_test_dual
        V_test_dual = self.V_test_dual
        dW_test = self.dW_test_dual
        payoff_matrix_test_dual = self.payoff_matrix_test_dual
        average_matrix_test_dual = self.average_matrix_test_dual

        # add additional variables to input signature array
        input_test_dual = self._add_polynomials(input_test_dual, X_test_dual, V_test_dual, self.degree_poly_dual)
        input_test_dual = self._add_state_variables(input_test_dual, X_test_dual, payoff_matrix_test_dual, average_matrix_test_dual, self.state_specs_dual)

        M, N_grid_plus_one, D = np.shape(input_test_dual)
        N_grid = N_grid_plus_one - 1

        # get discounted payoffs at exercise dates
        TimePoints_grid = np.linspace(0,self.T, N_grid+1)
        step_size = int(N_grid / self.N_exercise)
        exercise_indices = [0] + [step_size * i for i in range(1, self.N_exercise)] + [N_grid]
        payoff_matrix = payoff_matrix_test_dual[:,exercise_indices]
        discounted_payoff = (payoff_matrix * np.exp(-self.r*TimePoints_grid[exercise_indices])).astype(np.float32, copy=False)

        input_NN_test = input_test_dual[:,:-1,:].astype(np.float32, copy=False)

        # predict max difference for each path
        dual_objective_test = self.dual_model.predict([input_NN_test, dW_test, discounted_payoff], verbose=0).flatten()

        # mean is the estimated upper bound
        upper_bound = np.mean(dual_objective_test)
        # and the standard error of the estimate
        Upper_bound_std = np.std(dual_objective_test) / np.sqrt(M)

        return upper_bound, Upper_bound_std