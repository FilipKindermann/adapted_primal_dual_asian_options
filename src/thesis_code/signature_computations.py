######################################################################################
######################################################################################
#
# Description: computes signatures for the deep_pricer.py
#
######################################################################################
######################################################################################

# Supports three signature lifts:
#  - "baseline":   Z_t = (t, X_t)
#  - "payoff":     Z_t = (t, X_t, (K - 1/t int_0^t S_s ds)^+)
#  - "volatility": Z_t = (t, vol_t)  [+ X_t appended as state variable]
#  - "qv_volatility": Z_t = ([S]_t, vol_t)  [+ X_t appended as state variable]
# Note: X can be log price or level price, S is always the level price

# Supports three signature specs:
#  - "standard signature": Truncated signature of Z_t
#  - "log signature":      Log signature of Z_t
#  - "basis words signature": Entries of the truncated signature corresponding to Lyndon words

# Note that the Lyndon word signature components representation was named "basis words signature", although this name refers to a basis in the polynomial relationship sense

import numpy as np
import iisignature as ii
import time
from itertools import product

################################################################################################
# Lyndon-word helpers
################################################################################################
def _is_lyndon(word):
    """
    A word is Lyndon if it is strictly lexicographically smaller than
    all of its proper suffixes.
    """
    n = len(word)
    for i in range(1, n):
        suffix = word[i:]
        if suffix <= word: # proper suffix must be strictly greater
            return False
    return True
 
 
def _lyndon_words(alphabet_size: int, max_degree: int):
    """
    Generate all words over {0, ..., alphabet_size-1} up to length max_degree and keeps only those that are Lyndon words
    Parameters:
    - alphabet_size ... number of letters are in the alphabet
    - max_degree ... maximum word length
    """
    words = []
    # for each possible length
    for n in range(1, max_degree + 1):
        # get all words of this length
        for word in product(range(alphabet_size), repeat=n):
            # check if it is a Lyndon word
            if _is_lyndon(word):
                # if yes, store it
                words.append(word)
    return words
 
def _sig_index_of_word(word, alphabet_size):
    """
    Return the index of a word within the iisignature ordering (excluding 0 -> smallest letter has index 0), which is first by length, second by lexicographic order
    Parameters:
    - word ... word used to get a signature component
    - alphabet_size ... number of letters in the alphabet
    """
    # get word length
    level = len(word)
    # get number of words with a smaller length
    offset = sum(alphabet_size ** k for k in range(1, level))
    # get number of words that are smaller than this word with the same length
    pos = 0
    for ch in word:
        pos = pos * alphabet_size + ch
    # both together is the index (excluding empty word)
    return offset + pos

def _lyndon_indices(alphabet_size: int, max_degree: int):
    """
    Return the indices corresponding to all Lyndon words up to length max_degree
    Parameters:
    - alphabet_size ... number of letters are in the alphabet
    - max_degree ... maximum word length
    """
    lw = _lyndon_words(alphabet_size, max_degree)
    return sorted(_sig_index_of_word(w, alphabet_size) for w in lw)


################################################################################################
# Class to compute the signature 
################################################################################################

class SignatureComputer:

    def __init__(self, T, N, K_sig, signature_spec, signature_lift, strike=None):
        """
        Initialize parameters relevant for pricing the instrument
        Parameters:
        - T ... maturity
        - N ... number of time steps
        - K_sig ... truncation level of the truncated signature
        - signature_spec ... representation, takes elements in ["standard signature", "log signature", "basis words signature"]
        - signature_lift ... input process, takes elements in ["baseline", "payoff", "volatility", "qv_volatility"]
        - strike ... strike price
        """
        self.T = T
        self.N = N
        self.K_sig = K_sig
        self.strike = strike
        self.signature_spec = signature_spec
        self.signature_lift = signature_lift
        self.tt = np.linspace(0, T, N + 1)

    # function called to get the signature after initializing the class
    def compute_signature(self, X, vol, log_price):
        """
        Compute signature for the full sample and until all time steps
        Parameters:
        - X ... (log) price
        - vol ... volatility
        """
        print(
            f"Computing '{self.signature_spec}' with '{self.signature_lift}' lift "
            f"(K_sig={self.K_sig}, strike={self.strike})"
        )
        # get input process
        Z = self._build_lifted_path(X, vol, log_price)
        # get signature in the required representation
        Sig = self._compute_spec(Z, X)
        return Sig

    def _compute_running_integral(self, X, log_price):
        """
        Compute the integral I_t = int_0^t X_s ds for each path using the trapezoidal rule 
        Used for the payoff input process and QV process in _build_lifted_path
        Parameters:
        - X ... log price or level price 
        - log_price ... says whether to use log prices (True) or level prices (False)
        """
        # get level price
        S = X.copy()
        if log_price:
            S = np.exp(S)

        dt = self.T / self.N
        # Trapezoidal increments are 0.5 * dt * (S[:, k] + S[:, k+1])
        increments = 0.5 * dt * (S[:, :-1] + S[:, 1:])
        I = np.zeros_like(S)
        # get integral by cumsum
        I[:, 1:] = np.cumsum(increments, axis=1)
        return I

    def _build_lifted_path(self, X, vol, log_price):
        """
        Construct the input path, depending on the selected signature_lift
        - "baseline" is Z_t = (t, X_t)
        - "payoff" is Z_t = (t, X_t, (K - 1/t int_0^t X_s ds)^+)
        - "volatility" is Z_t = (t, vol_t)
        - "qv_volatility" is Z_t = ([S]_t, vol_t)
        """
        M = X.shape[0]
        # repeat time points
        t = np.tile(self.tt, (M, 1))

        # input process of baseline (time-augmented (log) price)
        if self.signature_lift == "baseline":
            # add to one process
            Z = np.stack([t, X], axis=-1)

        # input process of payoff (time-augmented (log) price and state payoff)
        elif self.signature_lift == "payoff":
            # get integral estimate
            RunningIntegral = self._compute_running_integral(X, log_price)
            RunningAverage = np.empty_like(RunningIntegral)
            RunningAverage[:, 0] = np.exp(X[:, 0]) if log_price else X[:, 0]
            RunningAverage[:, 1:] = RunningIntegral[:, 1:] / t[:, 1:]
            # get payoff
            running_payoff = np.maximum(self.strike - RunningAverage, 0.0)
            # add to one process
            Z = np.stack([t, X, running_payoff], axis=-1)

        # input process of volatility (time-augmented volatility)
        elif self.signature_lift == "volatility":
            # add all to one process
            Z = np.stack([t, vol], axis=-1)

        # input process of qv_volatility (QV-augmented volatility)
        elif self.signature_lift == "qv_volatility":
            # [S]_t = int_0^t v_s^2 ds
            qv = self._compute_running_integral(vol**2, False)
            Z = np.stack([qv, vol], axis=-1)

        else:
            raise ValueError(f"Wrong input process signature_lift='{self.signature_lift}', select one from 'baseline', 'payoff', 'volatility', 'qv_volatility'.")

        return Z

    def _compute_spec(self, Z, X):
        """
        Compute the truncated signature of the input process Z in the requested representation
        For "volatility" and "qv_volatility", X_t is added as a state variable
        Parameters:
        - Z ... input process
        - X ... (log) price
        """
        # if standard signature is asked
        if self.signature_spec == "standard signature":
            start = time.time()
            # compute standard signature
            sig = self._full_signature(Z)
            runtime = time.time()-start
            print(f"Runtime standard signature: {runtime} sec")

        # if log signature is asked
        elif self.signature_spec == "log signature":
            start = time.time()
            # compute log signature
            sig = self._full_log_signature(Z)
            runtime = time.time()-start
            print(f"Runtime standard signature: {runtime} sec")

        # if Lyndon word signature is asked
        elif self.signature_spec == "basis words signature":
            start = time.time()
            # get Lyndon word signature components
            sig = self._basis_words_signature(Z)
            runtime = time.time()-start
            print(f"Runtime standard signature: {runtime} sec")

        else:
            raise ValueError(f"Wrong signature representation signature_spec='{self.signature_spec}', select one from 'standard signature', 'log signature', 'basis words signature'.")

        # add X_t as an additional state variable for the volatility and qv_volatility lift
        if self.signature_lift in ["volatility", "qv_volatility"]:
            sig = np.concatenate([sig, X[:, :, np.newaxis]], axis=-1)

        return sig

    def _full_signature(self, Z):
        """
        Truncated standard signature of Z at every time point
        Parameter:
        - Z ... input process path
        """
        M, Np1, d = Z.shape
        # get number of standard signature components up to truncation level K_sig
        k = ii.siglength(d, self.K_sig)
        sig = np.zeros((M, Np1, k + 1))
        sig[:, :, 0] = 1.0
        for m in range(M):
            # get signatures for each sample - 2 returns for each time step 
            sig[m, 1:, 1:] = ii.sig(Z[m], self.K_sig, 2)
        return sig

    def _full_log_signature(self, Z):
        """
        Truncated log signature of Z at every time point
        Parameter:
        - Z ... input process path
        """
        M, N_plus_1, d = Z.shape
        # log needs prepare first
        # C says use Baker–Campbell–Hausdorff (BCH) formula
        bch = ii.prepare(d, self.K_sig, 'C')
        log_sig = np.zeros((M, N_plus_1, ii.logsiglength(d, self.K_sig)))
        # for each sample
        for m in range(M):
            # for each time step besides t=0
            for i in range(1, N_plus_1):
                # get log signature
                log_sig[m, i] = ii.logsig(Z[m, :i + 1], bch, 'C')
        return log_sig

    def _basis_words_signature(self, Z):
        """
        Entries of the truncated signature corresponding to Lyndon words
        Parameter:
        - Z ... input process path
        """
        M, Np1, d = Z.shape
        # get indices of Lyndon word signature components 
        # but they skip the empty word
        lyndon_idx = _lyndon_indices(d, self.K_sig)

        # Compute the standard signature
        full_sig = self._full_signature(Z)
        # plus 1 because of the empty word
        lyndon_cols = [idx + 1 for idx in lyndon_idx]

        # get 1 for empty word + Lyndon word signature components
        basis_sig = np.concatenate(
            [full_sig[:, :, :1],
             full_sig[:, :, lyndon_cols]],
            axis=-1
        )
        return basis_sig