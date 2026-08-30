# Code implementation of the master's thesis "Pricing American-style Asian options under rough volatility using path signatures and deep neural networks"

Code accompanying the thesis:

**"Pricing American-style Asian options under rough volatility using path signatures and deep neural networks"**
Filip Kindermann
University: WU Wien (Vienna University of Economics and Business)
Degree programme: Quantitative Finance
Supervisor: Assist.Prof. Priv.Doz.Dr. Paul Eisenberg
Year: 2026

## Overview

This repository contains the code for the master's thesis "Pricing American-style Asian options under rough volatility using path signatures and deep neural networks" by Filip Kindermann. The thesis investigates the problem of pricing American-style Asian put options in a rough volatility market, in particular rough Bergomi. The option is priced for several strike prices using the adapted primal-dual framework using truncated path signatures and neural networks, proposed by [Bayer, Pelizzari, and Zhu (2025, preprint)](https://doi.org/10.48550/arXiv.2501.06758). The relative duality gap (i.e., the difference between the upper and lower bound estimates relative to the upper bound estimate) is used to evaluate the pricing performance. Furthermore, parameters in the baseline configuration are varied to provide sensitivity tests for several rough Bergomi parameters, the number of discretization steps, and design choices of the input and neural networks. Additionally, a variable importance analysis is performed by training the networks and subsequently shuffling a pricing input variable across samples for all time steps, including truncated signature components and additional state variables. These inputs are used for pricing, and the relative difference to the standard bound estimate represents an importance measure. Moreover, the neural network implementation is compared to linear implementations of the adapted primal-dual framework. Finally, as a validation of the approach, the adapted primal-dual framework using truncated path signatures and neural networks is applied to Black-Scholes paths and compared to the results from [Rasmussen (2005)](https://doi.org/10.21314/JCF.2005.128). 

This repository shows the Python implementation of the simulation study and tests run for this thesis.

## Repository Structure

```text
.
├── src/
│   └── thesis_code/      # Main Python implementation
├── scripts/              # Executable experiment scripts
├── output/               # Aggregation scripts to combine several experiment runs
├── notebooks/            # Optional exploratory notebooks
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Locked dependency versions
├── .python-version       # Reference Python version
├── CITATION.md           # Citation metadata
├── LICENSE               # MIT license
└── README.md
```

The /src/thesis_code folder includes the implementation of the adapted primal-dual framework, including the Pricer class, the models to generate paths, and the SignatureComputer class to construct the signatures. 

The /scripts folder includes all scripts that are (repeatedly) executed to generate the results of the thesis. Subsequently, the modules in /output aggregate the results to compute average bound estimates, standard errors, and empirical standard deviations.

The /notebooks folder includes an illustrative notebook price_American_style_asian_options.ipynb that explains how a simulation can be run. The notebook run_script_colab.ipynb is used to run the experiment scripts of the /scripts folder in a Google Colab environment.

## Environment

The reference environment used for the thesis is:

* Python **3.12.10**
* TensorFlow **2.20.0**
* Keras **3.13.2**
* iisignature **GitHub version**
* gurobipy **13.0.2**

The complete dependency set is defined in `pyproject.toml` and locked in `uv.lock`.

Note: 
- iisignature is used to compute the truncated signature and truncated log signature efficiently. 
- gurobipy is used for solving the linear program in the linear implementation and requires a license free of charge for academic purposes. If you do not run the linear implementation, there is no license needed, but you still need to install the library, as it is an import in the deep_pricer.py module.

## Methodological and Implementation Sources:

- For the simulation of the rBergomi model, the hybrid simulation scheme of [Bennedsen, Lunde, and Pakkanen (2017)](https://doi.org/10.1007/s00780-017-0335-5) is used, with the implementation of [McCrickerd and Pakkanen](https://github.com/ryanmccrickerd/rough_bergomi).

- The primal and dual algorithms using a neural network implementation follow the concepts of [Bayer, Pelizzari, and Zhu (2025 preprint)](https://doi.org/10.48550/arXiv.2501.06758), and the linear implementation follows [Bayer, Pelizzari, and Schoenmakers (2025)](https://doi.org/10.1007/s00780-025-00570-8).

## Installation

### Using uv

Clone the repository:

```bash
git clone https://github.com/FilipKindermann/adapted_primal_dual_asian_options.git
cd adapted_primal_dual_asian_options
```

Install the environment:

```bash
uv sync
```

Run Python commands inside the environment with:

```bash
uv run python ...
```

Alternatively, activate the virtual environment directly.

On Windows:

```powershell
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

The environment is tested on Windows 11 and Google Colab. No tests have been performed for a local Linux installation.

### Using pip

The project can also be installed with pip:

```bash
pip install .
```

For an editable development installation:

```bash
pip install -e .
```

## Quick Start

For a quick start on how the pricing algorithm can be run, refer to the notebook price_American_style_asian_options.ipynb, which demonstrates the training and pricing for a given configuration and one strike price. 

Subsequently, the run scripts in /scripts show how to run the simulation study for multiple strike prices and varying parameters.

## Configuration

The simulation study has a baseline configuration that is used to evaluate the pricing performance. The sensitivity tests tend to override one parameter within this baseline configuration and leave the others constant. 

The baseline configuration can be found under 

```text
src/
  └── thesis_code/
      └── configs/
          └── baseline_config.py
```

It corresponds to an option with maturity $T=1$, $N=12$ early exercise dates, and $J=48$ finer time steps. The rough Bergomi parameters are $r=0.05, \eta=1.9, \xi = 0.09, \rho = -0.9, H=0.07, \tilde S_0=1$. It uses $2^{18}$ paths for training and new paths for pricing of the primal and the dual algorithm, respectively. The signature representation is equal to the Lyndon word signature components, and the input process is the QV-augmented volatility process for the primal and the time-augmented volatility process for the dual with the log price as an additional state variable for both. The truncation level is set to $3$, and the strike prices are $\{0.70,0.80,0.90,1.00,1.10,1.20\}$. Furthermore, the primal and dual neural network architecture is specified there.

## Notes on Reproducibility

The experiments reported in the thesis were run using **Python 3.12.10**.

Dependency versions are recorded in `uv.lock`.

Numerical results may vary slightly depending on:

* hardware,
* operating system,
* TensorFlow backend,
* random seeds,
* parallel execution,
* and floating-point behavior.