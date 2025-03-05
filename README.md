# TOPOLOGICAL-DICTIONARY-LEARNING

This is the official repository containing all the code to run the simulations and test the learning algorithm proposed in the paper **Topological DIctionary Learning**.

## Table of Contents

- [Abstract](#abstract)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Clone the Repository](#clone-the-repository)
  - [Setup Environment](#setup-environment)

## Abstract

This project is aimed to the development of an efficient algorithm for the joint learning of dictionary atoms, sparse representations, and the underlying topology of signals defined over cell complexes. The study then compares state-of-the-art Dictionary Learning techniques with new methodologies based on topology-informed learnable dictionaries.

This dictionary is created through the concatenation of several sub-dictionaries, where each sub-dictionary functions as a convolutional cell filter (polynomials of the Hodge Laplacians associated with the underlying topological domain), and is parameterized by distinct coefficients for upper and lower adjacencies, in accordance with what we call the Separated Hodge Laplacian parameterization.

A non-convex optimization problem is formulated for dictionary and sparse representation learning and is efficiently solved through an iterative alternating direction algorithm. The same algorithm is employed to compare the signal approximation results from the proposed parameterization, with those obtained from other topology-based parameterization techniques, as well as from analytical dictionaries based on topological Fourier transform, wavelet transform, and Slepians. The algorithm is further enhanced by incorporating a topology learning step, enabled through a either a greedy search for the optimal upper Laplacian or a relaxed iterative optimization algorithm. This novel approaches merge the representative power of overcomplete, learnable dictionaries with the efficiency of controllable optimization methods.

This framework adopts an innovative perspective in signal processing, emphasizing the utilization of geometric structures and algebraic operators for the development of algorithms with robust generalization capabilities. Additionally, it allows for future enhancements, including the incorporation of alternative topology learning techniques and the integration of model-based deep learning methods to further augment the overall algorithm.

## Project Description

The project is fully developed in Python. The repository contains two main directories:

- `topolearn`, which includes the module with essential functions and classes for implementing the proposed algorithm,
- `scripts`, where scripts used to generate the numerical results on both synthetic and real data, as reported in the related paper, can be found.

Both the real-world and synthetically generated datasets are also available for download in their respective folders within the repository.

The core class of the project is `TopoSolver`, located in the topolearn module. This class manages both the joint topology and dictionary learning procedures introduced in this work, as well as sparse representation using analytical dictionaries, such as Slepians and Hodgelets. For optimization, the project utilizes the `MOSEK` solver for semidefinite programming (SDP) problems and `GUROBI` for quadratic reformulations. Both solvers are accessed via the `CVXPY` library's API to streamline the alternating direction algorithm process.

## Project Structure

  ```bash
  TSP-DictionaryLearning/
  │
  ├── cache/          # Caching and memoizing intermediate results
  |
  ├── config/
  │   ├── algorithm.yaml       # Configuring the algorithm hyperparameters and the used methods
  │   ├── config.yaml          # Main config file for setting-up the topology, the parameters for the data generation process
  │   ├── visualization.yaml   # Config for the aesthetic characteristics of plots
  |
  ├── scripts/             # Folder containing all the experiments and simulations reported in the thesis
  │   ├── analyt_dict_learn.py # Sparse representation with analytical dictionaries
  |   ├── dict_learn.py        # Dictionary learning with parametric dictionaries
  |   ├── dict_topo_learn.py   # Joint topology and dictionary learning
  |   ├── utils.py
  |
  ├── logs/                    # Directory for the logging files coming from the experiments
  |
  ├── plots/                   # Folder for the automatic save of plots in .png format
  |
  ├── synthetic_data/          # Synthetic dataset for several generating setups
  │   ├── ...
  |
  ├── real_data/  
  │   ├── real_data.mat        # DFN dataset for experiments on real topological signals      
  |
  ├── topolearn/               # Package directory
  │   ├── __init__.py          # Initializes the package
  │   ├── data_generation.py   # Functions for generation of synthetic datasets of topological signals
  │   ├── EnancedGraph.py      # Class for generating 2nd-order Cell Complexes
  │   ├── Hodgelets.py         # Classes for the implementation of Hodgelets and Slepians
  |   ├── TopoSolver.py        # Main class containing the procedure for joint topology and dictionary learning
  |   ├── utils.py             # Utils functions for memoization and saving plots and results
  |   ├── utilsHodgelets.py    # Auxiliary functions for Hodgelets and Slepians
  |   ├── utilsTopoSolver.py   # Auxiliary functions for TopoSolver class
  │   └── visualization.py     # Module for plots and results visualization
  │
  |
  ├── README.md              # Project overview and usage instructions
  └── pyproject.toml       # Dependencies and environment management
  ```

## Usage

### Clone the Repository

```bash
git clone https://github.com/SPAICOM/topological-dictionary-learning
cd topological-dictionary-learning
```

### Setup Environment

We suggest to use [uv](https://docs.astral.sh/uv/) as package and environment manager to quickly solve any dependency issues. For instance, if you want to run any general script `script.py` contained in the `scripts` directory, simply run:

```bash
uv run scripts\script.py
```
