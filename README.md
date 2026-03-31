# HOTA: Hamiltonian Optimal Transport Advection

*This repository provides HOTA implementation used for experiments in the papers.*

## Code Structure

**Key Components:**

1. `src/method/hota.py`: Implementation of the HOTA method (Section 4)
2. `run`: Interactive demonstrations of HOTA applied to datasets from the paper (Section 5)
	1. `two_dimensional_benchmark.ipynb` (Section 5.2)
	2. `sphere_dataset_3_dim.ipynb` (Section 5.3)
	
3. `figures`: Some experiments visualization logs
4. `requirements.txt`: File with Python dependencies

```
├── README.md              # Repository documentation
├── requirements.txt       # Python dependencies
├── figures/               # Visualization produced notebooks from experiments directory
├── run/                   # Jupyter notebooks for paper experiments
│   ├── two_dimensional_benchmark.ipynb    # Section 5.2
│   ├── sphere_dataset_3_dim.ipynb         # Section 5.3 (3D case)
└── src/
    ├── datasets/
    │   ├── datasets.py              # Datasets w/o Potential barriers
    │   └── lagrangian_potentials.py # Potential barriers
    ├── method/
    │   ├── dynamics.py    # Flow definition
    │   ├── hota.py        # Main HOTA algorithm
    ├── networks/
    │   └── mlp.py         # Time-augmented MLPs
    └── utils/
        └── metrics.py     # Sinkhorn loss
```

## Environment Setup

1. Conda Environment Setup

	```zsh
	# Create and activate conda environment
	conda create -n HOTA python=3.11
	conda activate HOTA	
	
	# Install dependencies
	pip install -r requirements.txt
	```
	
2. Environment Configuration

	```zsh
	# Base project directory
	export PROJECT_DIR=/absolute/path/to/HOTA
	
	# Python path configuration
	export SRC_DIR=${PROJECT_DIR}/src
	export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"
	```
	
## Demonstration

The notebooks below allow you to run HOTA on datasets from the paper and visualize results. 

| Notebook                          | Paper Section | Comment                                                                 |
|-----------------------------------|---------------|-------------------------------------------------------------------------|
| [two\_dimensional\_benchmark.ipynb](run/two_dimensional_benchmark.ipynb) | 5.2           | Demonstrates HOTA on 2D benchmark tasks (e.g., Vneck, GMM)       |
| [sphere\_dataset\_3\_dim.ipynb](run/sphere_dataset_3_dim.ipynb)      | 5.3           | Demonstrates HOTA on 3D Sphere dataset                                  |
