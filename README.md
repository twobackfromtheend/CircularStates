### Hamiltonians

Generate matrices (in the *n*, *l*, *j*, *mj* basis) with `system/hamiltonians/hamiltonians.py`.
Running the file will generate the `mat_1`, `mat_2`, `mat_2_minus`, and `mat_2_plus` matrices and save them into
the `system/hamiltonians/generated_hamiltonians` folder.
These matrices are saved as a single (combined) npz file.

The `load_hamiltonian()` function provides an interface for loading the matrices from the npz file.


### Transformations
The information required for basis transformation is generated with 
`system/transformations/transformation_generator.py`.

This is done with `nljmj_to_nlmlms()` and `nlmlms_to_n1n2mlms()`.
No other transformations are supported at this time.


### States and Basis
`system.states.States` enable the generation of basis states in the following basis:
- `Basis.N_L_J_MJ` 
- `Basis.N_L_ML_MS` 
- `Basis.N1_N2_ML_MS` 


### Usage

#### Setup
1. Generate Hamiltonian matrices with `system/hamiltonians/hamiltonians.py`
2. Generate basis transformation matrices with `system/transformations/transformation_generator.py`

#### Functionality
Energy against *ml* is plotted with `plots/ladder.py`. This plot is interactive, and clicking on the figure prints 
information about the state clicked on.

Eigenvalues across an ARP protocol is plotted with `plots/eigenvalues_across_protocol.py`.




### Requirements
Because `pathlib` and `typing` is used, this requires Python >=3.5.
A higher version of Python may be required due to changes in the library interfaces.

Development was done with Python 3.8.3.


The following command should include all necessary packages: 

`pip install numpy scipy matplotlib ARC-Alkali-Rydberg-Calculator qutip tqdm`

 
