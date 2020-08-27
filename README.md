Code for calculation of the data for the paper "The role of feature space in atomistic learning" currently in the review process.
Please refer to the `run.sh` script to see what `*-experiments.py` file corresponds to what figure of the paper.

Datasets (if available in `/data`):
 - `selection-10k.extxyz` *random methane* 

   TODO
 - `C-VII-pp-wrapped.xyz` *carbon* to obtain the dataset please refer to 

   Chris J. Pickard, AIRSS data for carbon at 10GPa and the C+N+H+O system at 1GPa, Materials Cloud Archive **2020.0026/v1** (2020), doi: 10.24435/materialscloud:2020.0026/v1.
 - `manif-minus-plus.extxyz` *degenerate methane*

   S. N. Pozdnyakov, M. J. Willatt, A. P. Bart´ok, C. Ortner, G. Cs´anyi, and M. Ceriotti, arXiv preprint arXiv:2001.11696 (2020)

 - `displaced-methane-step_size=STEP_SIZE-range=[0.5,4.5]-seed=None.extxyz` *displaced methane* for `STEP_SIZE=0.05` and `STEP_SIZE=0.01` dataset generator file `/data/generate_displaced_methane_dataset.py`

Features (if precomputed in `/features`):
 - SOAP features are computed with librascal https://github.com/cosmo-epfl/librascal
 - BPSF were precomputed with RuNNer https://www.uni-goettingen.de/de/560580.html generation input file code snippets are available in `/features/BPSF`
 - NICE features were precomputed with TODO
