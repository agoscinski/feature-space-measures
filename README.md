Code for calculation of the data for the paper "The role of feature space in atomistic learning" currently in the review process.
Please refer to the `run.sh` script to see what `*-experiments.py` file corresponds to what figure of the paper.

Datasets (in `/data`):
 - `selection-10k.extxyz` *random methane* 

   TODO
 - `C-VII-pp-wrapped.xyz` *carbon* to obtain the dataset please refer to 

   Chris J. Pickard, AIRSS data for carbon at 10GPa and the C+N+H+O system at 1GPa, Materials Cloud Archive **2020.0026/v1** (2020), doi: 10.24435/materialscloud:2020.0026/v1.
 - `manif-minus-plus.extxyz` *degenerate methane*

   S. N. Pozdnyakov, M. J. Willatt, A. P. Bart´ok, C. Ortner, G. Cs´anyi, and M. Ceriotti, arXiv preprint arXiv:2001.11696 (2020)
 - `displaced-methane-step_size=PLACE_HOLDER-range=PLACE_HOLDER-seed=PLACE_HOLDER.extxyz` *displaced methane* dataset generator file `/data/generate_displaced_methane_dataset.py`
