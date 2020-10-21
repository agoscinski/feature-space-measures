Create gschnet conda environment, as described in G-Schnet repository https://github.com/atomistic-machine-learning/G-SchNet/tree/74cea9b00bf27e62f7c324e4e6f3a7b6f1f45e23
The following commands will create a new conda environment called _"gschnet"_ and install all dependencies (tested on Ubuntu 18.04):

    conda create -n gschnet python=3.7 pytorch=1.5.0 torchvision cudatoolkit=10.2 ase=3.19.0 openbabel=2.4.1 rdkit=2019.09.2.0 -c pytorch -c openbabel -c defaults -c conda-forge
    conda activate gschnet
    pip install 'schnetpack==0.3'

Replace _"cudatoolkit=10.2"_ with _"cpuonly"_ if you do not want to utilize a GPU for training/generation. However, we strongly recommend to use a GPU if available.

Download model

    wget http://www.quantum-machine.org/datasets/trained_schnet_models.zip
    unzip trained_schnet_models.zip

Run in conda environment_"gschnet"_ 

    python compute_schnet_features.py
