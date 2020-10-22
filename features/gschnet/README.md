Clone and create gschnet conda environment, as described in G-Schnet repository:
The following commands will create a new conda environment called _"gschnet"_ and install all dependencies (tested on Ubuntu 18.04):

    git clone https://github.com/atomistic-machine-learning/G-SchNet.git
    cd G-SchNet
    git checkout 74cea9b00b
    cd ..
    conda create -n gschnet python=3.7 pytorch=1.5.0 torchvision cudatoolkit=10.2 ase=3.19.0 openbabel=2.4.1 rdkit=2019.09.2.0 -c pytorch -c openbabel -c defaults -c conda-forge
    conda activate gschnet
    pip install 'schnetpack==0.3'

Replace _"cudatoolkit=10.2"_ with _"cpuonly"_ if you do not want to utilize a GPU for training/generation. However, we strongly recommend to use a GPU if available.

Download model

    wget http://www.quantum-machine.org/data/trained_gschnet_model.zip
    unzip trained_gschnet_model.zip

Run in conda environment_"gschnet"_ 

    python compute_gschnet_features.py
