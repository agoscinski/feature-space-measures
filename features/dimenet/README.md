Clone the diment fork which adds easier output of representation, create virtual environment and run the compute file

    git clone https://github.com/agoscinski/dimenet
    conda create -n dimenet python=3.7 scipy=1.3 sympy=1.5 tensorflow=2.1 tensorflow_addons tqdm
    conda activate dimenet
    python compute_dimenet_features.py

