Clone the diment fork which adds easier output of representation, create virtual environment and run the compute file

    git clone https://github.com/agoscinski/dimenet
    conda create -n dimenet python=3.7 scipy sympy tensorflow
    conda activate dimenet
    pip install tensorflow_addons pyyaml
    python compute_dimenet_features.py

