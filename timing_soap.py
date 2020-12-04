import numpy as np
from timeit import default_timer as timer

from rascal.representations import SphericalInvariants

from src.experiment import read_dataset
import functools

frames = read_dataset("C-VII-pp-wrapped.xyz", "", set_methane_dataset_to_same_species=False)
print(len(frames))

cutoff = 4
sigma = 0.5
cutoff_smooth_width = 0.5
normalize = False
max_radials_angulars = [(2, 2), (4, 3), (6, 4), (8, 5)]

DEFAULT_ITERATIONS = 10

def bench(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        nb_iterations = kwargs.get("nb_iterations", DEFAULT_ITERATIONS)

        total_times = np.zeros(nb_iterations)
        for i in range(nb_iterations):
            start_time = timer()
            value = func(*args, **kwargs)
            end_time = timer()
            total_times[i] += end_time - start_time

        average_time = np.mean(total_times)
        std_time = np.std(total_times)
        print(f"Finished in {average_time:.2f} s with std {std_time:.2f} over {nb_iterations} runs")
        return value
    return wrapper_timer

@bench
def compute_soap(hypers):
    representation = SphericalInvariants(**hypers)
    representation.transform(frames).get_features(representation)
    return


print("SOAP GTO timings")
gto_hypers = [{
        "soap_type": "PowerSpectrum",
        "radial_basis": "GTO",
        "interaction_cutoff": cutoff,
        "max_radial": max_radial,
        "max_angular": max_angular,
        "gaussian_sigma_constant": sigma,
        "gaussian_sigma_type": "Constant",
        "cutoff_smooth_width": cutoff_smooth_width,
        "normalize": normalize
    } for max_radial, max_angular in max_radials_angulars]
print([compute_soap(hyper) for hyper in gto_hypers])



print("SOAP DVR timings")
dvr_hypers = [{
        "soap_type": "PowerSpectrum",
        "radial_basis": "DVR",
        "interaction_cutoff": cutoff,
        "max_radial": max_radial,
        "max_angular": max_angular,
        "gaussian_sigma_constant": sigma,
        "gaussian_sigma_type": "Constant",
        "cutoff_smooth_width": cutoff_smooth_width,
        "normalize": normalize
    } for max_radial, max_angular in max_radials_angulars]
print([compute_soap(hyper) for hyper in dvr_hypers])
