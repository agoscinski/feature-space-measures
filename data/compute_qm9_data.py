from schnetpack.datasets import QM9
import numpy as np
qm9data = QM9('./qm9.db', download=True, remove_uncharacterized=True)
property_keys = ["dipole_moment" , "isotropic_polarizability" , "homo" , "lumo" , "electronic_spatial_extent" , "zpve" , "energy_U0" , "energy_U" , "enthalpy_H" , "free_energy" , "heat_capacity"]
nb_structures = 10000
properties = {key: np.zeros(nb_structures) for key in property_keys}
for i in range(nb_structures):
    _, struc_property = qm9data.get_properties(idx=i)
    for key in property_keys:
        properties[key][i] = float(struc_property[key][0])

for key in property_keys:
    np.save("qm9_"+key+".npy", properties[key])
