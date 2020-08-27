import ase.io
import numpy as np
import copy

# Creates a methane dataset by dragging the first hydrogen atom away from the carbon atom
methane = ase.io.read("methane.xyz")
frames = [methane]
cutoff = 4.5
first_distance = 0.5
for step_size in [0.01, 0.05]:
    frames[0].info = {"hydrogen_distance": first_distance}
    frames[0][1].position = [0, 0, first_distance]
    nb_samples = int((cutoff-first_distance)/step_size)
    print(nb_samples)
    for i in range(nb_samples):
        new_frame = copy.deepcopy(frames[-1])
        new_frame[1].position += [0,0,step_size]
        new_frame.info = {"hydrogen_distance": round(new_frame[1].position[2],4)}
        frames.append(new_frame)
    seed = None #0x5f3759df
    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(frames)
    ase.io.write("displaced-methane-step_size="+str(step_size)+"-range=["+str(first_distance)+","+str(cutoff)+"]-seed="+str(seed)+".extxyz", frames)
