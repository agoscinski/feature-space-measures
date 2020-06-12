import ase.io
import copy

# Creates a methane dataset by dragging the first hydrogen atom away from the carbon atom

methane = ase.io.read("methane.xyz")
frames = [methane]
nb_samples = 10000
cutoff = 6
step_size = (cutoff-1)/nb_samples
for i in range(nb_samples):
    new_frame = copy.deepcopy(frames[-1])
    new_frame[1].position += [0,0,step_size]
    frames.append(new_frame)

ase.io.write("dragged-methane.xyz", frames)
