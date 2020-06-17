import ase.io
import copy

# Creates a methane dataset by dragging the first hydrogen atom away from the carbon atom

methane = ase.io.read("methane.xyz")
frames = [methane]
nb_samples = 1000
cutoff = 4
step_size = (cutoff-1)/nb_samples
frames[0].info = {"hydrogen_distance": round(frames[0][1].position[2],4)}
for i in range(nb_samples):
    new_frame = copy.deepcopy(frames[-1])
    new_frame[1].position += [0,0,step_size]
    new_frame.info = {"hydrogen_distance": round(new_frame[1].position[2],4)}
    frames.append(new_frame)

ase.io.write("dragged-methane.extxyz", frames)
