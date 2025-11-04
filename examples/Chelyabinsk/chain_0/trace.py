import matplotlib.pyplot as plt
import numpy as np

# Load samples
samples = np.loadtxt('posterior_sample.txt')

# Plot parameter traces
fig, axes = plt.subplots(5, 1, figsize=(10, 10))
param_names = ['velocity', 'angle', 'Ch', 'sigma', 'rho_d']

for i, (ax, name) in enumerate(zip(axes, param_names)):
    ax.plot(samples[:, 0])
    ax.set_ylabel(name)
    ax.set_xlabel('Sample number')

plt.tight_layout()
plt.savefig('trace_plots.png')