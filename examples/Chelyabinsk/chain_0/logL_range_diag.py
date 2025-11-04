import numpy as np
import matplotlib.pyplot as plt

# Load samples
samples = np.loadtxt('posterior_sample.txt')
params = samples[:, :-2]  # Remove logL and weight
logL = samples[:, -2]

print(f"Total samples: {len(samples)}")
print(f"\n{'='*60}")

# Check parameter exploration
param_names = ['velocity', 'angle', 'Ch', 'sigma', 'rho_d']
for i, name in enumerate(param_names):
    param_range = np.ptp(params[:, i])
    param_std = np.std(params[:, i])
    
    print(f"{name}:")
    print(f"  Min: {np.min(params[:, i]):.4e}")
    print(f"  Max: {np.max(params[:, i]):.4e}")
    print(f"  Range: {param_range:.4e}")
    print(f"  Std: {param_std:.4e}")
    print(f"  Mean: {np.mean(params[:, i]):.4e}")
    print()

# Check log-likelihood
print(f"{'='*60}")
print("Log-likelihood statistics:")
print(f"  Min: {np.min(logL):.2e}")
print(f"  Max: {np.max(logL):.2e}")
print(f"  Range: {np.ptp(logL):.2e}")
print(f"  Unique values: {len(np.unique(np.round(logL, 2)))}")

# Check if all logL are basically identical
if np.ptp(logL) < 100:
    print("\n⚠️  WARNING: Log-likelihoods are nearly identical!")
    print("   This means parameters aren't affecting the fit.")

# Plot
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Parameter traces
for i, (ax, name) in enumerate(zip(axes.flat[:5], param_names)):
    ax.plot(params[:, i])
    ax.set_ylabel(name)
    ax.set_xlabel('Sample')
    ax.grid(True, alpha=0.3)

# Log-likelihood trace
axes.flat[5].plot(logL)
axes.flat[5].set_ylabel('Log-likelihood')
axes.flat[5].set_xlabel('Sample')
axes.flat[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics_full.png', dpi=150)
print("\nSaved diagnostics_full.png")