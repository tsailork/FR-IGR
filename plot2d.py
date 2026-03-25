import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Data
df = pd.read_csv("solution_2d.csv")
df = df.sort_values(by=['y', 'x'])

# Reshape for contour plotting
# Assuming structured grid order in CSV
N_pts = int(np.sqrt(len(df))) # Estimate resolution
x = df['x'].values.reshape(N_pts, N_pts)
y = df['y'].values.reshape(N_pts, N_pts)
rho = df['rho'].values.reshape(N_pts, N_pts)
sigma = df['sigma'].values.reshape(N_pts, N_pts)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Density
c1 = axes[0].contourf(x, y, rho, levels=50, cmap='viridis')
axes[0].set_title(r"Density ($\rho$)")
plt.colorbar(c1, ax=axes[0])

# Plot Entropic Pressure (Regularization)
c2 = axes[1].contourf(x, y, sigma, levels=50, cmap='plasma')
axes[1].set_title(r"Entropic Pressure ($\Sigma$)")
plt.colorbar(c2, ax=axes[1])

plt.tight_layout()
plt.show()
