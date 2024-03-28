#%%
import matplotlib.pyplot as plt
import numpy as np

# Coordinates for the shape
coordinates = np.array([
   [        519,         516],
       [        507,         528],
       [        504,         528],
       [        495,         537],
       [        492,         537],
       [        483,         546],
       [        483,         561],
       [        480,         564],
       [        480,         597],
       [        483,         600],
       [        483,         603],
       [        486,         606],
       [        489,         606],
       [        492,         609],
       [        528,         609],
       [        531,         606],
       [        540,         606],
       [        543,         603],
       [        549,         603],
       [        552,         600],
       [        564,         600],
       [        567,         597],
       [        615,         597],
       [        627,         585],
       [        630,         585],
       [        633,         582],
       [        633,         579],
       [        636,         576],
       [        636,         573],
       [        642,         567],
       [        642,         555],
       [        645,         552],
       [        645,         528],
       [        642,         525],
       [        642,         522],
       [        639,         519],
       [        636,         519],
       [        633,         516]
], dtype=np.float32)

# Plotting the shape
plt.figure(figsize=(8, 6))
plt.plot(coordinates[:, 0], -coordinates[:, 1], 'o-')  # Inverting y-axis for visual purposes
plt.title("Shape Drawn from Coordinates")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('output_shape.png')