#%%
import matplotlib.pyplot as plt
import numpy as np

# Coordinates for the shape
coordinates = np.array([
    [369, 528], [366, 531], [363, 531], [351, 543], [348, 543], [339, 552], 
    [333, 552], [330, 555], [327, 555], [327, 570], [324, 573], [324, 603], 
    [327, 606], [327, 621], [330, 624], [333, 624], [336, 627], [354, 627], 
    [357, 630], [363, 630], [366, 627], [369, 627], [372, 624], [375, 624], 
    [378, 621], [414, 621], [417, 618], [426, 618], [429, 615], [432, 615], 
    [435, 612], [441, 612], [444, 609], [453, 609], [456, 606], [462, 606], 
    [465, 603], [465, 600], [471, 594], [471, 591], [474, 588], [474, 549], 
    [471, 546], [471, 543], [465, 537], [465, 534], [462, 531], [459, 531], 
    [456, 528]
], dtype=np.float32)

# Plotting the shape
plt.figure(figsize=(8, 6))
plt.plot(coordinates[:, 0], -coordinates[:, 1], 'o-')  # Inverting y-axis for visual purposes
plt.title("Shape Drawn from Coordinates")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('output_shape.png')