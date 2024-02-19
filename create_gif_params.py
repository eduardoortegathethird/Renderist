# Author | Eduardo Ortega
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import glob
from PIL import Image

# Prepare arrays x, y, z
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
for s in range(1,len(x)):    
    ax = plt.figure().add_subplot(projection='3d')
    s_x = x[0:s]
    s_y = y[0:s]
    s_z = z[0:s]
    ax.plot(x, y, z, label='parametric curve')
    ax.scatter(s_x, s_y, s_z, label='Variable Interpolation')
    ax.legend()
    plt.savefig(f'PARAM/fig_{s}.png')

files = glob.glob(r"PARAM/*.png")
images = []
for my_file in files:
    i = Image.open(my_file)
    images.append(i)
# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the initial image
im = ax.imshow(images[0], animated=True)

def update(i):
    im.set_array(images[i])
    return im, 

# Create the animation object
animation_fig = animation.FuncAnimation(fig, update, frames=len(images), interval=100, blit=True,repeat_delay=10,)

# Show the animation
#plt.show()

animation_fig.save("PARAMS.gif")
