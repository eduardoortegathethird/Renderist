# Author | Eduardo Ortega
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import glob
from PIL import Image



def convert_parametric(x, y, z):
    sw_l = [abs(s/5) for s in x]
    cw_l = [abs(s/5) for s in y]
    tvw_l = [6**s for s in z]
    return sw_l, cw_l, tvw_l

# Prepare inputs
df = pd.read_csv('parametric_xyz.csv')
z = df['z']
x = df['x']
y = df['y']
sw, cw, tvw = convert_parametric(x, y, z)

for s in range(1,len(x)):    
    ax = plt.figure().add_subplot(projection='3d')
    s_x = sw[0:s]
    s_y = cw[0:s]
    s_z = tvw[0:s]
    ax.plot(sw, cw, tvw, label='training weights curve')
    ax.scatter(s_x, s_y, s_z, label='Current weights')
    ax.set_xlabel('style weight')
    ax.set_ylabel('content weight')
    ax.set_zlabel('tot. var. weight')
    ax.legend(loc='upper center')
    plt.savefig(f'PARAM/input_{s}.png')

files = glob.glob(r"PARAM/input_*.png")
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

animation_fig.save("INPUTS.gif")
