import numpy as np
from parameters import *

# constants

# gravitational acceleration
g = 9.81
# mass
mass = 3
# side of image
image_side = 512



# derived constants

# maximum initial speed of jump
max_initial_speed = np.sqrt(2 * g * max_jump_height)
# path to image for map
path_to_image = f"../images/{image_name}.png"