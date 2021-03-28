from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
import numpy as np
from PIL import Image
import sys

# constants

# resolution <= 512
board_size = 40
# maximum height of points
max_height = 20
# size of dots
dot_size = 4
# gravitational acceleration
g = 9.81
# mass
mass = 3
# range for jump heights
max_jump_height = 30
# maximum initial speed
max_initial_speed = np.sqrt(2 * g * max_jump_height)
# adding speed to make jumps lower
min_speed_boost, max_speed_boost = 1, 2
# show dots
are_visible_dots = False
# width of lines on path
line_width = 1
# for picture dots
dot_opaqueness = 1.
number_of_random_walks = 40
length_of_random_walk = 100
# resolution of jump
points_in_jump = 20
# side of image
image_side = 512

app = QApplication([])
w = gl.GLViewWidget()
w.show()

# g = gl.GLGridItem()
# g.setSize(x=board_size, y=board_size,z=max_height)
# w.addItem(g)

ax = gl.GLAxisItem()
ax.setSize(x=board_size, y=board_size, z=max_height)
w.addItem(ax)


def read_image(path):
    image = Image.open(path)
    image = np.asarray(image).astype(np.float32)
    assert image.shape == tuple([image_side, image_side, 4])
    return image


def get_compressed_image(path):
    image = read_image(path)
    factor = np.ceil(image_side / board_size).astype(int)
    compressed_image = image[::factor][:, ::factor]
    return compressed_image


def get_coordinate_grid():
    x_grid, y_grid = np.meshgrid(np.arange(board_size), np.arange(board_size), indexing="ij")
    return x_grid.ravel(), y_grid.ravel()


def get_rgba_to_height(image):
    def rgba_height(p):
        return p[0] ** 0.2 + p[1] ** 0.5 + p[2] ** 0.6

    height = np.amax(np.apply_along_axis(rgba_height, 1, image))
    return lambda rgba: rgba_height(rgba) / height * max_height


def get_platform():
    image = get_compressed_image('images/deadpool.png')
    x_grid, y_grid = get_coordinate_grid()
    flat_image = np.reshape(image, (image.shape[0] ** 2, 4))
    rgb_to_height = get_rgba_to_height(flat_image)
    z_grid = np.apply_along_axis(rgb_to_height, 1, flat_image)

    # need values in [0;1]
    normalized_image_colors = (flat_image / 255.).astype(np.float32)

    return x_grid, y_grid, z_grid, normalized_image_colors


x_grid, y_grid, z_grid, color = get_platform()
print(x_grid.shape, y_grid.shape, z_grid.shape)

def set_opaqueness_of_image_dots():
    color[:, 3] = dot_opaqueness


def draw_scatter_picture():
    set_opaqueness_of_image_dots()

    pos = np.array([x_grid, y_grid, z_grid]).T
    pos[:, 0] -= board_size // 2
    pos[:, 1] -= board_size // 2
    sp2 = gl.GLScatterPlotItem(pos=pos, color=color, size=dot_size)
    sp2.setGLOptions('translucent')
    w.addItem(sp2)


if are_visible_dots:
    draw_scatter_picture()


# ------- plotting walks

def signed_height_difference(x1, y1, x2, y2):
    return z_grid[x2 * board_size + y2] - z_grid[x1 * board_size + y1]


def unsigned_height_difference(x1, y1, x2, y2):
    return np.abs(signed_height_difference(x1, y1, x2, y2))


def sufficient_speed(height):
    return np.sqrt(2 * g * height)


def get_points_of_step(x, y, x_next, y_next, speed):
    x_difference, y_difference = x_next - x, y_next - y
    z = z_grid[(x + x_difference) * board_size + (y + y_difference)] - z_grid[x * board_size + y]
    d = np.sqrt(x_difference ** 2 + y_difference ** 2)
    xs = np.linspace(x, x + x_difference, points_in_jump)
    ys = np.linspace(y, y + y_difference, points_in_jump)

    def z_coord(distance):
        return speed * np.sin(theta) * distance - (g * distance ** 2) / 2

    theta = np.arctan((speed ** 2 + np.sqrt(speed ** 4 - g * (g * d ** 2 + 2 * z * speed ** 2))) / (g * d))
    t = d / (np.cos(theta) * speed)
    ts = np.linspace(0, t, points_in_jump)

    zs = z_grid[x * board_size + y] + z_coord(ts)

    return np.array([xs, ys, zs]).T


def draw_step(x, y, x_nxt, y_nxt):
    height_diff = signed_height_difference(x, y, x_nxt, y_nxt)
    speed = sufficient_speed(
        (1 if height_diff < 0 else height_diff)
        + np.random.randint(min_speed_boost, max_speed_boost))

    points = get_points_of_step(x, y, x_nxt, y_nxt, speed)

    # center the plot
    points[:, [0, 1]] -= np.array([board_size // 2, board_size // 2])

    line = gl.GLLinePlotItem()
    line.setGLOptions('translucent')
    line.setData(pos=points, color=tuple(color[x * board_size + y]), width=line_width)
    w.addItem(line)


def on_board(x, y):
    return 0 <= x < board_size and 0 <= y < board_size


def get_one_step_directions():
    dx = [-1, 0, 1]
    xs, ys = np.meshgrid(dx, dx)
    d_coord = np.transpose(np.array([xs.ravel(), ys.ravel()]))
    d_coord = d_coord[(d_coord[:, 0] != 0) | (d_coord[:, 1] != 0)]
    return d_coord


one_step_directions = get_one_step_directions()


def get_random_walk(path_length=100, trials=100):
    path = np.zeros((path_length + 1, 2))
    path[0] = [board_size // 2, board_size // 2]

    def in_path(x_coordinate, y_coordinate):
        return any(np.equal(path, [x_coordinate, y_coordinate]).all(1))

    for i in range(path_length - 1):
        found = False
        for j in range(trials):
            x_nxt, y_nxt = path[i] + one_step_directions[np.random.randint(one_step_directions.shape[0])]
            if on_board(x_nxt, y_nxt) and not in_path(x_nxt, y_nxt):
                path[i + 1] = [x_nxt, y_nxt]
                found = True
                break
        if not found:
            break

    return path.astype(int)


def draw_walk(walk):
    for i in range(walk.shape[0] - 2):
        if np.array_equal(walk[i + 1], walk[i + 2]):
            break
        x, y = walk[i]
        x_nxt, y_nxt = walk[i + 1]
        draw_step(x, y, x_nxt, y_nxt)


for i in range(number_of_random_walks):
    draw_walk(get_random_walk(path_length=length_of_random_walk))

if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QApplication.instance().exec_()
