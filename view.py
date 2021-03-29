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
min_speed_boost, max_speed_boost = 4, 5
# show dots
are_visible_dots = False
# width of lines on path
line_width = 1
# for picture dots
dot_opaqueness = 1.
number_of_random_walks = 1
length_of_random_walk = 20
# resolution of jump
time_frames_in_jump = 20
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


def get_point_cloud_and_colors():
    # returns cloud of points as 3 arrays and their colors
    image = get_compressed_image('images/deadpool.png')
    x_grid, y_grid = get_coordinate_grid()
    flat_image = np.reshape(image, (image.shape[0] ** 2, 4))
    rgb_to_height = get_rgba_to_height(flat_image)
    z_grid = np.apply_along_axis(rgb_to_height, 1, flat_image)

    # need values in [0;1]
    normalized_image_colors = (flat_image / 255.).astype(np.float32)

    return x_grid, y_grid, z_grid, normalized_image_colors


x_grid, y_grid, z_grid, color = get_point_cloud_and_colors()


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


# ------- plotting walks

def signed_height_difference(x1, y1, x2, y2):
    return z_grid[x2 * board_size + y2] - z_grid[x1 * board_size + y1]


def unsigned_height_difference(x1, y1, x2, y2):
    return np.abs(signed_height_difference(x1, y1, x2, y2))


def sufficient_speed(height):
    return np.sqrt(2 * g * height)


def get_points_of_step(x, y, x_next, y_next, speed):
    dx, dy = x_next - x, y_next - y
    dz = z_grid[(x + dx) * board_size + (y + dy)] - z_grid[x * board_size + y]
    distance_in_xy_plane = np.sqrt(dx ** 2 + dy ** 2)
    xs = np.linspace(x, x + dx, time_frames_in_jump)
    ys = np.linspace(y, y + dy, time_frames_in_jump)

    def z(time):
        return speed * np.sin(theta) * time - (g * time ** 2) / 2

    theta = np.arctan(
        (speed ** 2 + np.sqrt(speed ** 4 - g * (g * distance_in_xy_plane ** 2 + 2 * dz * speed ** 2)))
        / (g * distance_in_xy_plane))
    time = distance_in_xy_plane / (np.cos(theta) * speed)
    times = np.linspace(0, time, time_frames_in_jump)

    zs = z_grid[x * board_size + y] + z(times)

    return np.array([xs, ys, zs]).T


def draw_step(x, y, x_next, y_next):
    height_difference = signed_height_difference(x, y, x_next, y_next)
    speed = sufficient_speed(
        (1 if height_difference < 0 else height_difference)
        + np.random.randint(min_speed_boost, max_speed_boost))

    points = get_points_of_step(x, y, x_next, y_next, speed)

    # center the plot
    points[:, [0, 1]] -= np.array([board_size // 2, board_size // 2])

    line = gl.GLLinePlotItem()
    # so that points are not transparent
    line.setGLOptions('translucent')
    line.setData(pos=points, color=tuple(color[x * board_size + y]), width=line_width)
    w.addItem(line)


def on_board(point):
    return np.all((point >= 0) & (point < board_size))


def get_one_step_directions():
    dx = [-1, 0, 1]
    xs, ys = np.meshgrid(dx, dx)
    d_coord = np.transpose(np.array([xs.ravel(), ys.ravel()]))
    d_coord = d_coord[(d_coord[:, 0] != 0) | (d_coord[:, 1] != 0)]
    return d_coord


one_step_directions = get_one_step_directions()


def get_random_walk(path_length=100):
    path = np.zeros((path_length, 2))
    path[0] = [board_size // 2, board_size // 2]

    def in_path(point):
        return any(np.equal(path, point).all(1))

    for step in range(path_length - 1):
        direction = (path[step] + one_step_directions)
        direction = direction[np.apply_along_axis(on_board, 1, direction) &
                              (np.logical_not(np.apply_along_axis(in_path, 1, direction)))]
        if len(direction) != 0:
            path[step + 1] = direction[np.random.randint(len(direction))]
        else:
            path = path[:step + 1]
            break
    return path.astype(int)


def draw_walk(walk):
    for i in range(walk.shape[0] - 1):
        draw_step(*np.ravel([walk[i], walk[i + 1]]))


def draw_walks():
    for i in range(number_of_random_walks):
        draw_walk(get_random_walk(path_length=length_of_random_walk))


if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    if are_visible_dots:
        draw_scatter_picture()
    draw_walks()
    QApplication.instance().exec_()
