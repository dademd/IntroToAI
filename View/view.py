from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
import numpy as np
from PIL import Image
import sys
from PyQt5 import QtTest

from constants import *
from parameters import *

app = QApplication([])
w = gl.GLViewWidget()
w.show()
w.setCameraPosition(distance=view_height, elevation=elevation, azimuth=azimuth)

np.random.seed(random_seed)


def useless_code():
    g = gl.GLGridItem()
    g.setSize(x=board_size, y=board_size, z=board_height)
    w.addItem(g)

    ax = gl.GLAxisItem()
    ax.setSize(x=board_size, y=board_size, z=board_height)
    w.addItem(ax)


class ReadImage:
    @staticmethod
    def read_image(image_path=path_to_image):
        """open image and check it for validity"""
        image = Image.open(image_path)
        image = np.asarray(image).astype(np.float32)
        return image

    @staticmethod
    def check_image_size(image):
        """check if image is RGBA square with given side"""
        assert image.shape == tuple([image_side, image_side, 4])

    def read_valid_image(self):
        image = self.read_image(path_to_image)
        self.check_image_size(image)
        return image

    def get_compressed_image(self):
        """select a sub-image"""
        image = self.read_valid_image()
        factor = image_side // board_size
        compressed_image = image[::factor][:, ::factor]
        compressed_image = compressed_image[:board_size, :board_size]
        return compressed_image

    @staticmethod
    def get_image():
        """reading RGBA image from file"""
        image = ReadImage.get_compressed_image(path_to_image)
        assert image.shape == (board_size, board_size, 4)
        return image


class ConvertImageToPointCloud:
    def __init__(self):
        self.x_grid, self.y_grid = self.get_coordinate_grid()
        flat_image, self.z_grid = self.get_z_grid()
        self.normalized_image_colors = self.normalize_image_colors(flat_image)

    @staticmethod
    def get_coordinate_grid():
        """get x, y grid coordinates"""
        x_grid, y_grid = np.meshgrid(
            np.arange(board_size), np.arange(board_size), indexing="ij"
        )
        return x_grid.ravel(), y_grid.ravel()

    @staticmethod
    def get_rgba_to_height(image):
        """return mapping function from RGBA to height"""

        def rgba_hash(point):
            """example of hash function of rgba point"""
            return point[0] ** 0.2 + point[1] ** 0.5 + point[2] ** 0.6

        height = np.amax(np.apply_along_axis(rgba_hash, 1, image))
        return lambda rgba: rgba_hash(rgba) / height * board_height

    def get_z_grid(self):
        """return z-coordinates of grid points"""
        image = ReadImage.get_image()
        flat_image = np.reshape(image, (image.shape[0] ** 2, 4))
        rgba_to_height = self.get_rgba_to_height(flat_image)
        z_grid = np.apply_along_axis(rgba_to_height, 1, flat_image)
        return flat_image, z_grid

    @staticmethod
    def normalize_image_colors(image):
        """scale image colors into [0;1] range"""
        return (image / 255.0).astype(np.float32)


class VisualizePointCloud:
    def __init__(self):
        point_cloud = ConvertImageToPointCloud()
        self.x_grid = point_cloud.x_grid
        self.y_grid = point_cloud.y_grid
        self.z_grid = point_cloud.z_grid
        self.color = point_cloud.normalized_image_colors

    def set_opaqueness_of_image_dots(self):
        self.color[:, 3] = dot_opaqueness

    def get_centered_board(self):
        board_coordinates = np.array([self.x_grid, self.y_grid, self.z_grid]).T - np.array([board_size//2, board_size//2, 0])
        return board_coordinates

    def draw_scatter_picture(self):
        self.set_opaqueness_of_image_dots()
        pos = self.get_centered_board()
        sp2 = gl.GLScatterPlotItem(pos=pos, color=self.color, size=dot_size)
        sp2.setGLOptions("translucent")

        w.addItem(sp2)


# x_grid, y_grid, z_grid, color = ConvertImageToPointCloud.get_point_cloud_and_colors()


# ------- plotting walks

class JumpSettings:
    def signed_height_difference(self, x1, y1, x2, y2):
        return z_grid[x2 * board_size + y2] - z_grid[x1 * board_size + y1]

    def unsigned_height_difference(self, x1, y1, x2, y2):
        return np.abs(self.signed_height_difference(x1, y1, x2, y2))

    def sufficient_speed(self, height):
        return np.sqrt(2 * g * height)

    def get_points_of_jump(self, x, y, x_next, y_next):
        height_difference = signed_height_difference(x, y, x_next, y_next)
        speed = sufficient_speed(
            (2 if height_difference < 0 else height_difference)
            + np.random.randint(min_speed_boost, max_speed_boost)
        )

        dx, dy = x_next - x, y_next - y
        dz = z_grid[(x + dx) * board_size + (y + dy)] - z_grid[x * board_size + y]
        distance_in_xy_plane = np.sqrt(dx ** 2 + dy ** 2)
        xs = np.linspace(x, x + dx, time_frames_in_jump)
        ys = np.linspace(y, y + dy, time_frames_in_jump)

        def z(time):
            return speed * np.sin(theta) * time - (g * time ** 2) / 2

        jump_trajectory_shortener = -1 if shortest_jump else 1

        theta = np.arctan(
            (
                    speed ** 2
                    + jump_trajectory_shortener
                    * np.sqrt(
                speed ** 4 - g * (g * distance_in_xy_plane ** 2 + 2 * dz * speed ** 2)
            )
            )
            / (g * distance_in_xy_plane)
        )
        time = distance_in_xy_plane / (np.cos(theta) * speed)
        times = np.linspace(0, time, time_frames_in_jump)

        zs = z_grid[x * board_size + y] + z(times)

        return np.array([xs, ys, zs]).T

    def draw_jump(self, x, y, x_next, y_next):
        points = get_points_of_jump(x, y, x_next, y_next)
        colors = np.full((points.shape[0], 4), color[x * board_size + y])
        # center the plot
        points[:, [0, 1]] -= np.array([board_size // 2, board_size // 2])

        line = gl.GLLinePlotItem()
        # so that points are not transparent
        line.setGLOptions("translucent")
        line.setData(pos=points, color=colors, width=line_width)

        w.addItem(line)
        if real_time_path_drawing_enabled:
            QtTest.QTest.qWait(ms=line_delay)

    def on_board(self, point):
        return np.all((point >= 0) & (point < board_size))

    def get_one_step_directions(self):
        dx = [-1, 0, 1]
        xs, ys = np.meshgrid(dx, dx)
        d_coord = np.transpose(np.array([xs.ravel(), ys.ravel()]))
        d_coord = d_coord[(d_coord[:, 0] != 0) | (d_coord[:, 1] != 0)]
        return d_coord

    one_step_directions = get_one_step_directions()

    def get_random_walk(self, path_length=100):
        path = np.zeros((path_length, 2))
        path[0] = [board_size // 2, board_size // 2]

        def in_path(point):
            return any(np.equal(path, point).all(1))

        for step in range(path_length - 1):
            direction = path[step] + one_step_directions
            direction = direction[
                np.apply_along_axis(on_board, 1, direction)
                & (np.logical_not(np.apply_along_axis(in_path, 1, direction)))
                ]
            if len(direction) != 0:
                path[step + 1] = direction[np.random.randint(len(direction))]
            else:
                path = path[: step + 1]
                break
        return path.astype(int)

    # print(repr(get_random_walk(path_length=length_of_random_walk)))

    def draw_walk(self, walk):
        for i in range(walk.shape[0] - 1):
            draw_jump(*np.ravel([walk[i], walk[i + 1]]))

    def draw_walks_from_file(self):
        for i in range(number_of_paths):
            # print(i)
            snake_path = np.loadtxt(fname=f"paths/{i:03}.csv", delimiter=",").astype(int)
            # print(repr(snake_path.T))
            draw_walk(snake_path.T)

    def draw_random_walk(self):
        # print(get_random_walk(path_length=length_of_random_walk))
        draw_walk(get_random_walk(path_length=length_of_random_walk))

    # draw_random_walk()

    def draw_random_walks(self):
        for i in range(number_of_random_walks):
            draw_random_walk()


if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
    if are_visible_dots:
        draw_scatter_picture()
    draw_walks_from_file()
    # draw_random_walks()
    QApplication.instance().exec_()
