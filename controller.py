# %%

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Fixing random state for reproducibility
# np.random.seed(19680801)

# constants
# resolution <= 512
board_size = 50
# maximum height of points
max_height = 20
# size of dots
dot_size = 2
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

# ------- initializing the box
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_visible(False)
ax.axis('off')
ax.set_xlabel('X axis', fontsize=19)
ax.set_ylabel('Y axis', fontsize=19)
ax.set_zlabel('Z axis', fontsize=19)


# ------- plotting platforms


def get_platform():
    a = Image.open('images/deadpool.png')
    a = np.asarray(a)
    side = 512
    assert side == len(a)
    factor = side // board_size

    xpos, ypos = np.meshgrid(np.arange(board_size), np.arange(board_size), indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()

    # compressed pic
    # not integers !
    t = a[::factor][:, ::factor]
    t = t.astype(int)

    # plt.imshow(t)
    # plt.show()
    def rgb_to_height_hash():
        rgb_range = 0

        def rgb_hash(p):
            return p[0] ** 0.2 + p[1] ** 0.5 + p[2] ** 0.6

        # def rgb_hash(p):
        #     return p[0] + p[1] * 1000 + p[2] * 1000000

        for row in range(t.shape[0]):
            for col in range(t.shape[1]):
                tup = t[row][col][:3]
                res = rgb_hash(tup)
                if rgb_range < res:
                    rgb_range = res

        return lambda rgb_triplet: rgb_hash(rgb_triplet) / rgb_range * max_height

    rgb_to_height = rgb_to_height_hash()

    zpos = np.ndarray((board_size, board_size))
    for i in range(board_size):
        for j in range(board_size):
            zpos[i][j] = rgb_to_height(t[i][j])

    zpos = zpos.ravel()

    # converts 3-tuple of floats to hex rgb
    def RGBA_to_hex(t):
        return f'#{t[0]:02x}{t[1]:02x}{t[2]:02x}'

    def to_color_map(n, t):
        color_map = [0 for i in range(n ** 2)]
        for i in range(n):
            for j in range(n):
                color_map[i * n + j] = RGBA_to_hex(t[i][j])
        return color_map

    return xpos, ypos, zpos, to_color_map(board_size, t)


x_pos, y_pos, z_pos, color_map = get_platform()
ax.view_init(elev=90., azim=0)

if are_visible_dots:
    ax.scatter3D(x_pos, y_pos, z_pos, s=dot_size, color=color_map)


# plt.show()

# exit()


# ------- plotting jumps

def not_negative(d, z, v):
    return v ** 4 - g * (g * d ** 2 + 2 * z * v ** 2) >= 0


def signed_height_difference(x1, y1, x2, y2):
    return z_pos[x2 * board_size + y2] - z_pos[x1 * board_size + y1]


def unsigned_height_difference(x1, y1, x2, y2):
    return np.abs(signed_height_difference(x1, y1, x2, y2))


def sufficient_speed(height):
    return np.sqrt(2 * g * height)


def safe(x1, y1, x2, y2):
    return unsigned_height_difference(x1, y1, x2, y2) <= max_jump_height


def step_f(x, y, x_nxt, y_nxt, v):
    x_diff, y_diff = x_nxt - x, y_nxt - y
    m_steps = 20
    # print(x,y)
    z = z_pos[(x + x_diff) * board_size + (y + y_diff)] - z_pos[x * board_size + y]
    d = np.sqrt(x_diff ** 2 + y_diff ** 2)
    xs = np.linspace(x, x + x_diff, m_steps)
    ys = np.linspace(y, y + y_diff, m_steps)

    def h(p):
        return v * np.sin(theta) * p - (g * p ** 2) / 2

    if not_negative(d, z, v) and safe(x, y, x + x_diff, y + y_diff):
        theta = np.arctan((v ** 2 + np.sqrt(v ** 4 - g * (g * d ** 2 + 2 * z * v ** 2))) / (g * d))
        t = d / (np.cos(theta) * v)
        ts = np.linspace(0, t, m_steps)

        zs = z_pos[x * board_size + y] + h(ts)
        ax.plot3D(xs, ys, zs, color_map[x * board_size + y])


def random_walk(min_dir=-1, max_dir=1, path_length=100, trials=100):
    xs, ys = np.meshgrid(np.arange(min_dir, max_dir + 1), np.arange(min_dir, max_dir + 1))
    d_coord = np.transpose(np.array([xs.ravel(), ys.ravel()]))
    d_coord = d_coord[(d_coord[:, 0] != 0) | (d_coord[:, 1] != 0)]

    def on_board(x, y):
        return 0 <= x < board_size and 0 <= y < board_size

    # to check fo 0,0
    random_path = np.zeros((path_length + 1, 2))
    random_path[0] = [board_size // 2, board_size // 2]

    def in_path(x_coord, y_coord):
        return any(np.equal(random_path, [x_coord, y_coord]).all(1))

    for i in range(path_length - 1):
        found = False
        for j in range(trials):
            x_nxt, y_nxt = random_path[i] + d_coord[np.random.randint(d_coord.shape[0])]
            if on_board(x_nxt, y_nxt) and not in_path(x_nxt, y_nxt):
                random_path[i + 1] = [x_nxt, y_nxt]
                found = True
                break
        if not found:
            break
    return random_path.astype(int)


# print(random_walk())
# walk = random_walk(path_length=2000)
#
# print(walk)

def put_walk(walk):
    for i in range(walk.shape[0] - 2):
        if np.array_equal(walk[i + 1], walk[i + 2]):
            break
        # print(walk[i])
        x, y = walk[i]
        x_nxt, y_nxt = walk[i + 1]
        height_diff = signed_height_difference(x, y, x_nxt, y_nxt)
        speed = sufficient_speed((1 if height_diff < 0
                                  else height_diff) + np.random.randint(min_speed_boost, max_speed_boost))
        step_f(x, y, x_nxt, y_nxt, speed)

n = 40
for i in range(n):
    put_walk(random_walk(path_length=300))

# for i in range(1, n - 1):
#     for j in range(1, n - 1):
#         for dx in range(-1, 2):
#             for dy in range(-1, 2):
#                 if dx != 0 or dy != 0:
#                     # h = height_difference(i, j, i+dx, j+dy) + 2
#                     # max_initial_speed = np.sqrt(2 * g * h)
#                     step_f(i, j, dx, dy, 1 + np.random.rand(1)[0] * max_initial_speed)

ax.view_init(elev=90., azim=0)

# unforunately, python can't handle histogram objects as 3d:(
plt.show()
