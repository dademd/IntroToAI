import numpy as np

max_visibility_distance = 7
default_visibility_distance = 4
board_size = 6

np.random.seed(1828283)

# chebyshev_line = [[] for i in range(max_visibility_distance + 1)]

def get_chebyshev_line(distance):
    return


chebyshev_line = [get_chebyshev_line(distance) for distance in range(max_visibility_distance + 1)]

clockwise_turn = np.array([[0, -1], [1, 0]])
clockwise_powers = np.array([np.linalg.matrix_power(clockwise_turn, i) for i in range(4)])

directions = 4


def get_chebyshev_indices(direction, distance):
    xs, ys = np.meshgrid([distance], np.arange(-distance, distance + 1))
    line = np.array([xs.ravel(), ys.ravel()])
    turn_indices = [(directions + i) % directions for i in range(direction - 1, (direction + 1) + 1)]
    turned_lines = [np.dot(clockwise_powers[i], line) for i in turn_indices]
    return np.concatenate((turned_lines), axis=1)


chebyshev_indices = [[get_chebyshev_indices(direction=direction, distance=distance)
                      for distance in range(max_visibility_distance + 1)]
                     for direction in range(4)]

print(chebyshev_indices[0][1])


def get_visible_indices(x, y, direction, distance):
    ind = chebyshev_indices[direction][distance] + np.array([x, y])
    avail_ind = ind[np.all((ind >= 0) & (ind < board_size))]
    return avail_ind


field = np.full((board_size, board_size), set())

# np.random.seed(192)
# id = 2
# xs, ys = np.random.random_integers(0, board_size-1, (2, 10))
# for i, j in zip(xs, ys):
#     field[i][j].add(id)
#
# # print(field)
#
# def obstacle_at_dist(x, y, direction, id):
#     for distance in range(1, max_visibility_distance+1):
#         avail_ind = get_visible_indices(x=x, y=y, distance=distance, direction=direction)
#         if len(chebyshev_indices[direction][distance]) != avail_ind.shape[0]:
#             return distance
#         for i, j in avail_ind:
#             if id in field[i][j]:
#                 return distance
#     return default_visibility_distance
#
# snakes_number = 20
# directions = 7
# # 0-cells from directions store snake's scores+
#
# # snakes_eyes = np.random.rand((snakes_number, directions, max_visibility_distance+1))
#
# # get_visible_indices(x=2, y=4, distance=2, direction=2)
# obstacle_at_dist(x=3, y=3, direction=0, id = 2)
