import numpy as np

max_visibility_distance = 3
brain_size = (max_visibility_distance * 2 + 1) * (max_visibility_distance + 1)
default_visibility_distance = 4
board_size = 5

view_directions = 4
move_directions = 7
min_weight, max_weight = 0, 100

np.random.seed(1828283)

# I use row coordinates [x[], y[]]

clockwise_turn = np.array([[0, -1], [1, 0]])
clockwise_powers = np.array([np.linalg.matrix_power(clockwise_turn, i) for i in range(4)])
symmetric_turn = np.array([[1,0], [0,-1]])

def get_initial_brain_coordinates():
    pos = np.array(np.meshgrid(np.arange(-max_visibility_distance, max_visibility_distance + 1),
                               np.arange(0, max_visibility_distance + 1), indexing='ij'))
    return np.reshape(pos, (2, pos.shape[1] * pos.shape[2]))

def get_symmetric_brain_coordinates(direction):
    pos = get_initial_brain_coordinates()
    pos = np.dot(symmetric_turn, pos)
    return np.dot(clockwise_powers[direction], pos)


def get_brain_coordinates(direction):
    pos = get_initial_brain_coordinates()
    return np.dot(clockwise_powers[direction], pos)

brain_coordinates = np.array([get_brain_coordinates(direction) for direction in range(view_directions)])
symmetric_brain_coordinates = \
    np.array([get_symmetric_brain_coordinates(direction) for direction in range(view_directions)])

def get_initial_brain_weights():
    return np.random.randint(min_weight, max_weight + 1, (brain_size, move_directions))

def get_cautiousness_coefficients():
    def chebyshev_coefficient(coordinate):
        max_coordinate = np.amax(np.abs(coordinate))
        return 1. - max_coordinate/(max_visibility_distance+1)
    cautiousness = np.apply_along_axis(chebyshev_coefficient, 0, get_initial_brain_coordinates())
    # we don't consider weights under head
    head_coordinate = (max_visibility_distance + 1) * max_visibility_distance
    cautiousness.T[head_coordinate] = 0
    # we use symmetric weights and don't want a danger to be counted twice
    cautiousness[::max_visibility_distance+1] /= 2
    return cautiousness

cautiousness_coefficients = get_cautiousness_coefficients()
brain_weights = get_initial_brain_weights()
cautious_brain_weights = brain_weights * cautiousness_coefficients[:, np.newaxis]

field = np.array([[set() for i in range(board_size)] for j in range(board_size)])


def add_randomly_visited_cells(id):
    number_of_randomly_visited_cells = board_size
    for cell in range(number_of_randomly_visited_cells):
        i, j = np.random.randint(0, board_size - 1, (2))
        field[i][j].add(id)

snake_ids = [2, 3, 4]
for snake_id in snake_ids:
    add_randomly_visited_cells(snake_id)

# print(field)
# print(field[np.array([0,0])])

def get_bad_indices(x, y, brain_half, id):
    visible_coordinates = brain_half + [[x], [y]]
    print(visible_coordinates)
    def on_board(point):
        return np.all((0 <= point) & (point < board_size))
    def visited(point):
        return on_board(point) and (id in field[point[0]][point[1]])
    def bad(point):
        return not on_board(point) \
               or on_board(point) and visited(point)
    return np.where(np.apply_along_axis(bad, 0, visible_coordinates))[0]

def get_all_bad_indices(x, y, dir, id):
    left_half = get_bad_indices(x, y, symmetric_brain_coordinates[dir], id)
    right_half = get_bad_indices(x, y, brain_coordinates[dir], id)
    return np.append(left_half, right_half)


print(get_all_bad_indices(2,2,0,2))


# print(np.reshape(cautiousness_coefficients, (max_visibility_distance*2+1, max_visibility_distance+1)))

# get_cautiousness_coefficients()

# print(get_initial_brain_weights())

# print(get_brain_coordinates(1))
#
# def get_chebyshev_indices(direction, distance):
#     xs, ys = np.meshgrid([distance], np.arange(-distance, distance + 1))
#     line = np.array([xs.ravel(), ys.ravel()])
#     turn_indices = [(directions + i) % directions for i in range(direction - 1, (direction + 1) + 1)]
#     turned_lines = [np.dot(clockwise_powers[i], line) for i in turn_indices]
#     return np.concatenate((turned_lines), axis=1)
#
#
# # 0:^, 1:> 2:v 3:<
# chebyshev_indices = [[get_chebyshev_indices(direction=direction, distance=distance)
#                       for distance in range(max_visibility_distance + 1)]
#                      for direction in range(4)]
#
#
#
# def get_visible_coordinates(x, y, direction, distance):
#     return (chebyshev_indices[direction][distance] + np.array([[x], [y]])).T
#
# def get_on_board_indices(x, y, direction, distance):
#     visible_coordinates = get_visible_coordinates(x=x, y=y, direction=direction, distance=distance)
#     on_board = np.bitwise_and.reduce((0 <= visible_coordinates) & (visible_coordinates < board_size), axis=1)
#     return np.where(on_board)[0]
#
#
# def get_invalid_indices(x, y, direction, distance):
#     on_board_indices = get_on_board_indices(x=x, y=y, direction=direction, distance=distance)
#     np.
#     # visible_coordinates = get_visible_coordinates(x=x, y=y, direction=direction, distance=distance)
#     # are_bad = np.logical_not(
#     #     np.bitwise_and.reduce((0 <= visible_coordinates) & (visible_coordinates < board_size), axis=1))
#     # return np.where(are_bad)[0]
#
#
# # print(get_invalid_indices(x=2, y=2, direction=1, distance=3))
#
# field = np.array([[set() for i in range(board_size)] for j in range(board_size)])
#
#
# def add_randomly_visited_cells(id):
#     number_of_randomly_visited_cells = board_size
#     for cell in range(number_of_randomly_visited_cells):
#         i, j = np.random.randint(0, board_size - 1, (2))
#         field[i][j].add(id)
#
#
# snake_ids = [2, 3, 4]
# for snake_id in snake_ids:
#     add_randomly_visited_cells(snake_id)
#
# print(field)
#
# def on_board(point):
#     return np.all((0 <= point) & (point < board_size))
#
# def obstacle_at_dist(x, y, direction, snake_id):
#     for distance in range(1, max_visibility_distance + 1):
#         number_of_visible_coordinates = chebyshev_indices[direction][distance].shape[1]
#         invalid_ind = get_invalid_indices(x=x, y=y, distance=distance, direction=direction)
#         visited_ind = get_on_board_indices(x=x, y=y, distance=distance, direction=direction)
#         # print(visited_ind)
#         # all indices of visible cells that can not/shouldn't  be visited
#         invalid_ind = np.sort(np.append(invalid_ind, [visited_ind]).astype(int))
#         print(invalid_ind)
#         if invalid_ind.shape[0] > 0:
#             danger_direction, danger_count = \
#                 np.unique(invalid_ind // (number_of_visible_coordinates // 3), return_counts=True)
#             danger_counts = np.full(3, 1.)
#             danger_counts[danger_direction] = 1./danger_count
#             return distance, danger_counts
#     return max_visibility_distance+1, np.array([0,0,0])
#
#
# print(obstacle_at_dist(1, 1, 0, 3))
#
# # snakes_number = 20
# # directions = 7
# # # 0-cells from directions store snake's scores+
# #
# # # snakes_eyes = np.random.rand((snakes_number, directions, max_visibility_distance+1))
# #
# # # get_visible_indices(x=2, y=4, distance=2, direction=2)
# # obstacle_at_dist(x=3, y=3, direction=0, id = 2)
