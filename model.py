import numpy as np

epochs = 100
batch_size = 10
number_of_snakes = 20

assert number_of_snakes % 2 == 0
snake_ids = np.arange(number_of_snakes)

# use n+n generation
number_of_parents = 5
number_of_kids = number_of_snakes - number_of_parents

min_weight, max_weight = -5, 5

max_visibility_distance = 5
brain_size = (max_visibility_distance * 2 + 1) * (max_visibility_distance + 1)
default_visibility_distance = 4
board_size = 100

min_mutation, max_mutation = -5, 5
mutation_probability = 1 / 6

view_directions = 4
move_directions = 7

np.random.seed(18789293)

weights_size = brain_size * (max_visibility_distance + 1)
mutation_sample_size = int(mutation_probability * weights_size)

# I use row coordinates [x[], y[]]

clockwise_turn = np.array([[0, -1], [1, 0]])
clockwise_powers = np.array([np.linalg.matrix_power(clockwise_turn, i) for i in range(4)])
symmetric_turn = np.array([[1, 0], [0, -1]])


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
    return np.loadtxt(fname='brains/268.csv', delimiter=',').reshape((brain_size, move_directions))
    # return np.random.randint(min_weight, max_weight + 1, (brain_size, move_directions))


def get_cautiousness_coefficients():
    def chebyshev_coefficient(coordinate):
        max_coordinate = np.amax(np.abs(coordinate))
        return 1. - max_coordinate / (max_visibility_distance + 1)

    cautiousness = np.apply_along_axis(chebyshev_coefficient, 0, get_initial_brain_coordinates())
    # we don't consider weights under head
    head_coordinate = (max_visibility_distance + 1) * max_visibility_distance
    cautiousness.T[head_coordinate] = 0
    # we use symmetric weights and don't want a danger to be counted twice
    cautiousness[::max_visibility_distance + 1] /= 2
    return cautiousness


def get_cautious_brain_weights():
    cautiousness_coefficients = get_cautiousness_coefficients()
    brain_weights = get_initial_brain_weights()
    return brain_weights * cautiousness_coefficients[:, np.newaxis]


field = None


def clear_field():
    global field
    field = np.array([[set() for _ in range(board_size)] for _ in range(board_size)])


clear_field()

# def add_randomly_visited_cells(snake):
#     number_of_randomly_visited_cells = board_size
#     for cell in range(number_of_randomly_visited_cells):
#         i, j = np.random.randint(low=0, high=board_size - 1, size=2)
#         field[i][j].add(snake)

# for snake_id in snake_ids:
#     add_randomly_visited_cells(snake_id)

snake_brains = np.array([get_cautious_brain_weights() for snake_id in snake_ids])


# snake_brains = np.array()
#                         for snake_id in snake_ids)
# print(snake_brains)


def on_board(point):
    return np.all((0 <= point) & (point < board_size))


def visited(point, snake):
    return on_board(point) and (snake in field[point[0]][point[1]])


def bad(point, snake):
    return not on_board(point) \
           or on_board(point) and visited(point, snake)


def get_bad_indices(point, brain_half, snake):
    visible_coordinates = brain_half + point.T[:, np.newaxis]
    return np.where([bad(point=point, snake=snake) for point in visible_coordinates.T])


def get_all_bad_indices(point, direction, snake):
    left_half = get_bad_indices(point, symmetric_brain_coordinates[direction], snake)
    right_half = get_bad_indices(point, brain_coordinates[direction], snake)
    return np.append(left_half, right_half)


def get_move_direction(point, direction, snake):
    return np.argmax(np.sum(snake_brains[snake][get_all_bad_indices(point, direction, snake)], axis=0))


def get_step_coordinate(view_direction):
    initial_step_coordinates = np.array([[1, 1], [0, 1]])
    return np.dot(clockwise_powers[view_direction],
                  np.concatenate([np.dot(clockwise_powers[i], initial_step_coordinates)
                                  for i in range(view_directions)], axis=1))


step_coordinates = np.array([get_step_coordinate(view_direction=direction)
                             for direction in range(view_directions)]).astype(int)


def get_snake_path(snake):
    path_length = 100
    path = np.zeros((2, path_length)).astype(int)
    path[:, 0] = np.random.randint(low=0, high=board_size, size=2)
    view_direction = 0

    for i in range(path_length - 1):
        move_direction = get_move_direction(point=path[:, i], direction=view_direction, snake=snake)
        view_direction = move_direction // 2
        next_point = path[:, i] + step_coordinates[view_direction][:, move_direction]
        if not bad(next_point, snake=snake):
            path[:, i + 1] = next_point
            field[next_point[0]][next_point[1]].add(snake)
        else:
            path = path[:, :i + 1]
            break
    return path


def get_snake_paths(needed_paths):
    snake = 0
    for i in range(needed_paths):
        clear_field()
        np.savetxt(fname=f"paths/{i:03}", X=get_snake_path(snake).astype(int), delimiter=',', fmt='%i')


def get_snake_path_length(snake):
    path_length = 1
    current_point = np.random.randint(low=0, high=board_size - 1, size=2)
    view_direction = 0

    for i in range(board_size ** 2):
        move_direction = get_move_direction(point=current_point, direction=view_direction, snake=snake)
        view_direction = move_direction // 2
        next_point = current_point + step_coordinates[view_direction][:, move_direction]
        if not bad(point=next_point, snake=snake):
            field[next_point[0]][next_point[1]].add(snake)
            path_length += 1
            current_point = next_point
            # print(next_point)
        else:
            break
    return path_length


def run_snakes(iteration):
    path_lengths = np.zeros(number_of_snakes).astype(int)
    for snake in snake_ids:
        # path_lengths[snake] = get_snake_path(snake).shape[1]
        path_lengths[snake] = get_snake_path_length(snake)
    top_snakes = np.argsort(-path_lengths)[:number_of_parents]
    top_lengths = path_lengths[top_snakes]

    if iteration == batch_size - 1:
        print(top_lengths)
        np.savetxt(fname=f"brains/{top_lengths[0]}.csv", X=snake_brains[top_snakes[0]], fmt='%.3f', delimiter=',')

    crossover_probabilities = top_lengths / np.sum(top_lengths)
    pairs = np.random.choice(a=top_snakes, p=crossover_probabilities, size=(2, number_of_kids))
    snake_brains[:number_of_parents] = snake_brains[np.sort(top_snakes)[::-1]]

    for index, [pa, ma] in enumerate(pairs.T):
        crossover(pa, ma, index + number_of_parents)

    for snake in snake_ids:
        mutation(snake)

    clear_field()


def crossover(snake_a_id, snake_b_id, target_id):
    point = np.random.randint(0, brain_size)
    snake_brains[target_id][point:] = snake_brains[snake_a_id][point:]
    snake_brains[target_id][:point] = snake_brains[snake_b_id][:point]


def mutation(pa):
    mutation_indices = np.random.randint(low=0, high=weights_size, size=mutation_sample_size)
    mutation_values = np.random.randint(low=min_mutation, high=max_mutation + 1, size=mutation_sample_size)
    snake_brains[pa].ravel()[mutation_indices] += mutation_values


# run GA
# for i in range(epochs):
#     for j in range(batch_size):
#         run_snakes(j)

# produce paths
get_snake_paths(20)
