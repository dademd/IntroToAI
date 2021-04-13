import board
from globals import np
from parameters import board_size, view_directions
from steps import bad, step_coordinates, get_move_direction


def get_snake_path(snake=0, path_length=100):
    """generates a path from random point"""
    path = np.zeros((2, path_length)).astype(int)
    path[:, 0] = np.random.randint(low=0, high=board_size, size=2)
    view_direction = np.random.randint(low=0, high=4)

    for step in range(path_length - 1):
        move_direction, view_direction = get_move_direction(
            point=path[:, step], direction=view_direction, snake=snake
        )
        next_point = path[:, step] + step_coordinates[view_direction][:, move_direction]
        if not bad(next_point, snake=snake):
            path[:, step + 1] = next_point
            board.field[next_point[0]][next_point[1]].add(snake)
        else:
            path = path[:, : step + 1]
            break
    return path


def get_snake_paths(paths_number=0, snake=0):
    """saves several paths into files"""
    for i in range(paths_number):
        board.field = board.clear_field()
        snake_path = get_snake_path(snake).astype(int)
        if snake_path.shape[1] == 1:
            snake_path = np.append(snake_path, snake_path + [[0], [1]], axis=1)
        np.savetxt(fname=f"./paths/{i:03}.csv", X=snake_path, delimiter=",", fmt="%i")


def get_snake_path_length(snake):
    """calculates length of a path"""
    path_length = 1
    current_point = np.random.randint(low=0, high=board_size - 1, size=2)
    # current_point = np.array([board_size//2, board_size//2])
    view_direction = np.random.randint(view_directions)

    for step in range(board_size ** 2):
        move_direction, view_direction = get_move_direction(
            point=current_point, direction=view_direction, snake=snake
        )
        next_point = current_point + step_coordinates[view_direction][:, move_direction]
        if not bad(point=next_point, snake=snake):
            board.field[next_point[0]][next_point[1]].add(snake)
            path_length += 1
            current_point = next_point
        else:
            break
    return path_length
