
import numpy as np

max_visibility_distance = 7
default_visibility_distance = 4
board_size = 6

chebyshev_list = [[] for i in range(max_visibility_distance+1)]

for d in range(1, max_visibility_distance+1):
    xs, ys = np.meshgrid([d], np.arange(-d, d+1))
    chebyshev_list[d] = np.array([xs.ravel(), ys.ravel()])

# print(np.array(chebyshev_list))

# directions 0>,1^,2<,3v

# for each direction, contains multipliers for xs, ys to watch in clockwise direction around map
# x, y

# counter-clockwise codes for indices
cc = np.array([[[1, 0], [1, 1]],
               [[0, 1], [1, 1]],
               [[1, 0], [1, -1]],
               [[0, 1], [-1, 1]]])
len_cc = len(cc)

def get_chebyshev_indices(direction, distance):
    assert 0 <= direction <= 3
    assert 0 <= distance <= max_visibility_distance
    if distance == 0:
        return np.empty(0,int)
    indices = np.empty((0,2), int)
    for i in range(direction-1, (direction+1)+1):
        ind_cc = (len_cc + i) % len_cc
        perm, mult = cc[ind_cc]
        chebyshev = chebyshev_list[distance]
        perm = mult[:, np.newaxis] * np.array([chebyshev[i] for i in perm])
        indices = np.append(indices, perm.T, axis=0)
    indices = indices.astype(int)
    index = np.unique(indices.astype(int), axis=0, return_index=True)[1]
    return np.array([indices])

chebyshev_indices = [[get_chebyshev_indices(distance=distance, direction=direction)
                      for distance in range(0, max_visibility_distance)]
                     for direction in range(0, 4)]

lst = chebyshev_indices[0][1]
# print(lst)
# lst_c = np.zeros(lst.shape)
# for i in range(len(lst)-1):
#     print(lst[i+1]-lst[i])
#
# print(lst_c)

def get_visible_indices(x, y, distance, direction):
    ind = chebyshev_indices[direction][distance] + np.array([x,y])
    avail_ind = ind[(0 <= ind[:,0]) & (ind[:,0] < board_size) & (0 <= ind[:,1]) & (ind[:,1] < board_size)]
    return avail_ind

field = np.array([[set() for i in range(board_size)] for j in range(board_size)])

np.random.seed(192)
id = 2
xs, ys = np.random.random_integers(0, board_size-1, (2, 10))
for i, j in zip(xs, ys):
    field[i][j].add(id)

# print(field)

def obstacle_at_dist(x, y, direction, id):
    for distance in range(1, max_visibility_distance+1):
        avail_ind = get_visible_indices(x=x, y=y, distance=distance, direction=direction)
        if len(chebyshev_indices[direction][distance]) != avail_ind.shape[0]:
            return distance
        for i, j in avail_ind:
            if id in field[i][j]:
                return distance
    return default_visibility_distance

snakes_number = 20
directions = 7
# 0-cells from directions store snake's scores+

# snakes_eyes = np.random.rand((snakes_number, directions, max_visibility_distance+1))

# get_visible_indices(x=2, y=4, distance=2, direction=2)
obstacle_at_dist(x=3, y=3, direction=0, id = 2)
