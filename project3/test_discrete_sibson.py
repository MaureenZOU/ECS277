import scipy.interpolate
import numpy as np

import naturalneighbor

num_points = 10
num_dimensions = 3
points = np.random.rand(num_points, num_dimensions)
values = np.random.rand(num_points)

grids = tuple(np.mgrid[0:100:1, 0:50:100j, 0:100:2])
print(np.mgrid[0:100:1, 0:100:1, 0:1:1][:,0,0,0])
scipy_interpolated_values = scipy.interpolate.griddata(points, values, grids)

grid_ranges = [[0, 100, 1], [0, 50, 100j], [0, 100, 2]]
nn_interpolated_values = naturalneighbor.griddata(points, values, grid_ranges)