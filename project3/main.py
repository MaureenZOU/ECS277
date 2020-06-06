from data import sample_gt_data
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import naturalneighbor
from scipy.misc import imsave

if __name__ == '__main__':
    sample_num = 100
    size = [100,100]

    xy, f = sample_gt_data(num=sample_num, size=size)

    cur_datapoint = xy[0:5]
    cur_f = f[0:5]
    remain_datapoint = xy[5:]
    remain_f = f[5:]
    iteration = (sample_num - len(cur_datapoint)) // 5

    for i in range(0, iteration):
        vor = Voronoi(cur_datapoint)

        fig = voronoi_plot_2d(vor)
        plt.savefig('out/' + '_vor.png')
        plt.close()

        # sibson interpolate
        z = np.zeros((cur_datapoint.shape[0], 1))
        xyz = np.concatenate((cur_datapoint,z), axis=1)
        grid_ranges = [[0, size[0], 1], [0, size[1], 1], [0, 1, 1]]
        nn_interpolated_values = naturalneighbor.griddata(xyz, cur_f, grid_ranges)[:,:,0]

        imsave('out/' + str(i) + '_sibson_interpolate.png', nn_interpolated_values)

        # get sibson interpolation with remain datapoint
        remain_interpolate = []
        for j in range(remain_datapoint.shape[0]):
            pts = remain_datapoint[j]
            remain_interpolate.append(nn_interpolated_values[int(pts[0]), int(pts[1])])
        remain_interpolate = np.array(remain_interpolate)

        d = (remain_interpolate - remain_f)**2
        rmse = (np.sum(d) / d.shape[0])**0.5

        # delauny triangulation
        min_radius = 0.25
        full_xy = np.concatenate((cur_datapoint, remain_datapoint), axis=0)
        full_f = np.concatenate((cur_f, remain_interpolate))
        full_f_correct = np.concatenate((cur_f, remain_f))

        # compute error of tile        
        x = full_xy[:,0]
        y = full_xy[:,1]
        triang = tri.Triangulation(x, y)
        triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                        y[triang.triangles].mean(axis=1))
                        < min_radius)
        fig1, ax1 = plt.subplots()
        ax1.set_aspect('equal')
        tpc = ax1.tripcolor(triang, full_f, shading='flat')
        plt.savefig('out/' + str(i) + '_delauny_triangulation.png')
        plt.close()

        print(full_f_correct-full_f)
        fig1, ax1 = plt.subplots()
        ax1.set_aspect('equal')
        tpc = ax1.tripcolor(triang, ((full_f_correct-full_f)**2)**0.5, shading='flat')
        plt.savefig('out/' + str(i) + '_delauny_triangulation_error.png')
        plt.close()

        sample_mse = [0 for j in range(0, cur_datapoint.shape[0])]
        sample_cnt = [0 for j in range(0, cur_datapoint.shape[0])]
        for j in range(0, remain_datapoint.shape[0]):
            sample = remain_datapoint[j:j+1]
            distance = np.sum((cur_datapoint - sample)**2, axis=1)
            sample_cnt[np.argmin(distance)] += 1
            sample_mse[np.argmin(distance)] += d[j]
        sample_rmse = (np.array(sample_mse) / (np.array(sample_cnt) + 1e-5))**0.5

        # visualize voronoi with color
        minima = min(sample_rmse)
        maxima = max(sample_rmse)

        norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues_r)

        voronoi_plot_2d(vor, show_points=True, show_vertices=False, s=1)
        for r in range(len(vor.point_region)):
            region = vor.regions[vor.point_region[r]]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                plt.fill(*zip(*polygon), color=mapper.to_rgba(sample_rmse[r]))

        plt.savefig('out/' + str(i) + '_err_vor.png')
        plt.close()

        topk = d.argsort()[0:5]
        leastk = d.argsort()[5:]
        insert_point = remain_datapoint[topk]
        insert_f = remain_f[topk]

        remain_datapoint = remain_datapoint[leastk]
        remain_f = remain_f[leastk]
        cur_datapoint = np.concatenate((cur_datapoint, insert_point), axis=0)
        cur_f = np.concatenate((cur_f, insert_f))