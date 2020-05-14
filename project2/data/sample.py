import numpy as np
from sklearn.neighbors import KDTree

def sample_scatter(data, sample_num=2000):
    '''
    return KDTree structured data
    '''
    w,h,d = data.shape
    
    X = np.arange(w)
    Y = np.arange(h)
    Z = np.arange(d)

    x,y,z = np.meshgrid(X,Y,Z)
    sample_points_order = np.concatenate((y[None,], x[None,], z[None,]), axis=0).reshape(3, w*h*d).transpose(1,0)
    # test = sample_points_order.reshape(w,h,d,3)

    sample_points = np.random.permutation(sample_points_order)[0:sample_num]

    sample_tree = KDTree(sample_points, leaf_size=4)

    mask = np.zeros((w,h,d))
    f_volume = np.zeros((w,h,d))

    F = []
    for i in range(0, sample_points.shape[0]):
        x,y,z = sample_points[i]
        F.append(data[x,y,z])
        mask[x,y,z] = 1
        f_volume[x,y,z] = data[x,y,z]

    F = np.array(F)

    # nn = sample_tree.query(sample_points[100:101], k=5, return_distance=False)
    # datas = sample_tree.get_arrays()[0]

    return sample_tree, F, sample_points_order, mask, f_volume