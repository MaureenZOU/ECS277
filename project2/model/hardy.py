import torch
from numpy.linalg import inv
import numpy as np
from numpy.linalg import solve

def global_hardy_positive(sample_points, F, F_locs, size, mask, f_volume):
    w,h,d = size
    R_2 = (w**2 + h**2 + d**2)
    n,c = F_locs.shape

    F_locs_dim1 = F_locs.reshape(n,1,c)
    F_locs_dim2 = F_locs.reshape(1,n,c)
    tmp = F_locs_dim1 - F_locs_dim2
    M = torch.sum((F_locs_dim1 - F_locs_dim2)**2, dim=2) + R_2
    M = torch.pow(M, 1/2)

    # M_i = torch.inverse(M)
    # C = M_i @ F.double()

    C = solve(M, F.double())
    C = torch.tensor(C).reshape(n,1)

    # print(M@C)
    # print(F)
    # exit()
    
    n,c = sample_points.shape
    sample_points = sample_points.reshape(n,1,c).double()
    n,c = F_locs.shape
    F_locs = F_locs.reshape(1,n,c).double()
    d = torch.pow(torch.sum((sample_points - F_locs)**2, dim=2) + R_2, 1/2)
    out = (d@C).reshape(size).double()

    out = out * (1-mask.double()) + f_volume.double()
    return out

def global_hardy_negative(sample_points, F, F_locs, size, mask, f_volume):
    w,h,d = size
    R_2 = w**2 + h**2 + d**2
    n,c = F_locs.shape

    F_locs_dim1 = F_locs.reshape(n,1,c)
    F_locs_dim2 = F_locs.reshape(1,n,c)
    M = torch.sum((F_locs_dim1 - F_locs_dim2)**2, dim=2) + R_2
    M = torch.pow(M, -1/2)
    C = solve(M, F)
    C = torch.tensor(C).reshape(n,1)

    n,c = sample_points.shape
    sample_points = sample_points.reshape(n,1,c).double()
    n,c = F_locs.shape
    F_locs = F_locs.reshape(1,n,c).double()
    d = torch.pow(torch.sum((sample_points - F_locs)**2, dim=2) + R_2, -1/2)
    out = (d@C).reshape(size).double()

    out = out * (1-mask.double()) + f_volume.double()
    return out


def get_local_d_c(sample_ids, F_locs, F):
    n,k = sample_ids.shape
    dense_F_locs = torch.zeros((n,k,3))
    dense_F = torch.zeros((n,k))

    for i in range(0, n):
        for j in range(0, k):
            dense_F[i,j] = F[sample_ids[i,j]]
            dense_F_locs[i,j] = F_locs[sample_ids[i,j]]

    return dense_F_locs.double(), dense_F.double()


def local_hardy_positive(sample_points, F, F_locs, size, mask, f_volume, scattered_tree, k=5):
    # get NN for sample points
    nn_samples = scattered_tree.query(sample_points, k=k, return_distance=False)
    dense_F_locs, dense_F = get_local_d_c(nn_samples, F_locs, F)

    w,h,d = size
    R_2 = w**2 + h**2 + d**2
    n,k,c = dense_F_locs.shape
    dense_F_locs_dim1 = dense_F_locs.reshape(n,k,1,c)
    dense_F_locs_dim2 = dense_F_locs.reshape(n,1,k,c)
    M = torch.sum((dense_F_locs_dim1 - dense_F_locs_dim2)**2, dim=3) + R_2
    M = torch.pow(M, 1/2)
    C = solve(M, dense_F)
    C = torch.tensor(C).reshape(n,k)

    n,c = sample_points.shape
    sample_points = sample_points.reshape(n,1,c).double()
    d = torch.pow(torch.sum((sample_points - dense_F_locs)**2, dim=2) + R_2, 1/2)
    out = torch.sum(C*d, dim=1).reshape(size).double()

    out = out * (1-mask.double()) + f_volume.double()
    return out


def local_hardy_negative(sample_points, F, F_locs, size, mask, f_volume, scattered_tree, k=5):
    # get NN for sample points
    nn_samples = scattered_tree.query(sample_points, k=k, return_distance=False)
    dense_F_locs, dense_F = get_local_d_c(nn_samples, F_locs, F)

    w,h,d = size
    R_2 = w**2 + h**2 + d**2
    n,k,c = dense_F_locs.shape
    dense_F_locs_dim1 = dense_F_locs.reshape(n,k,1,c)
    dense_F_locs_dim2 = dense_F_locs.reshape(n,1,k,c)
    M = torch.sum((dense_F_locs_dim1 - dense_F_locs_dim2)**2, dim=3) + R_2
    M = torch.pow(M, -1/2)
    C = solve(M, dense_F)
    C = torch.tensor(C).reshape(n,k)

    n,c = sample_points.shape
    sample_points = sample_points.reshape(n,1,c).double()
    d = torch.pow(torch.sum((sample_points - dense_F_locs)**2, dim=2) + R_2, -1/2)
    out = torch.sum(C*d, dim=1).reshape(size).double()

    out = out * (1-mask.double()) + f_volume.double()
    return out