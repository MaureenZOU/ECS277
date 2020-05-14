import torch
from numpy.linalg import solve

def get_gradient_data(sample_ids, F_locs, F):
    n,k = sample_ids.shape
    dense_F_locs = torch.zeros((n,k,3))
    dense_F = torch.zeros((n,k,1))

    for i in range(0, n):
        for j in range(0, k):
            dense_F[i,j] = F[sample_ids[i,j]]
            dense_F_locs[i,j] = F_locs[sample_ids[i,j]]

    return dense_F, dense_F_locs

def global_shaper_s3(volume_locs, scattered_tree, size, F_locs, F, mask, f_volume, k=4):
    nn_samples = scattered_tree.query(F_locs, k=k, return_distance=False)
    dense_F, dense_F_locs = get_gradient_data(nn_samples, F_locs, F)
    n,k,c = dense_F.shape
    b = torch.ones((n,k,1))
    X = torch.cat((b,dense_F_locs), dim=2)
    X_T = X.permute(0,2,1)
    G = (torch.inverse(X_T @ X) @ X_T @ dense_F)[:,1:,0]

    n,c = volume_locs.shape
    volume_locs = volume_locs.reshape((n,1,c)).double()
    n,c = F_locs.shape
    F_locs = F_locs.reshape((1,n,c)).double()
    n = F.shape[0]
    f_i = F.reshape((1,n,1)).double()
    n,c = G.shape
    G = G.reshape((1,n,c)).double()

    F_i = (f_i + torch.sum((G*(volume_locs-F_locs)),dim=2,keepdim=True))[:,:,0]
    d_i_2 = torch.sum((volume_locs-F_locs)**2,dim=2) + 1e-3
    out = torch.sum((1/d_i_2)*F_i, dim=1) / torch.sum((1/d_i_2), dim=1)

    w,h,d = size
    out = out.reshape((w,h,d))
    out = out.double() * (1-mask.double()) + f_volume.double()
    return out


def get_local_d_f(sample_ids, F_locs, F, G):
    n,k = sample_ids.shape
    dense_F_locs = torch.zeros((n,k,3))
    dense_F = torch.zeros((n,k,1))
    dense_G = torch.zeros((n,k,3))

    for i in range(0, n):
        for j in range(0, k):
            dense_F[i,j] = F[sample_ids[i,j]]
            dense_G[i,j] = G[sample_ids[i,j]]
            dense_F_locs[i,j] = F_locs[sample_ids[i,j]]

    return dense_F, dense_G, dense_F_locs


def local_shaper_s3(volume_locs, scattered_tree, size, F_locs, F, mask, f_volume, k1=8, k2=5):
    nn_samples = scattered_tree.query(F_locs, k=k1, return_distance=False)
    dense_F, dense_F_locs = get_gradient_data(nn_samples, F_locs, F)
    n,k,c = dense_F.shape
    b = torch.ones((n,k,1))
    X = torch.cat((b,dense_F_locs), dim=2)
    X_T = X.permute(0,2,1)
    G = (torch.inverse(X_T @ X) @ X_T @ dense_F)[:,1:,0]

    nn_samples = scattered_tree.query(volume_locs, k=k2, return_distance=False)
    dense_F, dense_G, dense_F_locs = get_local_d_f(nn_samples, F_locs, F, G)

    n,c = volume_locs.shape
    volume_locs = volume_locs.reshape((n,1,c)).double()
    n,k,c = dense_F_locs.shape
    dense_F_locs = dense_F_locs.reshape((n,k,c)).double()
    n,k,c = dense_F.shape
    f_i = dense_F.reshape((n,k,1)).double()
    n,k,c = dense_G.shape
    G = dense_G.reshape((n,k,c)).double()

    F_i = (f_i + torch.sum((G*(volume_locs-dense_F_locs)),dim=2,keepdim=True))[:,:,0]
    d_i_2 = torch.sum((volume_locs-dense_F_locs)**2,dim=2) + 1e-3
    out = torch.sum((1/d_i_2)*F_i, dim=1) / torch.sum((1/d_i_2), dim=1)

    w,h,d = size
    out = out.reshape((w,h,d))
    out = out.double() * (1-mask.double()) + f_volume.double()
    return out