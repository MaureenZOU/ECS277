import torch

def global_shaper_s2(volume_locs, data_samples, sample_locs, size, mask, f_volume):
    print('start compuate glocal shaper S2 ...')
    n,c = volume_locs.shape
    volume_locs = volume_locs.reshape(n,1,c).float()
    n = data_samples.shape[0]
    data_samples = data_samples.reshape(1,n,1).float()
    n,c = sample_locs.shape
    sample_locs = sample_locs.reshape(1,n,c).float()

    d = (torch.pow(volume_locs - sample_locs, 2) + 1e-4)
    d = torch.sum(d, dim=2, keepdim=True)
    d = 1 / d

    df = d * data_samples
    out = (torch.sum(df, dim=1)/torch.sum(d, dim=1))[:,0]

    w,h,d = size
    out = out.reshape((w,h,d))

    out = out * (1-mask) + f_volume
    print('end compuate glocal shaper S2 ...')
    return out

def get_local_d_f(sample_ids, F_locs, F):
    n,k = sample_ids.shape
    dense_F_locs = torch.zeros((n,k,3))
    dense_F = torch.zeros((n,k,1))

    for i in range(0, n):
        for j in range(0, k):
            dense_F[i,j] = F[sample_ids[i,j]]
            dense_F_locs[i,j] = F_locs[sample_ids[i,j]]

    return dense_F, dense_F_locs

def local_shaper_s2(volume_locs, scattered_tree, size, F_locs, F, mask, f_volume, k=5):
    print('start compuate local shaper S2 ...')
    nn_samples = scattered_tree.query(volume_locs, k=k, return_distance=False)
    dense_F, dense_F_locs = get_local_d_f(nn_samples, F_locs, F)

    n,c = volume_locs.shape
    volume_locs = volume_locs.reshape(n,1,c).float()
    dense_F_locs = dense_F_locs.reshape(n,k,c)
    dense_F = dense_F.reshape(n,k,1)

    d = (torch.pow(volume_locs - dense_F_locs, 2) + 1e-4)
    d = torch.sum(d, dim=2, keepdim=True)
    d = 1 / d

    df = d * dense_F
    out = (torch.sum(df, dim=1)/torch.sum(d, dim=1))[:,0]

    w,h,d = size
    out = out.reshape((w,h,d))

    out = out * (1-mask) + f_volume

    print('end compuate local shaper S2 ...')
    return out
