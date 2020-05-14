import numpy as np

def load_raw(filename, volsize):
    """ inspired by mhd_utils from github"""
    dim = 3
    element_channels = 1
    np_type = np.ubyte

    arr = list(volsize)
    volume = np.prod(arr[0:dim - 1])

    shape = (arr[dim - 1], volume, element_channels)
    with open(filename,'rb') as fid:
        data = np.fromfile(fid, count=np.prod(shape),dtype = np_type)
    data.shape = shape

    arr.reverse()
    data = data.reshape(arr)

    print('Loading data done.')
    return data