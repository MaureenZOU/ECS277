import numpy as np

def get_test_volume(volume_size):
    x = np.arange(volume_size[0]).reshape(volume_size[0], 1, 1).astype(np.float)
    y = np.arange(volume_size[1]).reshape(1, volume_size[1], 1).astype(np.float)
    z = np.arange(volume_size[2]).reshape(1, 1, volume_size[2])

    volume = (x**2+y**2+z**2)
    print('Loading data done.')
    return volume
