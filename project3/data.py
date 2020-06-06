import numpy as np

def sample_gt_data(num=50, size=[100,100]):
    X = np.random.randint(size[0], size=num)*1.0
    Y = np.random.randint(size[1], size=num)*1.0
    f = (X / size[0])**2 + (Y / size[1])**2

    return np.concatenate((X[:,None], Y[:,None]), axis=1), f