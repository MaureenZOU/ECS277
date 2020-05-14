import cv2
import numpy as np
import matplotlib.pyplot as plt

def render_image(data, plane='xy', axis=0):
    if plane == 'xy':
        img = data[:,:,axis]
    elif plane == 'xz':
        img = data[:,axis,:]
    elif plane == 'yz':
        img = data[axis,:,:]
    
    plt.matshow(img)
    # plt.colorbar()
    plt.axis('off')
    
    return plt
