import os
import cv2
import numpy as np

class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().
    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)

cfg = Config({
    'data_root': './data/raw/neghip_64x64x64_uint8.raw',
    'vol_size': [64, 64, 64],
    'file_name': 'local_shaper_s2' + '_k10',
    'method': 'local_shaper_s3',
    'knn': 5,
    'sample_num': 4000
})

def get_cfg():
    return cfg