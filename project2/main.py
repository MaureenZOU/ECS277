import torch
import cv2

from data import load_raw, get_test_volume, sample_scatter
from model import global_hardy_positive, global_hardy_negative, local_hardy_positive, local_hardy_negative, global_shaper_s3, local_shaper_s3, global_shaper_s2, local_shaper_s2

from config import get_cfg
from render.render import render_image

import numpy as np
np.random.seed(7)

def main():
    cfg = get_cfg()
    # volume = load_raw(cfg.data_root, cfg.vol_size)
    volume = get_test_volume(cfg.vol_size)

    scattered_tree, F, sample_points, mask, f_volume = sample_scatter(volume, sample_num=cfg.sample_num)
    F_locs = torch.tensor(scattered_tree.get_arrays()[0])
    F = torch.tensor(F)
    mask = torch.tensor(mask).float()
    f_volume = torch.tensor(f_volume).float()
    sample_points = torch.tensor(sample_points)
    # sample points (w,h,d,n)

    if cfg.method == 'global_shaper_s2':
        out = global_shaper_s2(sample_points, F, F_locs, cfg.vol_size, mask, f_volume)
    elif cfg.method == 'local_shaper_s2':
        out = local_shaper_s2(sample_points, scattered_tree, cfg.vol_size, F_locs, F, mask, f_volume, k=cfg.knn)
    elif cfg.method == 'global_hardy_negative':
        out = global_hardy_negative(sample_points, F, F_locs, cfg.vol_size, mask, f_volume)
    elif cfg.method == 'global_hardy_positive':
        out = global_hardy_positive(sample_points, F, F_locs, cfg.vol_size, mask, f_volume)
    elif cfg.method == 'local_hardy_negative':
        out = local_hardy_negative(sample_points, F, F_locs, cfg.vol_size, mask, f_volume, scattered_tree, k=cfg.knn)
    elif cfg.method == 'local_hardy_positive':
        out = local_hardy_positive(sample_points, F, F_locs, cfg.vol_size, mask, f_volume, scattered_tree, k=cfg.knn)
    elif cfg.method == 'global_shaper_s3':
        out = global_shaper_s3(sample_points, scattered_tree, cfg.vol_size, F_locs, F, mask, f_volume, k=cfg.knn)
    elif cfg.method == 'local_shaper_s3':
        out = local_shaper_s3(sample_points, scattered_tree, cfg.vol_size, F_locs, F, mask, f_volume, k1=8, k2=cfg.knn)

    # plt.colorbar()
    # plt.savefig('out.png')

    plt = render_image(out, plane='xy', axis=32)
    plt.savefig(cfg.file_name + '_xy.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.clf()

    # plt = render_image(out, plane='yz', axis=32)
    # plt.savefig(name + '_yz.png', bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.clf()

    # plt = render_image(out, plane='xz', axis=32)
    # plt.savefig(name + '_xz.png', bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.clf()

    plt.close()

if __name__ == "__main__":
    main()