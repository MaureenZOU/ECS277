# Project 1

## Basic experiments

### install
python3.0 +

```
pip install opencv-python
pip install numpy
```

### config
Please adjust values under config.py
```
cfg = Config({
    # 'data_root': './data/neghip_64x64x64_uint8.raw',
    'data_root': './data/baseline_7_7_7.npy',
    'interpolation': 'linear', # linear|cubic
    'vol_size': [10, 10, 10],
    'sample_rate': 5,
    'tff': cv2.COLORMAP_JET,
    'transpose': (0,1,2),
    'illu_intense': 0.1,
    'illu_pos': np.array([20, 20, 20]),
    'illumination': True,
    'file_name': 'linear_standard_opacue2_[10,5,jet,0|1|2,0.1,20].png'
})
```

### run

```bash
python main.py
```

## Advanced experiments
