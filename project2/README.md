# Project 1

## Basic experiments

### install
python3.0 +

```
pip install opencv-python
pip install numpy
pip install torch==1.2.0
```

### config
Please adjust values under config.py
```
cfg = Config({
    'data_root': './data/raw/neghip_64x64x64_uint8.raw',
    'vol_size': [64, 64, 64],
    'file_name': 'local_shaper_s2' + '_k10',
    'method': 'local_shaper_s3',
    'knn': 5,
    'sample_num': 4000
})
```

### run

```bash
python main.py
```
