import numpy as np
import cv2
import math
import time
from config import get_cfg

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

    return data

def read_data(file_name, volsize=None):
    if '.npy' in file_name:
        data = np.load(file_name)
    elif '.raw'  in file_name:
        data = load_raw(file_name, volsize)

    if data is not None:
        print('Finish loading data', file_name, '...')
    return data

def get_test_volume(volume_size):
    print('get test volume...')
    x = np.arange(volume_size[0]).reshape(volume_size[0], 1, 1).astype(np.float)
    y = np.arange(volume_size[1]).reshape(1, volume_size[1], 1).astype(np.float)
    z = np.arange(volume_size[2]).reshape(1, 1, volume_size[2])
    # z = np.ones((1, 1, volume_size[2]))

    # volume = (x**2+(y/2)**2+z**2+np.log(x+1)*np.sin(y))
    # volume = (x**2+(y/2)**2+z**2+np.log(x+1)*np.cos(y))
    volume = (x**2+y**2+z**2)

    return volume

def f_triLinear(position, param):
    '''
    position is a 3d array, param is 8d array
    '''
    x = position[0]
    y = position[1]
    z = position[2]

    return np.sum(param*np.array([1, x, y, z, x*y, x*z, y*z, x*y*z]))

def g_triLinear(position, param):
    '''
    position is a 3d array, param is 8d array
    '''
    x = position[0]
    y = position[1]
    z = position[2]

    g_x = param[1] + param[4]*y + param[5]*z + param[7]*y*z
    g_y = param[2] + param[4]*x + param[6]*z + param[7]*x*z
    g_z = param[3] + param[5]*x + param[6]*y + param[7]*x*y

    return np.array([g_x,g_y,g_z])

def get_triLinear_param(values):
    M = np.array([
        [1,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0],
        [1,0,1,0,0,0,0,0],
        [1,1,1,0,1,0,0,0],
        [1,0,0,1,0,0,0,0],
        [1,1,0,1,0,1,0,0],
        [1,0,1,1,0,0,1,0],
        [1,1,1,1,1,1,1,1]
    ])
    param = np.linalg.solve(M, values)
    return param

# def get_linear_intensity(volume, v_x, v_y, v_z):
#     I_000 = volume[v_x, v_y, v_z]
#     I_100 = volume[v_x+1, v_y, v_z]
#     I_010 = volume[v_x, v_y+1, v_z]
#     I_110 = volume[v_x+1, v_y+1, v_z]
#     I_001 = volume[v_x, v_y, v_z+1]
#     I_101 = volume[v_x+1, v_y, v_z+1]
#     I_011 = volume[v_x, v_y+1, v_z+1]
#     I_111 = volume[v_x+1, v_y+1, v_z+1]
    
#     values = np.array([I_000, I_001, I_010, I_110, I_001, I_101, I_011, I_111])
#     return values

def get_linear_intensity(volume, v_x, v_y, v_z):
    f_values = np.zeros([2,2,2])
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                f_values[i,j,k] = volume[v_x+i, v_y+j, v_z+k]
    
    return f_values

def fg_triLinear(position, values):
    x,y,z = position

    map_d = {}
    map_d[0] = -1
    map_d[1] = 1

    f_x_y_z = 0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                f_x_y_z += values[i,j,k]*((1-i) + map_d[i]*x) * ((1-j)+map_d[j]*y) * ((1-k)+map_d[k]*z)


    g_x = 0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                # print('x', values[i,j,k], (map_d[i]), ((1-j)+map_d[j]*y), ((1-k) + map_d[k]*z))
                g_x += values[i,j,k] * (map_d[i]) * ((1-j)+map_d[j]*y) * ((1-k) + map_d[k]*z)

    g_y = 0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                # print('y', values[i,j,k], ((1-i)+map_d[i]*x), (map_d[j]), ((1-k) + map_d[k]*z))
                g_y += values[i,j,k]* ((1-i)+map_d[i]*x) * (map_d[j]) * ((1-k)+map_d[k]*z)

    g_z = 0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                g_z += values[i,j,k]*((1-i) + map_d[i]*x)*((1-j) + map_d[j]*y)*(map_d[k])

    # print(position, 'position')
    # print(g_x, g_y, g_z, 'gradient')
    # print(values, 'values')

    return f_x_y_z, np.array([g_x, g_y, g_z])


def get_cubic_intensity(volume, v_x, v_y, v_z):
    v_x += 1
    v_y += 1
    v_z += 1

    I_111 = volume[v_x, v_y, v_z]
    I_011 = volume[v_x-1, v_y, v_z]
    I_211 = volume[v_x+1, v_y, v_z]
    I_101 = volume[v_x, v_y-1, v_z]
    I_121 = volume[v_x, v_y+1, v_z]
    I_110 = volume[v_x, v_y, v_z-1]
    I_112 = volume[v_x, v_y, v_z+1]
    
    values = np.array([I_111, I_011, I_211, I_101, I_121, I_110, I_112])
    return values

def get_cubic_gradient(values):
    I_111, I_011, I_211, I_101, I_121, I_110, I_112 = values
    g_x = 0.5*(I_211 - I_011)
    g_y = 0.5*(I_121 - I_101)
    g_z = 0.5*(I_112 - I_110)
    return np.array([g_x, g_y, g_z])

def get_cubic_box(f_values, g_values):
    box = np.zeros((4,4,4))
    f = np.zeros((2,2,2))

    for p in range(0, 2):
        for q in range(0, 2):
            for t in range(0, 2):
                f = f_values[p,q,t][0]
                g_x = g_values[p,q,t][0]
                g_y = g_values[p,q,t][1]
                g_z = g_values[p,q,t][2]

                for i in range(0, 2):
                    for j in range(0, 2):
                        for k in range(0, 2):
                            box[p*2+i,q*2+j,t*2+k] = f + (p*2+i-p*3)*(1/3)*g_x + (q*2+j-q*3)*(1/3)*g_y + (t*2+k-t*3)*(1/3)*g_z

    return box


def fg_triCubic(position, box):
    B_ix = np.zeros((4))
    B_jy = np.zeros((4))
    B_kz = np.zeros((4))
    x,y,z = position

    for i in range(0, 4):
        B_ix[i] = (math.factorial(3)*1.0/(math.factorial(i)*math.factorial(3-i)))*((1-x)**(3-i))*(x**i)
        B_jy[i] = (math.factorial(3)*1.0/(math.factorial(i)*math.factorial(3-i)))*((1-y)**(3-i))*(y**i)
        B_kz[i] = (math.factorial(3)*1.0/(math.factorial(i)*math.factorial(3-i)))*((1-z)**(3-i))*(z**i)


    B_2_ix = np.zeros((3))
    B_2_jy = np.zeros((3))
    B_2_kz = np.zeros((3))
    for i in range(0, 3):
        B_2_ix[i] = (math.factorial(2)*1.0/(math.factorial(i)*math.factorial(2-i)))*((1-x)**(2-i))*(x**i)
        B_2_jy[i] = (math.factorial(2)*1.0/(math.factorial(i)*math.factorial(2-i)))*((1-y)**(2-i))*(y**i)
        B_2_kz[i] = (math.factorial(2)*1.0/(math.factorial(i)*math.factorial(2-i)))*((1-z)**(2-i))*(z**i)


    f_triCubic = 0
    for k in range(0, 4):
        for j in range(0, 4):
            for i in range(0, 4):
                f_triCubic += box[i,j,k]*(B_ix[i])*(B_jy[j])*(B_kz[k])

    g_x = 0
    for k in range(0, 4):
        for j in range(0, 4):
            for i in range(0, 3):
                g_x += 3*(box[i+1,j,k] - box[i,j,k])*(B_2_ix[i])*(B_jy[j])*(B_kz[k])

    g_y = 0
    for k in range(0, 4):
        for j in range(0, 3):
            for i in range(0, 4):
                g_y += 3*(box[i,j+1,k] - box[i,j,k])*(B_ix[i])*(B_2_jy[j])*(B_kz[k])

    g_z = 0
    for k in range(0, 3):
        for j in range(0, 4):
            for i in range(0, 4):
                g_z += 3*(box[i,j,k+1] - box[i,j,k])*(B_ix[i])*(B_jy[j])*(B_2_kz[k])

    return f_triCubic, np.array([g_x, g_y, g_z])
    # return np.average(box), np.array([g_x, g_y, g_z])

def ray_casting(vector, max_volume):
    # bgc is black
    opaque = ((vector / max_volume*1.0)/2.0) + 0.25
    opaque = np.abs(np.cos(opaque*20)*np.sin(opaque*20))
    base = np.flip(1-opaque)
    return np.sum(np.flip(np.cumprod(base)/base) * vector * opaque)

# def ray_casting(vector, max_volume):
#     return np.average(vector)

def triCubic(cfg):
    # file_name = cfg.data_root
    # pad_volume = read_data(file_name, cfg.vol_size).astype(np.float)
    # h,w,d = pad_volume.shape
    # pad_volume = np.transpose(pad_volume, cfg.transpose)
    # volume = pad_volume[1:h-1,1:w-1,1:d-1]

    pad_volume = get_test_volume(cfg.vol_size)
    h,w,d = pad_volume.shape
    pad_volume = np.transpose(pad_volume, cfg.transpose)
    volume = pad_volume[1:h-1,1:w-1,1:d-1]
    
    max_volume = volume.max()
    sample_rate = cfg.sample_rate

    v_w, v_h, v_d = volume.shape
    w_w, w_h, w_d = (np.array(volume.shape)-1)*sample_rate
    display_pannel = np.zeros((w_h, w_w))

    volume_params = [[[None for k in range(0, v_d)] for j in range(0, v_w)] for i in range(0, v_h)]

    for i in range(0, w_h):
        print(i)
        for j in range(0, w_w):
            v_y = i // sample_rate
            v_x = j // sample_rate
            vector = []
            for k in range(0, w_d):
                v_z = k // sample_rate

                b_x = (j % sample_rate) / sample_rate*1.0
                b_y = (i % sample_rate) / sample_rate*1.0
                b_z = (k % sample_rate) / sample_rate*1.0
                
                if volume_params[v_y][v_x][v_z] is None:
                    f_values = [[[0 for i in range(0,2)] for i in range(0, 2)] for i in range(0, 2)]
                    g_values = [[[0 for i in range(0,2)] for i in range(0, 2)] for i in range(0, 2)]

                    for p in range(0, 2):
                        for q in range(0, 2):
                            for t in range(0, 2):
                                f_values[p][q][t] = get_cubic_intensity(pad_volume, v_x+p, v_y+q, v_z+t)
                                g_values[p][q][t] = get_cubic_gradient(f_values[p][q][t])

                    box = get_cubic_box(np.array(f_values), np.array(g_values))
                    volume_params[v_y][v_x][v_z] = box
                else:
                    box = volume_params[v_y][v_x][v_z]

                # print(box)
                f_x_y_z, g_x_y_z = fg_triCubic(np.array([b_x, b_y, b_z]), box)

                # add illumination
                if cfg.illumination:
                    vec_n = g_x_y_z
                    vec_l = cfg.illu_pos - np.array([b_x, b_y, b_z])
                    f_x_y_z = (f_x_y_z*((vec_n*vec_l).sum())*cfg.illu_intense*1.0)/np.linalg.norm(vec_l)

                vector.append(f_x_y_z)
            
            display_pannel[i,j] = ray_casting(np.array(vector), max_volume)

    display_pannel = display_pannel/display_pannel.max()
    # display_pannel = display_pannel/500.0
    display_pannel = np.array(display_pannel * 255, dtype = np.uint8)
    display_pannel = cv2.applyColorMap(display_pannel, cfg.tff)

    # file_name = [cfg.data_root.split('/')[-1][:-4], cfg.interpolation.__repr__(), cfg.vol_size.__repr__(), cfg.sample_rate.__repr__(), cfg.tff.__repr__(), cfg.transpose.__repr__()]
    # file_name = '_'.join(file_name) + '.png'
    file_name = cfg.file_name
    cv2.imwrite(file_name, display_pannel)

def triLinear(cfg):
    # file_name = cfg.data_root
    # volume = read_data(file_name, cfg.vol_size).astype(np.float)
    # volume = np.transpose(volume, cfg.transpose)

    volume = get_test_volume(cfg.vol_size)
    volume = np.transpose(volume, cfg.transpose)
 
    # print(volume)

    # display_pannel = np.sum(volume, axis=2)
    # display_pannel = display_pannel/display_pannel.max()
    # display_pannel = np.array(display_pannel * 255, dtype = np.uint8)
    # display_pannel = cv2.applyColorMap(display_pannel, cfg.tff)
    # cv2.imwrite('out.png', display_pannel)
    # exit()

    # volume = volume[50:53,50:53,50:60]
    max_volume = volume.max()
    sample_rate = cfg.sample_rate

    v_w, v_h, v_d = volume.shape
    w_w, w_h, w_d = (np.array(volume.shape)-1)*sample_rate
    display_pannel = np.zeros((w_h, w_w))

    volume_params = [[[None for k in range(0, v_d)] for j in range(0, v_w)] for i in range(0, v_h)]

    for i in range(0, w_h):
        print(i)
        for j in range(0, w_w):
            v_y = i // sample_rate
            v_x = j // sample_rate
            vector = []
            for k in range(0, w_d):
                v_z = k // sample_rate

                b_x = (j % sample_rate) / sample_rate*1.0
                b_y = (i % sample_rate) / sample_rate*1.0
                b_z = (k % sample_rate) / sample_rate*1.0
                
                if volume_params[v_y][v_x][v_z] is None:
                    values = get_linear_intensity(volume, v_x, v_y, v_z)
                    # params = get_triLinear_param(values)

                    volume_params[v_y][v_x][v_z] = values
                else:
                    values = volume_params[v_y][v_x][v_z]

                f_x_y_z, g_x_y_z = fg_triLinear(np.array([b_x, b_y, b_z]), values)
                # print(g_x_y_z)
                # f_x_y_z = f_triLinear(np.array([b_x, b_y, b_z]), params)
                # g_x_y_z = g_triLinear(np.array([b_x, b_y, b_z]), params)

                # add illumination
                if cfg.illumination:
                    vec_n = g_x_y_z
                    vec_l = cfg.illu_pos - np.array([b_x, b_y, b_z])
                    f_x_y_z = (f_x_y_z*((vec_n*vec_l).sum())*cfg.illu_intense*1.0)/np.linalg.norm(vec_l)

                vector.append(f_x_y_z)
            
            display_pannel[i,j] = ray_casting(np.array(vector), max_volume)
    
    display_pannel = display_pannel/display_pannel.max()
    # display_pannel = display_pannel/650.0
    display_pannel = np.array(display_pannel * 255, dtype = np.uint8)
    display_pannel = cv2.applyColorMap(display_pannel, cfg.tff)

    # file_name = [cfg.data_root.split('/')[-1][:-4], cfg.interpolation.__repr__(), cfg.vol_size.__repr__(), cfg.sample_rate.__repr__(), cfg.tff.__repr__(), cfg.transpose.__repr__()]
    # file_name = '_'.join(file_name) + '.png'
    file_name = cfg.file_name
    cv2.imwrite(file_name, display_pannel)

def main():
    cfg = get_cfg()

    if cfg.interpolation == 'linear':
        print('Running Linear interpolation ...')
        triLinear(cfg)
    elif cfg.interpolation == 'cubic':
        print('Running Cubic interpolation ...')
        triCubic(cfg)
    else:
        assert False, "Interpolation function not implemented."

if __name__ == '__main__':
    # triLinear()
    main()