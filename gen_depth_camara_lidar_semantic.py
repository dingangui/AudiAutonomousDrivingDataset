"""
负责给 a2d2 Camera Lidar Sensor Fusion 数据集生成 depth 的源码

文件结构: 

A2D2
├── camera_lidar_semantic
│   ├── 20180807_145028
│   │   ├── camera
│   │   ├── depth
│   │   ├── label
│   │   ├── lidar
│   │   ├── ssiw
│   │   └── undist_camera
│   ├── 20180810_142822
│   │   ├── same as 20180807_145028
│   ├── ...（共23个序列）
│   ├── 20181204_170238
│   │   ├── camera 
│   │   ├── depth
│   │   ├── label
│   │   ├── lidar
│   │   ├── ssiw
│   │   └── undist_camera
│   └── 20181204_191844 (无 lidar)
│       ├── camera
│       ├── label
│       └── ssiw
"""

import json
import pprint
import numpy as np
import numpy.linalg as la
import os
from os.path import join
import glob
import mmcv
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                        'camera_' + \
                        file_name_image[2] + '_' + \
                        file_name_image[3] + '.png'

    return file_name_image

def undistort_image(image, cam_name):
    if cam_name in ['front_left', 'front_center', \
                    'front_right', 'side_left', \
                    'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = \
                  np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = \
                  np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = \
                  np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']
        
        if (lens == 'Fisheye'):
            return cv2.fisheye.undistortImage(image, intr_mat_dist,\
                                      D=dist_parms, Knew=intr_mat_undist)
        elif (lens == 'Telecam'):
            return cv2.undistort(image, intr_mat_dist, \
                      distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image

def read_image_info(file_name):
    with open(file_name, 'r') as f:
        image_info = json.load(f)
        
    return image_info

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    image = np.copy(image_orig)
    
    # get rows and cols
    rows = (lidar['row'] + 0.5).astype(np.int32)
    cols = (lidar['col'] + 0.5).astype(np.int32)
  
    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['distance'])

    # get distances
    distances = lidar['distance']  
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                        np.sqrt(pixel_opacity), 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = \
                (1. - pixel_opacity) * \
                np.multiply(image[pixel_rows, pixel_cols, :], \
                colours[i]) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)

def rgb2id(color):
    """
    rewrite from panopticapi.utils.rgb2id
    origin: color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    current: 256 * 256 * color[:, :, 0] + 256 * color[:, :, 1] + color[:, :, 2]
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def gen_depth(file_name_lidar):

    lidar = np.load(file_name_lidar)
    keys =  list(lidar.keys())
    if('row' not in keys or 'col' not in keys or  'depth' not in keys):
        print('can\'t find', file_name_lidar , 'rows, cols or depth')
        return
    
    seq_name = file_name_lidar.split('/')[4]
    cam_name = file_name_lidar.split('/')[6]

    image_name = extract_image_file_name_from_lidar_file_name(file_name_lidar) # 原始数据集图片的文件名
    image_path = join(root_path, seq_name, 'camera', cam_name, image_name) # 原始数据集图片的绝对路径
    depth_path = image_path.replace('/camera/','/depth/') # 保存 depth map 的绝对路径
    lable_path = image_path.replace('/camera/','/label/').replace('_camera_','_label_') # 原始数据集 mask label 的绝对路径
    label_name = image_name.replace('_camera_','_label_') # 原始数据集 mask label 的文件名
    ssiw_path = lable_path.replace('/label/','/ssiw/') # 保存 ssiw mask 的绝对路径, 含文件名
    undist_img_path = image_path.replace('/camera/','/undist_camera/') # 保存 undist_img 的绝对路径, 含文件名
    
    cam_name = cam_name.split('cam_')[1]
    imgrgb = cv2.imread(image_path) # 数据集内图片, bgr编码
    # imgrgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB) # 数据集 rgb 图片
    undist_imgrgb = undistort_image(imgrgb, cam_name) # 数据集图片去畸变
    os.makedirs(undist_img_path.split(image_name)[0], exist_ok=True)
    cv2.imwrite(undist_img_path, undist_imgrgb)
    
    # depth a2d2 on undist_imgrgb
    # mapped_img = map_lidar_points_onto_image(undist_imgrgb, lidar, 3)

    # lblrgb = cv2.imread(lable_path) # 数据集内 mask, bgr编码
    # lblrgb = cv2.cvtColor(lblbgr, cv2.COLOR_BGR2RGB) # 数据集 mask 的 rgb 图片
    # undist_lblrgb = undistort_image(lblrgb, cam_name) # 数据集 mask 去畸变

    # ssiw 
    # rbg mask 转 24 bit 
    # undist_lblid = rgb2id(lblrgb)
    # for i, color in enumerate(color_ints):
    #     undist_lblid[np.where(undist_lblid == color)] = i
    # undist_lblid[np.where(undist_lblid == 51)] = 142
    # os.makedirs(ssiw_path.split(label_name)[0], exist_ok=True)
    # cv2.imwrite(ssiw_path, undist_lblid)
    
    
    # depth 
    # 全 0 背景 + 点云深度 + scale 200 倍 + 300 米以上做截断
    rows = (lidar['row'] + 0.5).astype(np.int32)
    cols = (lidar['col'] + 0.5).astype(np.int32)
    z_depth = lidar['depth']
    h, w = undist_imgrgb.shape[0], undist_imgrgb.shape[1]
    depth = np.zeros((h, w))
    depth[rows, cols] = z_depth
    depth *= 200.
    depth[np.where(depth > 200. * 300.)] = 0
    os.makedirs(depth_path.split(image_name)[0], exist_ok=True)
    mmcv.imwrite(depth.astype(np.uint16), depth_path)
     
    # os.makedirs(f'visualized_results/imgrgb/', exist_ok=True)
    # cv2.imwrite(f'visualized_results/imgrgb/{image_name}', imgrgb)
    
    # os.makedirs(f'visualized_results/undist_imgrgb/', exist_ok=True)
    # cv2.imwrite(f'visualized_results/undist_imgrgb/{image_name}', undist_imgrgb)
    
    # os.makedirs(f'visualized_results/mapped_img/', exist_ok=True)
    # cv2.imwrite(f'visualized_results/mapped_img/{image_name}', mapped_img)
 
    # os.makedirs(f'visualized_results/label/', exist_ok=True)
    # cv2.imwrite(f'visualized_results/label/{label_name}', lblrgb)
    
    # os.makedirs(f'visualized_results/undist_label/', exist_ok=True)
    # cv2.imwrite(f'visualized_results/undist_label/{label_name}', undist_lblrgb)
    
    # os.makedirs(f'visualized_results/ssiw/', exist_ok=True)
    # cv2.imwrite(f'visualized_results/ssiw/{label_name}', undist_lblid)
    
    
    ########################################################################
    # 可视化生成的 depth
    # depth_load =  cv2.imread(depth_path, -1)
    # rows= np.where(depth_load > 0)[1]
    # cols= np.where(depth_load > 0)[0]
    # colors = cm.jet(depth_load / np.max(depth_load))[np.where(depth_load > 0)]
    # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    # fig.tight_layout()
    # ax.axis('off')
    # ax.imshow(undist_imgrgb)
    # ax.scatter(rows, cols, color=colors, s=3)
    
    # print('min depth: ' + str(depth_load.min()) + ', max depth: ' + str(depth_load.max()))
    # os.makedirs(f'visualized_results/ax_new/', exist_ok=True)
    # plt.savefig(f'visualized_results/ax_new/{image_name}', bbox_inches='tight',pad_inches = 0)
    #######################################################################
    
if __name__ == '__main__':
    
    # cam matrix 
    with open ('/data/a2d2/cams_lidars.json', 'r') as f:
        config = json.load(f)
    
    # 保存相机内参
    # anno_dict = {}
    # for cam_name in ['front_left', 'front_center', \
    #                 'front_right', 'side_left', \
    #                 'side_right', 'rear_center']:
    #     anno_dict[f'{cam_name}'] =config["cameras"][cam_name]["CamMatrix"]
    # with open('/data/a2d2/camera_intrinsic.json', 'w') as outfile:
    #     json.dump(anno_dict, outfile, indent = 4)
    
    root_path = '/data/a2d2/camera_lidar_semantic/'
    # get the list of files in lidar directory
    # lidar_filenames = sorted(glob.glob(join(root_path, '*/lidar/*/*.npz')))
    # with open('camera_lidar_semantic_filenames.json', 'w') as outfile:
    #     json.dump(lidar_filenames, outfile, indent = 4)

    # class_list = json.load(open('class_list.json', 'r'))
    # color_ints = [int(i[1:],16) for i in class_list.keys()]
    
    # 多线程方法
    # lidar_file_names = json.load(open('camera_lidar_semantic_filenames.json', 'r'))
    # mmcv.track_parallel_progress(gen_depth,lidar_file_names, 64)

    # 测试用
    # for file_name_lidar in lidar_file_names[0:5000:1000]:
    #     gen_depth(file_name_lidar)
    # gen_depth('/data/a2d2/camera_lidar_semantic/20181107_132300/lidar/cam_front_center/20181107132300_lidar_frontcenter_000000050.npz')
    
    """
    - 统计 depth 路径下的文件列表
    - 以供转 ffrecord dataset 使用
    """
    lidar_file_names = sorted(glob.glob(join(root_path, '*/depth/*/*.png')))
    with open('/data/a2d2/camera_lidar_semantic/camera_lidar_semantic_depth_filenames.json', 'w') as outfile:
        json.dump(lidar_file_names, outfile, indent = 4)
