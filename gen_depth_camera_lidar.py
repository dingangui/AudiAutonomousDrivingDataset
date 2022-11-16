"""
负责给 a2d2 Camera Lidar Sensor Fusion 数据集生成 depth 的源码

文件结构: 

A2D2
├── camera_lidar
│   ├── 20180810_150607
│   │   ├── bus
│   │   │   ├── 20180810150607_bus_signals.json
│   │   ├── camera
│   │   │   ├── cam_front_center
│   │   │   ├── cam_front_left
│   │   │   ├── cam_front_right
│   │   │   ├── cam_rear_center
│   │   │   ├── cam_side_left
│   │   │   └── cam_side_right
│   │   ├── depth
│   │   │   └── same as camera
│   │   └── lidar
│   │   │   └── same as camera
│   ├── 20190401_121727
│   │   └── same as 20180810_150607
│   ├── 20190401_145936
│   │   └── same as 20180810_150607
│   └── ssiw
"""

import json
import numpy as np
import os
from os.path import join
import mmcv
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import numpy.linalg as la

EPSILON = 1.0e-10 # norm should not be small

def create_output(vertices, colors, filename):
    """
    输出 ply 文件, 用于 meshlab 可视化
    os.makedirs(f'visualized_results/Lidar_Coordinate_Points_0', exist_ok=True)
    create_output(points_0_concat, np.ones_like(points_0_concat) * 255, f'visualized_results/Lidar_Coordinate_Points_0/{filename}.ply')
    """
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            \n
            '''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

def get_axes_of_a_view(view):
    x_axis = view['x-axis']
    y_axis = view['y-axis']
     
    x_axis_norm = la.norm(x_axis)
    y_axis_norm = la.norm(y_axis)
    
    if (x_axis_norm < EPSILON or y_axis_norm < EPSILON):
        raise ValueError("Norm of input vector(s) too small.")
        
    # normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm
    
    # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)
 
    # create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)
    
    # calculate and check y-axis and z-axis norms
    y_axis_norm = la.norm(y_axis)
    z_axis_norm = la.norm(z_axis)
    
    if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
        raise ValueError("Norm of view axis vector(s) too small.")
        
    # make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm
    
    return x_axis, y_axis, z_axis

def get_origin_of_a_view(view):
    return view['origin']

def get_transform_to_global(view):
    """
    从 sensor view 到 global view 的转换矩阵 (外参矩阵的逆)
    """
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)
    
    # get origin 
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)
    
    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis
    
    # origin
    transform_to_global[0:3, 3] = origin
    
    return transform_to_global

def get_transform_from_global(view):
    """
    从 global view 到 sensor view 的转换矩阵 (外参矩阵)
    """
   # get transform to global
    transform_to_global = get_transform_to_global(view)
    trans = np.eye(4)
    rot = np.transpose(transform_to_global[0:3, 0:3])
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])
        
    return trans

def transform_from_to(src, target):
    """
    从 source view 到 target view 的转换矩阵
    """
    transform = np.dot(get_transform_from_global(target), \
                       get_transform_to_global(src))
    
    return transform

def transform_from_to(src, target):
    """
    从 source view 到 target view 的转换矩阵
    src_view = config['cameras']['front_left']['view']
    target_view = config['cameras']['front_right']['view']
    """
    transform = np.dot(get_transform_from_global(target), \
                       get_transform_to_global(src))
    
    return transform


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
    """
    rgb 图片去畸变    
    """
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
    rows = (lidar['pcloud_attr.row'] + 0.5).astype(np.int32)
    cols = (lidar['pcloud_attr.col'] + 0.5).astype(np.int32)
  
    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['pcloud_attr.distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['pcloud_attr.distance'])

    # get distances
    distances = lidar['pcloud_attr.distance']  
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
        return 256 * 256 * color[:, :, 0] + 256 * color[:, :, 1] + color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def gen_depth(file_name_lidar):
    lidar = np.load(file_name_lidar)
    keys =  list(lidar.keys()) 
    if('pcloud_attr.row' not in keys or 'pcloud_attr.col' not in keys or  'pcloud_attr.depth' not in keys):
        print('can\'t find', file_name_lidar , 'rows, cols or depth')
        return

    seq_name = file_name_lidar.split('/')[4] # 20180810_150607 或 20190401_121727 或 20190401_145936, 共 3 个
    cam_name = file_name_lidar.split('/')[6] # cam_front_center 或 cam_front_left ... 或 cam_side_right, 共 6 个
    view_name = cam_name.split('cam_')[1] # front_center 或 front_left ... 或 side_right, 共 6 个
    image_name = extract_image_file_name_from_lidar_file_name(file_name_lidar) # 原始数据集图片的文件名
    image_path = join(root_path, seq_name, 'camera', cam_name, image_name) # 原始数据集图片的绝对路径
    depth_path = image_path.replace('/camera/','/depth/') # 保存 depth map 的绝对路径

    imgrgb = cv2.imread(image_path) # 数据集内原始图片
    undist_imgrgb = undistort_image(imgrgb, cam_name) # 数据集去畸变后的 rgb 图片
    
    # ------------------------------------------------------------------------------ #
    # 可视化 img
    os.makedirs(f'visualized_results/dist_imgrgb/', exist_ok=True)
    cv2.imwrite(f'visualized_results/dist_imgrgb/{image_name}', imgrgb)
    os.makedirs(f'visualized_results/undist_imgrgb/', exist_ok=True)
    cv2.imwrite(f'visualized_results/undist_imgrgb/{image_name}', undist_imgrgb)
    # ------------------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------------------ #
    # 基于 a2d2 的 depth 可视化方法
    mapped_img = map_lidar_points_onto_image(undist_imgrgb, lidar, 3)
    os.makedirs(f'visualized_results/scat_points_in_a2d2_method/', exist_ok=True)
    cv2.imwrite(f'visualized_results/scat_points_in_a2d2_method/{image_name}', mapped_img)
    # ------------------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------------------ #
    # 基于 mmdet3d 的 depth 可视化方法
    rows = (lidar['pcloud_attr.row'] + 0.5).astype(np.int32)
    cols = (lidar['pcloud_attr.col'] + 0.5).astype(np.int32)
    colors = cm.jet(lidar['pcloud_attr.depth'] / lidar['pcloud_attr.depth'].max())
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    fig.tight_layout()
    ax.axis('off')
    undist_imgrgb_plt = cv2.cvtColor(undist_imgrgb, cv2.COLOR_BGR2RGB) # matplotlib 解析 rgb 方式不一样, 需要转换
    ax.imshow(undist_imgrgb_plt)
    ax.scatter(cols, rows, c=colors, s=3)
    print('min depth: ' + str(lidar['pcloud_attr.depth'].min()) + ', max depth: ' + str(lidar['pcloud_attr.depth'].max()))
    os.makedirs(f'visualized_results/scat_points_in_mmdet3d_method/', exist_ok=True)
    plt.savefig(f'visualized_results/scat_points_in_mmdet3d_method/{image_name}', bbox_inches='tight',pad_inches = 0)
    # ------------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------------ #
    # lidar view 的 lidar['pcloud_points'], camera view 的 lidar['pcloud_points'] 和 projected points 可视化
    
    # 1. lidar view 的激光雷达点云可视化
    pcloud_points_name = file_name_lidar.split('/')[7].replace('npz', 'ply') # 保存 lidar 点云的文件名, 用于在 meshlab 可视化
    lidar_view_pcloud_points = lidar['pcloud_points']
    os.makedirs(f'visualized_results/lidar_view_pcloud_points/', exist_ok=True)
    create_output(lidar_view_pcloud_points, np.ones_like(lidar_view_pcloud_points) * 255, f'visualized_results/lidar_view_pcloud_points/{pcloud_points_name}')
    
    # 2.  camera view 的激光雷达点云可视化
    lidar_view = config['lidars'][f'{view_name}']['view'] # 当前视角的激光雷达视图
    camera_view = config['cameras'][f'{view_name}']['view'] # 当前视角的相机视图
    lidar_to_camara_transform = transform_from_to(lidar_view, camera_view) # 激光雷达视角转相机视角转换矩阵
    ones = np.ones((lidar_view_pcloud_points.shape[0], 1)) 
    homo_lidar_view_pcloud_points = np.concatenate((lidar_view_pcloud_points, ones), axis=1) # 激光雷达坐标系下的坐标转齐次坐标 (x,y,z) -> (x,y,z,1)
    camera_view_pcloud_points = (lidar_to_camara_transform @ homo_lidar_view_pcloud_points.T).T[..., :3] # 相机坐标系下的点云坐标
    os.makedirs(f'visualized_results/camera_view_pcloud_points/', exist_ok=True)
    create_output(camera_view_pcloud_points , np.ones_like(camera_view_pcloud_points) * 255, f'visualized_results/camera_view_pcloud_points/{pcloud_points_name}')

    # 3. projected points 可视化
    projected_points = np.dstack((camera_view_pcloud_points[..., 0], camera_view_pcloud_points[..., 1], lidar['pcloud_attr.depth'])).squeeze()
    os.makedirs(f'visualized_results/projected_points/', exist_ok=True)
    create_output(projected_points , np.ones_like(projected_points) * 255, f'visualized_results/projected_points/{pcloud_points_name}')
    # ------------------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------------------ #
    # 保存 depth 
    # 全 0 背景 + 点云深度 + scale 200 倍 + 300 米以上做截断
    # rows = (lidar['pcloud_attr.row'] + 0.5).astype(np.int32)
    # cols = (lidar['pcloud_attr.col'] + 0.5).astype(np.int32)
    # z_depth = lidar['pcloud_attr.depth']
    # h, w = imgrgb.shape[0], imgrgb.shape[1]
    # depth = np.zeros((h, w))
    # depth[rows, cols] = z_depth
    # depth *= 200.
    # depth[np.where(depth > 200. * 300.)] = 0
    # os.makedirs(depth_path.split(image_name)[0], exist_ok=True)
    # mmcv.imwrite(depth.astype(np.uint16), depth_path)
    # ------------------------------------------------------------------------------ #
   
    # ------------------------------------------------------------------------------ #
    # 读取保存在文件中的 depth 进行可视化
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
    # os.makedirs(f'visualized_results/scat_points_loaded_from_file/', exist_ok=True)
    # plt.savefig(f'visualized_results/scat_points_loaded_from_file/{image_name}', bbox_inches='tight',pad_inches = 0)
    # ------------------------------------------------------------------------------ #
    

if __name__ == '__main__':
    root_path = '/data/a2d2/camera_lidar/'

    # 获得相机的参数 cam matrix
    with open ('/data/a2d2/cams_lidars.json', 'r') as f:
        config = json.load(f)

    # get the list of files in lidar directory
    """
    - 根据 lidar 的文件名列表确定需要转换 depth 的列表
    - 每次需要访问磁盘获取 lidar 的文件名列表
    - 为了提高效率, 可以将获取到的列表存到文件中
    - 每次转换直接读取文件即可
    - camera_lidar_filenames.json: 保存 root_path 下所有 lidar 的文件名
    """
    # lidar_file_names = sorted(glob.glob(join(root_path, '*/lidar/*/*.npz')))
    # with open('camera_lidar_filenames.json', 'w') as outfile:
    #     json.dump(lidar_file_names, outfile, indent = 4)
    lidar_file_names = json.load(open('camera_lidar_filenames.json', 'r'))

    # 多线程方法
    # mmcv.track_parallel_progress(gen_depth,lidar_file_names, 64)

    # 测试用
    for file_name_lidar in lidar_file_names[0:30000:5000]:
        gen_depth(file_name_lidar)