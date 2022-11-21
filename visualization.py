import json
import pprint
import numpy as np
import numpy.linalg as la

from os.path import join
import glob

import cv2
%matplotlib inline
import matplotlib.pylab as pt

with open ('/data/a2d2/cams_lidars.json', 'r') as f:
    config = json.load(f)

pprint.pprint(config)
config.keys()
config['vehicle'].keys()
config['lidars'].keys()
config['lidars']['front_left']
config['cameras'].keys()
config['cameras']['front_left']


root_path = '/data/a2d2/camera_lidar_semantic_bboxes/'
# get the list of files in lidar directory
file_names = sorted(glob.glob(join(root_path, '*/lidar/cam_front_left/*.npz')))

# select the lidar point cloud
file_name_lidar = file_names[7]

# read the lidar data
lidar_front_left = np.load(file_name_lidar)

def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                        'camera_' + \
                        file_name_image[2] + '_' + \
                        file_name_image[3] + '.png'

    return file_name_image


seq_name = file_name_lidar.split('/')[4]
print(file_name_lidar)
print(seq_name)

file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)
file_name_image = join(root_path, seq_name, 'camera/cam_front_left/', file_name_image)
image_front_left = cv2.imread(file_name_image)

image_front_left = cv2.cvtColor(image_front_left, cv2.COLOR_BGR2RGB)

pt.fig = pt.figure(figsize=(15, 15))

# display image from front left camera
pt.imshow(image_front_left)
pt.axis('off')
pt.title('front left')

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


undist_image_front_left = undistort_image(image_front_left, 'front_left')

print('diff: '+ str(np.where((image_front_left - undist_image_front_left) != 0)))
pt.fig = pt.figure(figsize=(15, 15))
pt.imshow(undist_image_front_left)
pt.axis('off')
pt.title('front left')


file_name_image_info = file_name_image.replace(".png", ".json")

def read_image_info(file_name):
    with open(file_name, 'r') as f:
        image_info = json.load(f)
        
    return image_info

image_info_front_left = read_image_info(file_name_image_info)  

pprint.pprint(image_info_front_left)


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
    rows = (lidar['row'] + 0.5).astype(np.int)
    cols = (lidar['col'] + 0.5).astype(np.int)
  
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

image = map_lidar_points_onto_image(undist_image_front_left, lidar_front_left)

pt.fig = pt.figure(figsize=(20, 20))
pt.imshow(image)
pt.axis('off')