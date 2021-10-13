# @Author   : Zilin Zhang
# @Time     : 10/10/2021 12:39
# @Function :

img_width = 64
img_height = 128

USE_LABEL_SMOOTHING = True

batch_size = 128

model_path = './models/cnn_baseline.h5'

# 3368 images
query_folder = '../data/PersonReID/market1501/query'

import os
query_root = os.path.join(os.getcwd(), query_folder)
query_image_names = sorted([x for x in os.listdir(query_root) if x.endswith('.jpg')])
query_img_name, query_img_path = zip(*[(x[:-4], os.path.join(query_root, x)) for x in query_image_names])

# 25259 images
gallery_folder = '../data/PersonReId/market1501/gt_bbox'
gallery_root = os.path.join(os.getcwd(), gallery_folder)
# Remove the duplicated images from the gallery set
gallery_image_names = sorted([x for x in os.listdir(gallery_root) if x.endswith('.jpg') and x not in query_image_names])

gallery_img_name, gallery_img_path = zip(*(x[:-4], os.path.join(gallery_root, x)) for x in gallery_image_names)

from tensorflow.keras.models import load_model

if USE_LABEL_SMOOTHING:
    from utils import cross_entropy_label_smoothing
    model = load_model(model_path, custom_objects={'cross_entropy_label_smooting': cross_entropy_label_smoothing})
else:
    model = load_model(model_path)

model.summary()

