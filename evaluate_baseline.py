# @Author   : Zilin Zhang
# @Time     : 10/10/2021 12:39
# @Function :

img_width = 64
img_height = 128

USE_LABEL_SMOOTHING = True

batch_size = 128

model_path = './models/MobileNetV2_baseline.h5'

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

gallery_img_name, gallery_img_path = zip(*[(x[:-4], os.path.join(gallery_root, x)) for x in gallery_image_names])

from tensorflow.keras.models import load_model

if USE_LABEL_SMOOTHING:
    from utils import cross_entropy_label_smoothing
    model = load_model(model_path, custom_objects={'cross_entropy_label_smoothing': cross_entropy_label_smoothing})
else:
    model = load_model(model_path)

model.summary()

# 1280-Dim
dense_features = model.get_layer('global_max_pooling2d').output

from tensorflow.keras.models import Model

model_extract_features = Model(model.input, dense_features)

from tensorflow.keras.optimizers import SGD
optimizer = SGD(learning_rate=0.01)

model_extract_features.compile(loss='categorical_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])

from utils import generator_batch_predict
query_generator = generator_batch_predict(img_path_list=query_img_path,
                                          img_width=img_width,
                                          img_height=img_height,
                                          batch_size=batch_size)

query_features = model_extract_features.predict(query_generator, verbose=1,
                                                steps=len(query_img_path) // batch_size + 1
                                                )
# shape: (3368, 1280)
print('query features shape:', query_features.shape)

from sklearn.preprocessing import normalize
query_features = normalize(query_features, norm='l2')


gallery_generator = generator_batch_predict(img_path_list=gallery_img_path,
                                            img_width=img_width,
                                            img_height=img_height,
                                            batch_size=batch_size)

gallery_features = model_extract_features.predict(gallery_generator, verbose=1,
                                                  steps=len(gallery_img_path) // batch_size + 1
                                                  )
# shape: (25259, 1280)
print('gallery features shape:', gallery_features.shape)

gallery_features = normalize(gallery_features, norm='l2')

# For each query (1280-D feature vector), KNN search
# q = 1280 x 1, G = 25259 x 1280 ==> G x q --> 25259 x 1
# Q = 1280 x N, G = 25259 x 1280 ==> G x Q --> 25259 x N
import numpy as np
# (3368 x 1280) x (1280 x 25259) ==> (3368 x 25259)
distance_query_gallery = 1 - (np.dot(query_features, np.transpose(gallery_features))
                              / (np.linalg.norm(query_features) * np.linalg.norm(gallery_features)))

idx_list = np.argsort(distance_query_gallery, axis=1)

print("idx_list shape:", idx_list.shape)

top_1_acc = 0
for i, query_name in enumerate(query_img_name):
    # query_img_name is a list of '0001_c1s1_001051_00' ...
    query_id = int(query_name[:4])
    ranking_id = int(gallery_img_name[int(idx_list[i][0])][:4])
    if query_id == ranking_id:
        top_1_acc += 1

# Top-1 acc > 70%
# 27/27 [==============================] - 4s 110ms/step
# query features shape: (3368, 1280)
# 172/172 [==============================] - 21s 119ms/step
# gallery features shape: (21891, 1280)
# idx_list shape: (3368, 21891)
# top_1_acc: 0.7102137767220903
print("top_1_acc:", top_1_acc / len(query_img_name))
