# @Author   : Zilin Zhang
# @Time     : 09/10/2021 16:58
# @Function : PersonReID_Keras cnn baseline

img_width = 64
img_height = 128
learning_rate = 0.01

from tensorflow.keras.optimizers import SGD
optimizer = SGD(learning_rate=learning_rate)

batch_size = 128
nbr_epochs = 100

data_folder = "../data/PersonReID/market1501/bounding_box_train"

import os
data_root = os.path.join(os.getcwd(), data_folder)

image_names = sorted([x for x in os.listdir(data_root) if x.endswith('.jpg')])

# 0002_c1s1_0000451_03.jpg --> 0002: Person ID
# x = [1, 2] y = [3, 4] ==> zip(x, y) = [(1, 3), (2, 4)], zip(*[(1, 3), (2, 4)]) = [1, 2], [3, 4]
img_name, img_path = zip(*[(x[:-4], os.path.join(data_root, x)) for x in image_names])

person_id_original_list = [x[:4] for x in img_name]
nbr_person_ids = len(set(person_id_original_list))

print('Number of Person IDs: ', nbr_person_ids)

from sklearn.preprocessing import LabelEncoder

id_encoder = LabelEncoder()
id_encoder.fit(person_id_original_list)
person_id_encoded = id_encoder.transform(person_id_original_list)

# Train, Validation
from sklearn.model_selection import train_test_split
# 80%, 20%
train_img_path, val_img_path, train_person_ids, val_person_ids = train_test_split(
    img_path, person_id_encoded, test_size=0.2, random_state=2021)

# For a typical CNN, the input_shape is (batch, height, width, channel)
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# include_top=False --> 分类的层不需要了
cnn_model = MobileNetV2(include_top=False, weights='imagenet', alpha=0.5, input_shape=(img_height, img_width, 3),
                        pooling='max')
# pooling = 'max', 7x7x512 --> 1x1x512

global_pool = cnn_model.layers[-1].output
from tensorflow.keras.layers import Dense, Activation

dense = Dense(nbr_person_ids)(global_pool)

# Insert 2 FC layers: Dense + Dropout ?

softmax_output = Activation('softmax')(dense)

from tensorflow.keras.models import Model
baseline_model = Model(cnn_model.input, softmax_output)

baseline_model.summary()

USE_LABEL_SMOOTHING = True
if USE_LABEL_SMOOTHING:
    from utils import cross_entropy_label_smoothing

    baseline_model.compile(loss=cross_entropy_label_smoothing,
                           optimizer=optimizer,
                           metrics=['accuracy'])
else:
    baseline_model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('./models/cnn_baseline.h5',
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True
                             )
from utils import generator_batch

train_generator = generator_batch(
    img_path_list=train_img_path,
    img_label_list=train_person_ids,
    nbr_classes=nbr_person_ids,
    img_width=img_width,
    img_height=img_height,
    batch_size=batch_size,
    shuffle=True,
    save_to_dir=False,
    augment=True
)
val_generator = generator_batch(
    img_path_list=val_img_path,
    img_label_list=val_person_ids,
    nbr_classes=nbr_person_ids,
    img_width=img_width,
    img_height=img_height,
    batch_size=batch_size,
    shuffle=False,
    save_to_dir=False,
    augment=False
)

# validation acc: 80%+, at least > 70%
baseline_model.fit(train_generator,
                   steps_per_epoch=len(train_img_path) // batch_size,
                   validation_data=val_generator,
                   validation_steps=len(val_img_path) // batch_size,
                   verbose=1,
                   shuffle=True,  # 一个batch中进行shuffle
                   epochs=nbr_epochs,
                   callbacks=[checkpoint])
