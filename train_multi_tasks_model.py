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

from tensorflow.keras.layers import Input

# anchor_input = Input((img_height, img_width, 3), name='anchor_input')
# positive_input = Input((img_height, img_width, 3), name='positive_input')
# negative_input = Input((img_height, img_width, 3), name='negative_input')
#
# anchor_embedding = cnn_model(anchor_input)
# positive_embedding = cnn_model(positive_input)
# negative_embedding = cnn_model(negative_input)
#
# from tensorflow.keras.layers import concatenate
# merged_vector = concatenate([anchor_embedding, positive_embedding, negative_embedding],
#                             axis=-1, name='triplet')

# triplet_model = Model(inputs=[anchor_input, positive_input, negative_input],
#                       ouputs=merged_vector
#                       )

global_pool = cnn_model.layers[-1].output
from tensorflow.keras.layers import Dense, Activation, Lambda
from tensorflow.keras import backend as K

dense_normalized = Lambda(lambda x: K.l2_normalize(x, axis=1), name='triplet')(global_pool)

dense = Dense(nbr_person_ids)(global_pool)

softmax_output = Activation('softmax')(dense)

from tensorflow.keras.models import Model
triplet_model = Model(inputs=cnn_model.input, outputs=[softmax_output, dense_normalized])

triplet_model.summary()

import tensorflow_addons as tfa
triplet_semi_hard_loss = tfa.losses.TripletSemiHardLoss(margin=0.3)

USE_LABEL_SMOOTHING = True
if USE_LABEL_SMOOTHING:
    from utils import cross_entropy_label_smoothing
    triplet_model.compile(loss=[cross_entropy_label_smoothing, triplet_semi_hard_loss],
                          loss_weights=[0.1, 0.9],
                          optimizer=optimizer,
                          metrics=[['accuracy'], ['accuracy']])
else:
    triplet_model.compile(loss=['categorical_crossentropy', triplet_semi_hard_loss],
                          loss_weights=[0.1, 0.9],
                          optimizer=optimizer,
                          metrics=[['accuracy'], ['accuracy']])

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('./models/MobileNetV2_triplet.h5',
                             monitor='val_activation_accuracy',
                             verbose=1,
                             save_best_only=True
                             )
from utils import generator_batch_triplet_hard

train_generator = generator_batch_triplet_hard(
    img_path_list=train_img_path,
    img_label_list=train_person_ids,
    nbr_classes=nbr_person_ids,
    img_width=img_width,
    img_height=img_height,
    shuffle=True,
    save_to_dir=False,
    augment=True
)
val_generator = generator_batch_triplet_hard(
    img_path_list=val_img_path,
    img_label_list=val_person_ids,
    nbr_classes=nbr_person_ids,
    img_width=img_width,
    img_height=img_height,
    shuffle=False,
    save_to_dir=False,
    augment=False
)

# validation acc: 80%+, at least > 70%
# Epoch 100/100
# 80/80 [==============================] - 85s 1s/step - loss: 1.8314 - accuracy: 0.8366 - val_loss: 1.8178 - val_accuracy: 0.8188
#
# Epoch 00100: val_accuracy improved from 0.81250 to 0.81875, saving model to ./models/MobileNetV2_baseline.h5
triplet_model.fit(train_generator,
                   steps_per_epoch=len(train_img_path) // batch_size + 1,
                   validation_data=val_generator,
                   validation_steps=len(val_img_path) // batch_size + 1,
                   verbose=1,
                   shuffle=True,  # 一个batch中进行shuffle
                   epochs=nbr_epochs,
                   callbacks=[checkpoint])
