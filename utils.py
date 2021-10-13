# @Author   : Zilin Zhang
# @Time     : 09/10/2021 18:37
# @Function :
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Cutout(nb_iterations=1, size=0.2, squared=False),  # 图像中选择小方块，将方块中的像素涂黑，
        # iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        # sometimes(iaa.CropAndPad(
        #     percent=(-0.05, 0.1),
        #     pad_mode=ia.ALL,
        #     pad_cval=(0, 255)
        # )),
        sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
            rotate=(-10, 10), # rotate by -45 to +45 degrees
            shear=(-5, 5), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)

# label smoothing --> reduce overfitting
# [0, 1, 0, 0] --> theta = 0.1 --> 1 ==> 1 - 0.1, 0 ==> 0.033 = 0.1 / N
# after label smoothing: [0.1 / 3, 0.9, 0.1 / 3, 0.1 /3]

def cross_entropy_label_smoothing(y_true, y_pred):
    from tensorflow.keras.losses import categorical_crossentropy
    label_smoothing = 0.1
    return categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)

def load_img_batch(img_path_list, img_label_list, nbr_classes,
                   img_width, img_height):
    batch_size = len(img_path_list)
    import numpy as np
    X_batch = np.zeros((batch_size, img_height, img_width, 3))
    Y_batch = np.zeros((batch_size, nbr_classes))

    for i in range(batch_size):
        img_path = img_path_list[i]
        import cv2
        img_bgr = cv2.imread(img_path) # img.shape: (128, 64, 3)

        if img_bgr.shape != (img_height, img_width, 3):
            img_bgr = cv2.resize(img_bgr, (img_width, img_height))

        img = img_bgr[:, :, ::-1]

        X_batch[i] = img

        if img_label_list is not None:
            label = img_label_list[i]
            Y_batch[i, label] = 1

    if img_label_list is not None:
        return X_batch, Y_batch
    else:
        return X_batch

def generator_batch(img_path_list, img_label_list, nbr_classes,
                    img_width, img_height, batch_size=32,
                    shuffle=False, save_to_dir=None,
                    augment=False):

    assert len(img_path_list) == len(img_label_list), \
        "# imag_path_list is not equal with # img_label_list"

    N = len(img_path_list)
    if shuffle:
        from sklearn.utils import shuffle as shuffle_tuple
        img_path_list, img_label_list = shuffle_tuple(img_path_list, img_label_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        X_batch, Y_batch = load_img_batch(
            img_path_list[current_index: current_index + current_batch_size],
            img_label_list[current_index: current_index + current_batch_size],
            nbr_classes, img_width, img_height
        )

        if augment:
            # https://github.com/aleju/imgaug
            X_batch = X_batch.astype(np.uint8)
            X_batch_aug = seq.augment_images(X_batch)
            X_batch = X_batch_aug

        if save_to_dir:
            pass

        X_batch = X_batch / 255.0
        X_batch = (X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        yield X_batch, Y_batch

def generator_batch_predict(img_path_list,
                    img_width, img_height, batch_size=32):

    N = len(img_path_list)
    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        X_batch = load_img_batch(
            img_path_list[current_index: current_index + current_batch_size],
            None, 1, img_width, img_height
        )


        X_batch = X_batch / 255.0
        X_batch = (X_batch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        yield X_batch