from chainercv import transforms
import cv2 as cv
import numpy as np
from skimage import transform as skimage_transform


USE_OPENCV = False
def cv_rotate(img, angle):
    if USE_OPENCV:
        img = img.transpose(1, 2, 0) / 255.
        center = (img.shape[0] // 2, img.shape[1] // 2)
        r = cv.getRotationMatrix2D(center, angle, 1.0)
        img = cv.warpAffine(img, r, img.shape[:2])
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    else:
        # scikit-image's rotate function is almost 7x slower than OpenCV
        img = img.transpose(1, 2, 0) / 255.
        img = skimage_transform.rotate(img, angle, mode='edge')
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    return img


def transform(
        inputs, mean, std, random_angle=15., pca_sigma=255., expand_ratio=1.0,
        crop_size=(32, 32), train=True):
    #pdb.set_trace()
    img, label = inputs
    img = img.copy()

    # Random rotate
    if random_angle != 0:
        angle = np.random.uniform(-random_angle, random_angle)
        img = cv_rotate(img, angle)

    # Color augmentation
    if train and pca_sigma != 0:
        img = transforms.pca_lighting(img, pca_sigma)

    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]

    if train:
        # Random flip
        img = transforms.random_flip(img, x_random=True)
        # Random expand
        if expand_ratio > 1:
            img = transforms.random_expand(img, max_ratio=expand_ratio)
        # Random crop
        if tuple(crop_size) != (32, 32):
            img = transforms.random_crop(img, tuple(crop_size))
    else:
        if tuple(crop_size) != (32, 32):
            img = transforms.center_crop(img, tuple(crop_size))
    return img, label
