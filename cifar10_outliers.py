from chainer.datasets import cifar
from chainer.datasets import tuple_dataset
import numpy as np
import os
import pdb
from PIL import Image
def get_cifar10_outlier(scale, outlier_dir):
    train, valid = cifar.get_cifar10(scale)
    imgs = os.listdir(outlier_dir)
    data = np.zeros((len(imgs), 3, 32, 32), dtype=np.float32)
    for i, img in enumerate(imgs):
        #print img
        im = np.asarray(Image.open(outlier_dir + "/" + img), dtype=np.float32)
        if len(im.shape) == 2:
            im = np.tile(im.reshape(im.shape[0], -1, 1), (1, 1, 3))
            #pdb.set_trace()
        if im.shape[0] == 36:
            im = im[2:-2, 2:-2, :]
        if im.shape[0] != 32:
            pdb.set_trace()
        #pdb.set_trace()
        #try:
        data[i, :, :, :] = im.transpose((2, 0, 1)) / 255
        #except:
        #pdb.set_trace()
        #pdb.set_trace()
    valid = tuple_dataset.TupleDataset(
        np.concatenate((valid._datasets[0], data), 0),
        np.concatenate((valid._datasets[1], np.asarray([10] * data.shape[0])), 0)
        )
    return train, valid

def get_cifar100_outlier(scale, outlier_dir):
    train, valid = cifar.get_cifar100(scale)
    imgs = os.listdir(outlier_dir)
    data = np.zeros((len(imgs), 3, 32, 32), dtype=np.float32)
    for i, img in enumerate(imgs):
        #print img
        im = np.asarray(Image.open(outlier_dir + "/" + img), dtype=np.float32)
        if len(im.shape) == 2:
            im = np.tile(im.reshape(im.shape[0], -1, 1), (1, 1, 3))
            #pdb.set_trace()
        if im.shape[0] == 36:
            im = im[2:-2, 2:-2, :]
        if im.shape[0] != 32:
            pdb.set_trace()
        #pdb.set_trace()
        try:
            data[i, :, :, :] = im.transpose((2, 0, 1)) / 255
        except:
            pdb.set_trace()
        #pdb.set_trace()
    valid = tuple_dataset.TupleDataset(
        np.concatenate((valid._datasets[0], data), 0),
        np.concatenate((valid._datasets[1], np.asarray([100] * data.shape[0])), 0)
        )
    return train, valid


def get_outlier(scale, outlier_dir):
    train, valid = cifar.get_cifar10(scale)
    imgs = os.listdir(outlier_dir)
    data = np.zeros((len(imgs), 3, 32, 32), dtype=np.float32)
    for i, img in enumerate(imgs):
        #print img
        im = np.asarray(Image.open(outlier_dir + "/" + img), dtype=np.float32)
        if len(im.shape) == 2:
            im = np.tile(im.reshape(im.shape[0], -1, 1), (1, 1, 3))
            #pdb.set_trace()
        if im.shape[0] == 36:
            im = im[2:-2, 2:-2, :]
        if im.shape[0] != 32:
            pdb.set_trace()
        pdb.set_trace()
        try:
            data[i, :, :, :] = im.transpose((2, 0, 1)) / 255
        except:
            pdb.set_trace()

        #pdb.set_trace()
    valid = tuple_dataset.TupleDataset(
        data,
        np.asarray([10] * data.shape[0])
        )
    return train, valid


if __name__ == "__main__":
    train, valid = get_cifar10_outlier(255, "G:/shared/shared/codes/crosr-release/crosr-release/Imagenet/test/")
