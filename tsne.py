#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from functools import partial
from importlib import import_module
import json
import os
import pdb
import random
import re
import shutil
import sys
import time

import copy
import numpy as np

import chainer
from chainer import cuda
from chainer import iterators
from chainer import optimizers
from chainer import serializers
from chainer import training
from chainer import Variable
from chainer.datasets import TransformDataset
from chainer.datasets import cifar
import chainer.links as L
from chainer.training import extensions
from chainercv import transforms
import cv2 as cv
import pickle
from skimage import transform as skimage_transform

sys.path.append("./")
import cifar10_outliers
from transform import transform

from sklearn.manifold import TSNE
import random
# path to test images
data_root = "/mnt/data/data/mnist/val_test/"

net = None

def flatten(v):
    return np.max(v, (2, 3))

def embed(data):
    print data.shape
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(data)
    print X_reduced.shape
    pass

def embed_img(imgs, reidx):
    ndata = len(reidx)
    #ndata = len(imgs)
    ndata = 1000
    data = np.zeros((ndata, 10))
    if "zweight" in omax:
        zw = omax["zweight"]
    else:
        zw = 1.0
    for i in xrange(ndata):
    #for i in xrange(len(data)):
        idx = reidx[i]
        indata = imgs.get_example(idx)[0].reshape(1, 3, 28, 28)
        c, r, zs = net.predict_z(xp.array(indata))

        c.to_cpu()
        v = c.data
        if False:
        #for z in zs:
            z.to_cpu()
            v2 = z.data
            if len(v2.shape) > 2:
                v2 = flatten(v2)
            #pdb.set_trace()
            v = np.concatenate([v, v2 * zw], 1)
        pred_c = np.argmax(c.data)
        gtc = imgs.get_example(idx)[1]

        data[i, :] = v.reshape((-1))

    X_reduced = TSNE(n_components=2, random_state=0).\
        fit_transform(data.astype(np.float32))
    print X_reduced.shape
    return X_reduced

def draw_scatter_image(imgs, points, mean, std, reidx):
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])

    canvas_size = 1280
    im_resize = 28
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    for i, p in enumerate(points):
        idx = reidx[i]
        im = imgs.get_example(idx)[0]
        #pdb.set_trace()
        im = im.transpose((1, 2, 0))

        im = (im * std + mean).copy()
        im = (im * 255).astype(np.uint8)
        im = im[:, :, (2, 1, 0)].copy()
        cls = imgs.get_example(idx)[1]
        color = (255, 0, 0) if int(cls) != 10 else (0, 0, 255)
        cv.rectangle(im, (0, 0), (im_resize - 1, im_resize - 1),
            color)
        x = int((canvas_size - im_resize) * points[i, 0] / (x_max - x_min))
        y = int((canvas_size - im_resize) * points[i, 1] / (y_max - y_min))
        try:
            canvas[y : y + im_resize, x : x + im_resize, :] = im
        except:
            pass
    cv.imshow("canvs", canvas)
    cv.imwrite("tsne.png", canvas)
    cv.waitKey()


if __name__ == "__main__":

    # --- chainer init ---------------------------------------------------------
    sys.path.append("./models/")
    from vgg import *
    from vggladder import *
    from vggbtlladder import *
    from vggladder_denoise import *
    from densenet_btlladder import *

    # Enable autotuner of cuDNN
    chainer.config.autotune = True
    chainer.config.train = False
    chainer.global_config.train = False

    test_outliers = [
        "Imagenet",
        "LSUN",
        "Imagenet_resize",
        "LSUN_resize",
    ]

    exp_setting = {}
    exp_setting["network"] = "VGGBtlLadder"
    #exp_setting["network"] = "DenseNetBtlLadder"
    exp_setting["classifier"] = "OpenmaxZd-ts20-zw1.0"
    #exp_setting["outlier"] = "LSUN"


    exp = 1
    if exp == 0:
        net = VGGBtlLadder(10)
        serializers.load_npz(
            "results/VGGBtlLadder_2018-07-30_11-36-56_0/VGGBtlLadder.npz",
            net
            )
        f = open("results/VGGBtlLadder_2018-07-30_11-36-56_0/omax_z_ts100.pickle", "rb")
        omax = pickle.load(f)
        feval = None
    elif exp == 1: #norecon
        net = VGGBtlLadder(10)
        serializers.load_npz(
            "results/VGGBtlLadder_2018-07-30_11-36-56_0/VGGBtlLadder.npz",
            net
            )
        f = open("results/VGGBtlLadder_2018-07-30_11-36-56_0/omax_ts100.pickle", "rb")
        omax = pickle.load(f)
        #omax = None
        feval = None

    net.to_gpu()

    train, valid = cifar10_outliers.get_cifar10_outlier(
        255., "/mnt/bluebird/odin/%s/test" % test_outliers[3]) # path to outlier images
    #train, valid = cifar.get_cifar10(scale=255.)
    mean = np.mean([x for x, _ in train], axis=(0, 2, 3))
    std = np.std([x for x, _ in train], axis=(0, 2, 3))

    train_transform = partial(
        transform, mean=mean, std=std, train=True,
            random_angle=0., pca_sigma=0, expand_ratio=0, crop_size=(28, 28))
    valid_transform = partial(transform, mean=mean, std=std, train=False,
         random_angle=0., pca_sigma=0, expand_ratio=0, crop_size=(28, 28))

    train = TransformDataset(train, train_transform)
    valid = TransformDataset(valid, valid_transform)
    print('mean:', mean)
    print('std:', std)
    reidx = range(len(valid))
    random.shuffle(reidx)
    reidx = reidx[:2000]

    data = embed_img(valid, reidx=reidx)
    draw_scatter_image(valid, data, mean, std, reidx)
    #print img_list
