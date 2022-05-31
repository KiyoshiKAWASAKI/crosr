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
from skimage import transform as skimage_transform

import libmr
import pickle

sys.path.append("./")
import cifar10_outliers
from transform import transform

USE_OPENCV = False

net = None
xp = cuda.cupy

def flatten(v):
    return np.max(v, (2, 3))

def train_openmax(data, mean, std, nclass=10, tail_size=100, zweight=1.0):
    global net
    acc = 0.0
    ip2s = {i: [] for i in range(nclass)}
    mavs = {i: None for i in range(nclass)}
    ncorrect = [0 for i in range(nclass)]
    n_imgs = len(data)
    for i in xrange(n_imgs):
    #for i in xrange(len(data)):
        indata = data.get_example(i)[0].reshape(1, 3, 28, 28)
        #chainer.using_config('train', False)
        #c, r, zs = net.predict_z(xp.array(indata))
        c, r, zs = net.predict_x(xp.array(indata))
        if isinstance(c, tuple):
            c = c[0]
        c.to_cpu()
        for z in zs:
            z.to_cpu()
        pred_c = np.argmax(c.data)
        gt_c = data.get_example(i)[1]
        if pred_c == gt_c:
            acc += 1
            ncorrect[gt_c] += 1

            #accumulate stats
            v = c.data.copy()
            for z in zs:
                v2 = z.data
                if len(v2.shape) > 2:
                    v2 = flatten(v2)
                v = np.concatenate([v, v2 * zweight], 1)
            #pdb.set_trace()
            ip2s[gt_c].append(v)
            if mavs[gt_c] is not None:
                mavs[gt_c] += v
            else:
                mavs[gt_c] = v.copy()
        #pdb.set_trace()
    #acc /= len(data)
    acc /= n_imgs
    print "acc = %f" % acc

    for c in range(nclass):
      mavs[c] /= len(ip2s[c])

    mrs = {i: libmr.MR() for i in range(nclass)}
    for c in range(nclass):
      ns = np.linalg.norm(np.concatenate(ip2s[c]) - mavs[c], axis=1)
      #pdb.set_trace()
      mrs[c].fit_high(
      #mrs[c].fit_low(
        ns.reshape(-1),
        tail_size
      )
    omax = {
      "mavs": mavs,
      "mrs": mrs,
      "zweight": zweight
    }
    return omax

if __name__ == "__main__":
    sys.path.append("./models/")
    from vgg import *
    from vggladder import *
    from vggbtlladder import *
    from densenet_btlladder import *

    # Enable autotuner of cuDNN
    chainer.config.autotune = True
    chainer.config.train = False
    chainer.global_config.train = False

    net = VGGBtlLadder(10)
    serializers.load_npz(
        "results/VGGBtlLadder_2018-07-30_11-36-56_0/VGGBtlLadder.npz",
        net
        )

    net.to_gpu()
    train, valid = cifar10_outliers.get_cifar10_outlier(
        255., "/mnt/bluebird/odin/Imagenet/test")
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
    #res = eval_evaluator(valid)
    #eval(valid, mean, std)
    omax = train_openmax(train, mean, std, tail_size=20, zweight = 1.0)
    f = open("results/VGGBtlLadder_2018-07-30_11-36-56_0/trained_openmax.pickle", "wb")
    pickle.dump(omax, f)
