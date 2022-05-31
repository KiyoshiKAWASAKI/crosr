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
import pickle
from skimage import transform as skimage_transform

sys.path.append("./")
import cifar10_outliers
from transform import transform

USE_OPENCV = False

net = None
xp = cuda.cupy

def eval(data, mean, std):
    global net
    acc = 0.0
    ndata = len(data)
    for i in xrange(ndata):
    #for i in xrange(len(data)):
        indata = data.get_example(i)[0].reshape(1, 3, 28, 28)
        #chainer.using_config('train', False)
        c = net.predict(xp.array(indata))
        if isinstance(c, tuple):
            c = c[0]
        c.to_cpu()
        pred_c = np.argmax(c.data)
        if pred_c == data.get_example(i)[1]:
            acc += 1
        #pdb.set_trace()
    #acc /= len(data)
    acc /= ndata
    print "acc = %f" % acc

def calc_roc(scores, gts):
   pairs = [(x, y) for x, y in zip(scores, gts)]
   pairs.sort(key=lambda x: -x[0])
   fp = 0
   tp = 0
   num_p = np.sum(gts)
   prec = []
   rec = []
   fmeasure = 0
   #pdb.set_trace()
   for p in pairs:
       if p[1]:
           tp += 1
       else:
           fp += 1
       pr = float(tp) / (tp + fp)
       prec.append(pr)
       if num_p == 0:
         rc = 0
       else:
         rc = float(tp) / num_p
       rec.append(rc)
       if pr + rc != 0:
           newf = 2 * pr * rc / (pr + rc)
           fmeasure = max(newf, fmeasure)
   prs = {}
   for pr, rc in zip(prec, rec):
       if prs.has_key(rc) == 0:
           prs[rc] = pr
       else:
           prs[rc] = max(prs[rc], pr)

   print ("f =", fmeasure)
   mp = np.mean(prs.values())
   print("mp =", mp)
   #plt.plot(rec, prec)
   #plt.show()

   return fmeasure, mp

def softmax(av, omax):
  #pdb.set_trace()
  #pdb.set_trace()
  om = np.exp(av)
  om = om / np.sum(om, 1)
  om = np.append(om, np.asarray(1.0 - np.max(om, 1)).reshape((1, 1)), 1)
  #pdb.set_trace()
  return om

def openmax(av, omax):
  av2 = av.copy()
  w = np.zeros((1, 10))
  for i in range(10):
    wi = omax["mrs"][i].w_score_vector(
        np.linalg.norm(
            av - omax["mavs"][i], axis=1).astype(np.float64).reshape(-1)
        )
    w[0, i] = wi
    #pdb.set_trace()
    #nv[:, i] *= wi
    #av[:, i] *= (1.0 - wi)
  #pdb.set_trace()
  #av = np.append(av, np.asarray(np.sum(av2 - av, 1)).reshape(1, 1), 1)
  #pdb.set_trace()
  om = np.exp(av)
  om = om / np.sum(om, 1)
  om *= (1.0 - w)
  om = np.append(om, 1.0 - np.asarray(np.sum(om, 1)).reshape(1, 1), 1)
  #pdb.set_trace()
  return om

def eval_f1score(data, mean, std, omax=None, feval=openmax, memo=""):
    global net
    acc = 0.0
    probs_and_gts = []
    ndata = len(data)
    for i in xrange(ndata):
    #for i in xrange(len(data)):
        indata = data.get_example(i)[0].reshape(1, 3, 28, 28)
        #chainer.using_config('train', False)
        c = net.predict(xp.array(indata))
        if isinstance(c, tuple):
            c = c[0]
        c.to_cpu()
        pred_c = np.argmax(c.data)
        gtc = data.get_example(i)[1]
        if pred_c == gtc:
            acc += 1
        probs = F.softmax(c).data

        #softmax
        #pdb.set_trace()
        probs_and_gts.append((feval(c.data, omax), gtc))
        if False:
            if probs.shape[1] == 10:
                probs = np.append(probs, [[1.0 - np.max(probs)]], 1)
            probs_and_gts.append((probs, gtc))
        #openmax
        if False:
            probs_and_gts.append((openmax(c.data, omax), gtc))
        if False:
            probs_and_gts.append((osvmmax(c.data, omax), gtc))
        #pdb.set_trace()
    #acc /= len(data)
    acc /= ndata
    print "acc = %f" % acc

    fs = []
    mps = []
    for c in range(11):
       print "class =", c
       #pdb.set_trace()
       scores = [a[0][0, c] for a in probs_and_gts]
       gts = [int(a[1]) == c for a in probs_and_gts]
       f, mp = calc_roc(scores, gts)
       fs.append(f)
       mps.append(mp)
       #pdb.set_trace()
    print "mf_known =", sum(fs[:-1]) / (len(fs) - 1)
    print "mf =", sum(fs) / len(fs)
    print "mmp =", sum(mps) / len(mps)

    rtn = {}
    rtn["memo"] = memo
    rtn["mf"] = sum(fs) / len(fs)
    rtn["mmp"] = sum(mps) / len(mps)
    rtn["fs"] = fs
    rtn["mps"] = mps
    rtn["probs_and_gts"] = probs_and_gts
    return rtn

def eval_evaluator(data):
    global net
    test_iter = iterators.MultiprocessIterator(data, 100, False, False)
    rep = extensions.LogReport()
    ev = extensions.Evaluator(
        test_iter, L.Classifier(net), device=0)
    res = ev()
    print res
    return res

if __name__ == "__main__":
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
    exp_setting["network"] = "DenseNetBtlLadder_norecon2"
    exp_setting["classifier"] = "Openmax-ts20"

    exp = 0
    if exp == 0:
        net = VGGBtlLadder(10)
        serializers.load_npz(
            "results/VGGBtlLadder_2018-07-30_11-36-56_0/VGGBtlLadder.npz",
            net
            )
        f = open("results/VGGBtlLadder_2018-07-30_11-36-56_0/omax_ts100.pickle", "rb")
        omax = pickle.load(f)
        feval = openmax
    elif exp == 1:
        net = DenseNetBtlLadder(n_class=10)
        serializers.load_npz(
            "results/DenseNetBtlLadder_2018-09-01_16-32-50_0/DenseNetBtlLadder.npz",
            net
            )
        f = open("results/DenseNetBtlLadder_2018-09-01_16-32-50_0/omax_ts20.pickle", "rb") #no reconstruction
        feval = openmax
        omax = pickle.load(f)

    net.to_gpu()
    results = {}
    for test_outlier in test_outliers:
        train, valid = cifar10_outliers.get_cifar10_outlier(
            255., "/mnt/bluebird/odin/%s/test" % test_outlier)
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

        res = eval_f1score(valid, mean, std, omax, feval)
        results[test_outlier] = res
    f = open("test_results/"
        + "_".join(
            [
                exp_setting["network"],
                exp_setting["classifier"],
            ]
        )
        + ".pickle",
        "wb"
    )
    pickle.dump(results, f)
    f.close()
