import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pickle

import multiprocessing

def calc_roc(scores, gts):
   pairs = [(x, y) for x, y in zip(scores, gts)]
   pairs.sort(key=lambda x: -x[0])
   fp = 0
   tp = 0
   num_p = np.sum(gts)
   prec = []
   rec = []
   f_ths = []
   fmeasure = 0
   th = 1.0
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
           f_ths.append((p[0], newf))
           # select f measure @ th = 0.5
           if p[0] < th and p[0] >= 0.5:
               fmeasure = newf
               th = p[0]
   prs = {}
   for pr, rc in zip(prec, rec):
       if prs.has_key(rc) == 0:
           prs[rc] = pr
       else:
           prs[rc] = max(prs[rc], pr)

   mp = np.mean(prs.values())
   return fmeasure

def calc_f1scores_unk(probs_and_gts):
    fs = []
    for c in range(10, 11):
      #print "class =", c
      scores = [1.0 - np.max(a[0][0, :]) for a in probs_and_gts]
      gts = [int(a[1]) == c for a in probs_and_gts]
      f = calc_roc(scores, gts)
      fs.append(f)
    return sum(fs) / len(fs)

def calc_f1scores(probs_and_gts):
    fs = []
    for c in range(0, 11):
      #print "class =", c
      #pdb.set_trace()
      scores = [a[0][0, c] for a in probs_and_gts]
      gts = [int(a[1]) == c for a in probs_and_gts]
      #pdb.set_trace()
      f = calc_roc(scores, gts)
      fs.append(f)
    return sum(fs) / len(fs)

if __name__ == "__main__":
    results_root = "test_results/"

    parser = argparse.ArgumentParser(
        prog="crosr result printer",
        add_help=True
      )
    parser.add_argument(
        "-m", "--model", 
        choices=["vgg", "densenet"],
        required = True
        )
    args = parser.parse_args()

    #VGG Cifar10
    if args.model == "vgg":
        results = [
            "CVPR2019_Plain_Supervised_Softmax.pickle",
            "CVPR2019_Plain_Supervised_Openmax.pickle",
            "CVPR2019_Plain_Ladder_Softmax.pickle",
            "CVPR2019_Plain_Ladder_Openmax.pickle",
            "CVPR2019_Plain_DHRNet_Softmax.pickle",
            "CVPR2019_Plain_DHRNet_Openmax.pickle",
            "CVPR2019_Plain_DHRNet_CROSR.pickle",
        ]
    if args.model == "densenet":
        results = [
            "CVPR2019_Supervised_Softmax.pickle",
            "CVPR2019_Supervised_Openmax.pickle",
            "CVPR2019_DHRNet_Openmax.pickle",
            "CVPR2019_DHRNet_CROSR.pickle",
        ]
    
    for res in results:
        f = open(results_root + res, "rb")
        r = pickle.load(f)
        test_outliers = [
            "Imagenet",
            "LSUN",
            "Imagenet_resize",
            "LSUN_resize",
        ]
        print res, "\t",
        for test_outlier in test_outliers:
            #print "%s\t" % (r[test_outlier]["mf"]) ,
            print "%s\t" % (calc_f1scores(r[test_outlier]["probs_and_gts"])) ,
            #print "%s\t" % (calc_f1scores_unk(r[test_outlier]["probs_and_gts"])) ,
        print ""
