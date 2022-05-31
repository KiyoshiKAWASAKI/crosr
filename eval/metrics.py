import cv2
import numpy as np

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

def test_fmeasure(test_img_list_name, forward_func, memo=""):
     #img_list = open("../test_mnist_omniglotback.txt", "r").readlines()
     rtn = {}
     img_list = open(test_img_list_name, "r").readlines()
     #img_list = open("../test_mnist.txt", "r").readlines()
     img_list = [s.rstrip() for s in img_list]
     probs_and_gts = []
     for i, l in enumerate(img_list):
         #print i, ":", l
         tokens = l.split(" ")
         img = tokens[0]
         #out = net_forward(img)
         #om = out["prob"]
         om = forward_func(img)
         #om = forward_openmax(img, omax)
         probs_and_gts.append((np.copy(om), tokens[1]))

     fs = []
     mps = []
     for c in range(11):
       print "class =", c
       #pdb.set_trace()
       scores = [a[0][0, c] for a in probs_and_gts]
       gts = [int(a[1]) == c for a in probs_and_gts]
       #pdb.set_trace()
       f, mp = calc_roc(scores, gts)
       fs.append(f)
       mps.append(mp)
     print "mf_known =", sum(fs[:-1]) / (len(fs) - 1)
     print "mf =", sum(fs) / len(fs)
     print "mmp =", sum(mps) / len(mps)

     rtn["memo"] = memo
     rtn["mf"] = sum(fs) / len(fs)
     rtn["mmp"] = sum(mps) / len(mps)
     rtn["fs"] = fs
     rtn["mps"] = mps
     rtn["probs_and_gts"] = probs_and_gts
     return rtn

def view_sorted(sample_name, test_img_list, forward_func, c, start_idx=0):
    img_list = open(test_img_list, "r").readlines()
    #img_list = open("../test_mnist.txt", "r").readlines()
    img_list = [s.rstrip() for s in img_list]
    prob_gt_names = []
    for i, l in enumerate(img_list):
        #print i, ":", l
        tokens = l.split(" ")
        img = tokens[0]
        om = forward_func(img)
        #om = forward_openmax(img, omax)
        prob_gt_names.append((np.copy(om), tokens[1], img))

    #c = 4
    prob_gt_names = sorted(prob_gt_names, key=lambda x: -x[0][0, c])
    cat_img = None
    #prob_gt_names = filter(lambda x: x[0][0, c] < 0.999, prob_gt_names)
    #start_idx = 800
    prob_gt_names = prob_gt_names[start_idx : start_idx + 100]
    for pgn in prob_gt_names[:50]:
        img_name = pgn[2]
        print img_name
        print pgn[0][0, c]
        im = cv2.imread(img_name)
        im = cv2.resize(im, (28, 28))
        if cat_img == None:
            cat_img = im
        else:
            cat_img = cv2.hconcat([cat_img, im])
    #cv2.imshow("cat", cat_img)
    #cv2.waitKey()
    cv2.imwrite("samples/%d_%d_%s.png" % (c, start_idx, sample_name), cat_img)
