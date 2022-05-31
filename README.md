# Official code for "Classification-Reconstruction Learning for Open-Set Recognition"
- Paper: https://arxiv.org/abs/1812.04246
- Venue: CVPR 2019

## Preparation
- Dependencies:                                                                                        
-- OpenCV-Python
-- Chainer                                                                                             
-- LibMR             
-- Scikit-Image 
- Download outlier datsets from https://github.com/facebookresearch/odin
- Modify hard-coded paths to the datasets

## Usage 

- train_*.py
trains networks for feature extraction
Example:

```
$ python3 train_ladder.py --model_file models/vggbtlladder.py --model_name VGGBtlLadder --rloss_weight 1
```

- train_openmax_multilayer.py
trains the open-set classifier using features from the network trained above

- tests/eval_*.py
runs evaluation

- print_score.py
prints the scores

## Reference
- CIFAR 10 training code in chainer
https://github.com/mitmul/chainer-cifar10
