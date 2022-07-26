<<<<<<< HEAD
# w2n

## Overall

This code release the implementation of inference phase for W2N.  The full training code and data procession scripts will be integrated and released after the code review.

We implemented the code based on [detectron2 toolkit](https://github.com/facebookresearch/detectron2) and [unbiased teacher](https://github.com/facebookresearch/unbiased-teacher). Sincerely thanks for your resources.

## Hardware

We use 8 RTX 1080Ti GPU (11GB) to train and evaluate our method, GPU with larger memory is better (e.g., TITAN RTX with 24GB memory)

## Requirements

- Python 3.6 or higher
- CUDA 10.2 with cuDNN 7.6.5
- PyTorch 1.6.0
- numpy 1.19.2
- opencv 4.5.1

## Additional resources

### Datasets

For example, PASCAL VOC 2007 dataset

1. Download the training, validation, test data and VOCdevkit

   ```shell
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   ```

2. Extract all of these tars into one directory named `VOCdevkit`

   ```shell
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

3. It should have this basic structure

   ```sheel
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

4. Create symlinks for the PASCAL VOC dataset

   ```shell
   cd $PROJECTS_ROOT
   mkdir datasets
   cd datasets
   ln -s $VOCdevkit/VOC2007 VOC2007
   ```

### Pretrained Model

[OICR+REG+W2N](https://drive.google.com/file/d/1xqIiL1LhQkb45f1gjSoJg0nRzn2ww4sw/view?usp=sharing) on PASCAL VOC 2007

[CASD+W2N](https://drive.google.com/file/d/1my87LC63ZA7JNWZtuUPuzTwHOfU9B8pH/view?usp=sharing) on PASCAL VOC 2007

[LBBA+W2N](https://drive.google.com/file/d/1JJSliE1Oc3jbmWbeAy-VCd0UB2gtdeZZ/view?usp=sharing) on PASCAL VOC 2007



### Inference

```shell
python train_net.py --eval-only --config-file=configs\pascal_voc_no_labeled\faster_rcnn_R_50_FPN_pascasl_unlabeled_reg.yaml --num-gpus=8 MODEL.WEIGHTS model_w2n_lbba.pth
```





=======
# w2n_wsod
Official implementation of the paper ``W2N: Switching From Weak Supervision to Noisy Supervision for Object Detection"
# TODO
Code will be released.
>>>>>>> cd4253fbe6cc90d0ded6556e0f93c539f4636b8d
