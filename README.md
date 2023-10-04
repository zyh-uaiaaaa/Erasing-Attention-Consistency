# Erasing-Attention-Consistency
Official implementation of the ECCV2022 paper: [Learn From All: Erasing Attention Consistency for Noisy Label Facial Expression Recognition](https://arxiv.org/pdf/2207.10299.pdf)



## Abstract
Noisy label Facial Expression Recognition (FER) is more challenging than traditional noisy label classification tasks due to the inter-class similarity and the annotation ambiguity. Recent works mainly tackle this problem by filtering out large-loss samples. In this paper, we explore dealing with noisy labels from a new feature-learning perspective. We find that FER models remember noisy samples by focusing on a part of the features that can be considered related to the noisy labels instead of learning from the whole features that lead to the latent truth. Inspired by that, we propose a novel Erasing Attention Consistency (EAC) method to suppress the noisy samples during the training process automatically. Specifically, we first utilize the flip semantic consistency of facial images to design an imbalanced framework. We then randomly erase input images and use flip attention consistency to prevent the model from focusing on a part of the features. EAC significantly outperforms state-of-the-art noisy label FER methods and generalizes well to other tasks with a large number of classes like CIFAR100 and Tiny-ImageNet.



## Train

**Torch** 

We train EAC with Torch 1.8.0 and torchvision 0.9.0.

**Dataset**

Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), put the Image folder under the raf-basic folder:
```key
- raf-basic/
	 Image/aligned/
	     train_00001_aligned.jpg
	     test_0001_aligned.jpg
	     ...

```

**Pretrained backbone model**

Download the pretrained [ResNet-50 model](https://drive.google.com/file/d/1yQRdhSnlocOsZA4uT_8VO0-ZeLXF4gKd/view?usp=sharing) and then put it under the model directory. 

**Train the EAC model**

Train EAC with clean labels, 10\% noise, 20\% noise and 30\% noise. 

```key
cd src
sh train.sh
```


## Results

**Feature visualization**

The effectiveness of EAC can be shown by the feature visualization. The two images are the learned features by EAC, while with different labels. The left image shows the features labelled with noisy labels, which are also the training labels. The right image shows the features labelled with latent truth labels, which are the clean labels. The images illustrate that training with noisy labels, EAC can still learn useful features corresponding to the latent truth instead of directly fitting the noisy labels.

![](https://github.com/zyh-uaiaaaa/Erasing-Attention-Consistency/blob/main/imgs/feature_visualization.png)

**Accuracy**

Traing EAC on RAF-DB clean train set (ResNet-50 backbone) should achieve over 90\% accuracy on RAF-DB test set.

![](https://github.com/zyh-uaiaaaa/Erasing-Attention-Consistency/blob/main/imgs/accuracy.png)


## Others

**Frequently asked questions**

Changing backbone to ResNet-18 should first tune the learning rate from 1e-4 to 2e-4 in order to acquire high classification accuracy. The pretrained ResNet-18 model can be found in [this github repository](https://github.com/zyh-uaiaaaa/Relative-Uncertainty-Learning).

Previous pretrained ResNet-50 is unavailable, the new pretrained model can be downloaded from [here](https://drive.google.com/file/d/1yQRdhSnlocOsZA4uT_8VO0-ZeLXF4gKd/view?usp=sharing).


**Citation**

If you find our code useful, please consider citing our paper:

```shell
@inproceedings{zhang2022learn,
  title={Learn from all: Erasing attention consistency for noisy label facial expression recognition},
  author={Zhang, Yuhang and Wang, Chengrui and Ling, Xu and Deng, Weihong},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXVI},
  pages={418--434},
  year={2022},
  organization={Springer}
}
```
