# Erasing-Attention-Consistency
Official implementation of the ECCV2022 paper: Learn From All: Erasing Attention Consistency for Noisy Label Facial Expression Recognition



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

Download the pretrained ResNet-50 model and then put it under the model directory. 

**Train the EAC model**

Train EAC with clean labels, 10\% noise, 20\% noise and 30\% noise. 

```key
cd src
sh train.sh
```



