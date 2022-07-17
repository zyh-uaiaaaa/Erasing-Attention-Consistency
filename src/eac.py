##### flip attention plus gaussian mixture
__all__ = ['ResNet', 'resnet50']
import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import numpy as np
import random
import cv2
import csv
import math
import torch.utils.data as data
from torch.autograd import Variable
import pandas as pd
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture





parser = argparse.ArgumentParser()
parser.add_argument('--raf_path', type=str, default='../../../zyh/raf-basic',help='raf_dataset_path')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
parser.add_argument('--lam', type=float, default=5, help='kl_lambda')
parser.add_argument('--num_gradual', type = int, default = 5, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
args = parser.parse_args()



    

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=8631, include_top=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.include_top = include_top
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not self.include_top:
            return x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model = ResNet(Bottleneck, [3, 4, 6, 3])
    
import pickle
with open('../../resnet50_ft_weight.pkl', 'rb') as f:
    obj = f.read()
weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
model.load_state_dict(weights)
    

    

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
class Model2(nn.Module):
    
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0):
        super(Model2, self).__init__()
        res18 = model
        self.drop_rate = drop_rate
        self.features = nn.Sequential(*list(res18.children())[:-2])  
        self.features2 = nn.Sequential(*list(res18.children())[-2:-1])  
        self.fc = nn.Linear(2048, 7)  
        
        
    def forward(self, x):        
        x = self.features(x)
        #### 1, 2048, 7, 7
        feature = self.features2(x)
        #### 1, 2048, 1, 1
        
        feature = feature.view(feature.size(0), -1)
        output = self.fc(feature)
        
        params = list(self.parameters())
        fc_weights = params[-2].data
        fc_weights = fc_weights.view(1, 7, 2048, 1, 1)
        fc_weights = Variable(fc_weights, requires_grad = False)

        # attention
        feat = x.unsqueeze(1) # N * 1 * C * H * W
        hm = feat * fc_weights
        hm = hm.sum(2) # N * self.num_labels * H * W

        return output, hm
    
    
    
        
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    

device = torch.device('cuda:0')
# setup_seed(0)



model2 = Model2()


image_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()])


data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(scale=(0.02, 0.25))
])


data_transforms_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


train_dataset2 = RafDataset(args.raf_path, phase='train', transform=data_transforms,
                           image_transform=image_transforms)


test_dataset = RafDataset(args.raf_path, phase='test', transform=data_transforms_val,
                          image_transform=image_transforms)


train_loader2 = torch.utils.data.DataLoader(train_dataset2,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.workers,
                                           pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=args.workers,
                                          pin_memory=True)


model2.to(device)


optimizer1 = torch.optim.Adam(
   list(model2.parameters()) , lr=0.0001, weight_decay=1e-4)
scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.9)


train_acc = []
test_acc = []



best_acc = 0
best_epoch = 0
global_step = 0

# forget_rate = 0.4
# rate_schedule = np.ones(args.epochs)*forget_rate
# rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)


def generate_flip_grid(w, h):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().cuda()
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid

w1 = 7
h1 = 7
grid_l = generate_flip_grid(w1, h1)
    
for i in range(1, args.epochs + 1):
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    
    model2.train()
    

    
    total_loss = []
    for batch_i, (imgs, labels, indexes, imgs1) in enumerate(train_loader2):

        imgs2 = imgs.to(device)
        imgs1 = imgs1.to(device)
        labels2 = labels.to(device)
        

        criterion = nn.CrossEntropyLoss(reduction='none')
        
        

        output, hm1 = model2(imgs2)
        output_flip, hm2 = model2(imgs1)
        
        loss1 = nn.CrossEntropyLoss()(output, labels2)
        
        flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
        flip_grid_large = Variable(flip_grid_large, requires_grad = False)
        flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
        hm2_flip = F.grid_sample(hm2, flip_grid_large, mode = 'bilinear', padding_mode = 'border', align_corners=True)
        flip_loss_l = F.mse_loss(hm1, hm2_flip)
#         flip_loss_l = F.mse_loss(hm1, hm2)

        loss = loss1 + args.lam * flip_loss_l
        if batch_i == 1:
            print('loss1: ', loss1, 'flip loss: ', args.lam * flip_loss_l)
        
        

            
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()


        
        
        
        iter_cnt += 1
        _, predicts = torch.max(output, 1)

        correct_num = torch.eq(predicts, labels2).sum()
        correct_sum += correct_num
        running_loss += loss

    scheduler1.step()


     

    
    running_loss = running_loss / iter_cnt
    acc = correct_sum.float() / float(train_dataset2.__len__())
    train_acc.append(acc)
    print('Epoch : %d, train_acc : %.4f, train_loss: %.4f' % (i, acc, running_loss))
    pre_acc = 0
    with torch.no_grad():
        

        model2.eval()
        
        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0


        for batch_i, (imgs, labels, indexes, imgs1) in enumerate(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)


            outputs, _ = model2(imgs)


            loss = nn.CrossEntropyLoss()(outputs, labels)

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)
            
            
            
            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num

            running_loss += loss
            data_num += outputs.size(0)

        running_loss = running_loss / iter_cnt
        test_acc = correct_sum.float() / float(data_num)


        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = i
            
        with open('eac_no_noise.txt', 'a') as f:
            f.write('epoch: '+str(i)+'acc: '+str(test_acc)+'\n')
#         with open('resnet50_checkpoint.txt', 'a') as f:
#             f.write('epoch: '+str(i)+'acc: '+str(test_acc)+'\n')
#         torch.save({'model_state_dict': model2.state_dict(),}, "att_flip_03.pth")
#         print('Model saved.') 
        print('Epoch : %d, test_acc : %.4f, test_loss: %.4f' % (i, test_acc, running_loss))           
print('best acc: ', best_acc, 'best epoch: ', best_epoch)