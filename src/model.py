import pickle
from resnet import *
from torch.autograd import Variable



class Model(nn.Module):
    
    def __init__(self, args, pretrained=True, num_classes=7):
        super(Model, self).__init__()
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        with open(args.resnet50_path, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        resnet50.load_state_dict(weights)
        
        self.features = nn.Sequential(*list(resnet50.children())[:-2])  
        self.features2 = nn.Sequential(*list(resnet50.children())[-2:-1])  
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