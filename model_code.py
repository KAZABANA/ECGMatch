import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from copy import deepcopy
import numpy as np
from typing import Optional
from torch.optim.optimizer import Optimizer


class MyResidualBlock(nn.Module):
    def __init__(self,complexity,downsample):
        super(MyResidualBlock,self).__init__()
        self.downsample = downsample
        self.stride = 2 if self.downsample else 1
        K = 9
        P = (K-1)//2
        self.conv1 = nn.Conv2d(in_channels=complexity,
                               out_channels=complexity,
                               kernel_size=(1,K),
                               stride=(1,self.stride),
                               padding=(0,P),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(complexity)

        self.conv2 = nn.Conv2d(in_channels=complexity,
                               out_channels=complexity,
                               kernel_size=(1,K),
                               padding=(0,P),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(complexity)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
            self.idfunc_1 = nn.Conv2d(in_channels=complexity,
                                      out_channels=complexity,
                                      kernel_size=(1,1),
                                      bias=False)

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)
        x = x+identity
        return x

class NN(nn.Module):
    def __init__(self,nOUT,complexity,inputchannel):
        super(NN,self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=inputchannel,out_channels=complexity,kernel_size=(1,15),padding=(0,7),stride=(1,2),bias=False),
                                     nn.BatchNorm2d(complexity),
                                     nn.LeakyReLU(inplace=True),
                                     MyResidualBlock(complexity,downsample=True),
                                     MyResidualBlock(complexity,downsample=True),
                                     MyResidualBlock(complexity,downsample=True),
                                     MyResidualBlock(complexity,downsample=True),
                                     MyResidualBlock(complexity,downsample=True),nn.Dropout(0.5)
                                     )#
        self.mha = nn.MultiheadAttention(complexity,8)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.bottleneck = nn.Sequential(nn.Linear(complexity,complexity))
        self.classifier = nn.Linear(complexity,nOUT)
        self.ch_fc1 = nn.Linear(nOUT,complexity)
        self.ch_bn = nn.BatchNorm1d(complexity)
        self.ch_fc2 = nn.Linear(complexity,nOUT)
    def feature_extraction(self, x):
        x =self.encoder(x)
        x = x.squeeze(2).permute(2, 0, 1)
        x, s = self.mha(x, x, x)
        x = x.permute(1, 2, 0)
        x = self.pool(x).squeeze(2)
        return x
    def forward(self, source_data):
        source_feature = self.feature_extraction(source_data)
        source_prediction = self.classifier(self.bottleneck(source_feature))
        p = source_prediction.detach()
        return source_prediction,p
    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.encoder.parameters(), "lr_mult": 0.1},
            {"params": self.mha.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1},
            {"params": self.classifier.parameters(), "lr_mult": 1},
        ]
        return params

def gaussian_noise(input, std,device):
    input_shape =input.size()
    noise = torch.normal(mean=0, std=std, size =input_shape)
    noise = noise.to(device)
    return input + noise

def filp_time(input):
    input=torch.flip(input,[3])
    return input

def filp_channel(input):
    rand_index=torch.randperm(12)
    input=input[:,rand_index,:,:]
    return input

def dropout_burst(input):
    for i in range(input.shape[1]):
        length=np.random.randint(input.shape[3]/2)
        discard_start=np.random.randint(length,input.shape[3]-length)
        input[:,i,:,discard_start-length:discard_start+length]=0
    return input

def tar_augmentation(input, type,device):
    if type=='Weak':
        aug_type=np.random.randint(4)
        if aug_type == 0:
            input = filp_time(input)
        if aug_type == 1:
            input = dropout_burst(input)
        if aug_type == 2:
            input = gaussian_noise(input, 0.05, device)
        if aug_type == 3:
            input = filp_channel(input)
    elif type=='Strong':
        aug_list = [0,1,2,3]
        std = 0.5
        aug_que=np.unique(np.random.choice(aug_list, 4))
        np.random.shuffle(aug_que)
        for aug_type in aug_que:
            if aug_type == 0:
                input = filp_time(input)
            if aug_type == 1:
                input = dropout_burst(input)
            if aug_type == 2:
                input = gaussian_noise(input, std, device)
            if aug_type == 3:
                input = filp_channel(input)
    return input

class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75,max_iter: Optional[float] = 100):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter=max_iter
    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num/self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']
        self.iter_num += 1
        
class ModelEma(torch.nn.Module):
    def __init__(self,model,decay=0.999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)
    
    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v_name, model_v_name,ema_v, model_v in zip(self.module.state_dict(), model.state_dict(),self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                if 'running' in ema_v_name or 'batch' in ema_v_name:
                    ema_v.copy_(model_v)
                else:
                    ema_v.copy_(update_fn(ema_v, model_v))
                
    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
