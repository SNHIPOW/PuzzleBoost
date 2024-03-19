import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import torch.nn.functional as F

def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
    
class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        self.model1 = nn.Sequential(
            torch.nn.Linear(in_planes, out_planes),
        )
        self.apply(init_weight)

    def forward(self, x):
        return self.model1(x)


class Jigsaw_cls(torch.nn.Module):
    
    def __init__(self, cls_num, input_c, output_c):
        super(Jigsaw_cls,self).__init__()
        
        self.cls_num = cls_num
        
        self.model = nn.Sequential(
            nn.Conv2d(input_c, input_c, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(input_c),
            nn.ReLU(),
            nn.Conv2d(input_c, output_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        

        self.classifier = nn.Sequential(
            nn.Linear(output_c, cls_num**4)
        )
        
    def _cls_num(self):
        return self.cls_num
        
    def forward(self, x):
        out = self.model(x)
        out = self.pool(out)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        
        return out