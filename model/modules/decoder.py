import torch
import torch.nn as nn
from typing import List
from model.modules.blocks import ConvGroupSiLU

class Decoder(nn.Module):
    def __init__(self,channels:List[int],depth:int,im_channels:int,num_groups:int) -> None:
        super(Decoder,self).__init__()
        
        self.conv_in = nn.Conv2d(in_channels=im_channels,out_channels=channels[0],kernel_size=1)
        self.layers = nn.Sequential(
            *[
                ConvGroupSiLU(channels[i],channels[i+1],num_groups=num_groups) 
                for i in range(depth-1)
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.out_layer = nn.Sequential(
            nn.Linear(in_features=channels[-1],out_features=1),
            nn.Sigmoid()
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        out = x
        out = self.conv_in(out)
        out = self.layers(out)
        out = self.avg_pool(out)
        out.squeeze_(3).squeeze_(2)
        return self.out_layer(out)