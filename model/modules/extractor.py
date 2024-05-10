import torch
import torch.nn as nn
from typing import List
from model.modules.blocks import ConvGroupSiLU

class Extractor(nn.Module):
    def __init__(self,channels:List[int],depth:int,message_length:int,im_channels:int,num_groups:int) -> None:
        super(Extractor,self).__init__()
        
        self.in_layer = nn.Conv2d(in_channels=im_channels,out_channels=channels[-1],kernel_size=1)
        self.layers = nn.Sequential(
            *[
                ConvGroupSiLU(channels[i],channels[i+1],num_groups=num_groups) for i in range(depth-1)
            ]
        )
        self.out_layer = nn.Sequential(
            ConvGroupSiLU(in_channels=channels[-1],out_channels=message_length,num_groups=1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.message_layer = nn.Sequential(
              nn.Linear(message_length,message_length),
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        out = x
        out = self.in_layer(out)
        out = self.layer(out)
        out = self.out_layer(out)
        out.squeeze_(3).squeeze_(2)
        return self.message_layer(out)