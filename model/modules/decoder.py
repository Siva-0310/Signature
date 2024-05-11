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
        self.out_layer = nn.Sequential(
            nn.GroupNorm(num_channels=channels[-1],num_groups=num_groups),
            nn.SiLU(),
            nn.Conv2d(in_channels=channels[-1],out_channels=1)
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        out = x
        out = self.conv_in(out)
        out = self.layers(out)
        return self.out_layer(out)