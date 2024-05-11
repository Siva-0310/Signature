import torch
import torch.nn as nn
from typing import List
from model.modules.blocks import ConvGroupSiLU

class Encoder(nn.Module):
    def __init__(self,im_channels:int,channels:List[int],depth:int,num_groups:int) -> None:
        super(Encoder,self).__init__()

        self.conv_in = nn.Conv2d(in_channels=im_channels+1,out_channels=channels[0],kernel_size=1)

        self.layers = nn.Sequential(
            *[
                ConvGroupSiLU(in_channels=channels[i],out_channels=channels[i+1],num_groups=num_groups)
                for i in range(depth-1)
            ]
        )
        
        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups,num_channels=channels[0]*2),
            nn.SiLU(),
            nn.Conv2d(in_channels=channels[0]*2,out_channels=im_channels,kernel_size=1),
        )

    def forward(self,x:torch.Tensor,message:torch.Tensor) -> torch.Tensor:
        out = x
        out = torch.cat([out,message],dim=1)
        store = out.clone()
        out = self.conv_in(out)
        out = self.layers(out)
        out = torch.cat([out,store],dim=1)
        return self.conv_out(out)