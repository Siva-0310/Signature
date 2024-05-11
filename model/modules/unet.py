import torch
import torch.nn as nn
from model.modules.blocks import DownSample,UpSample,ConvGroupSiLU

class Unet(nn.Module):
    def __init__(self,depth:int,channels:list,num_groups:int,im_channels:int,message_length:int,H:int,W:int) -> None:
        super(Unet,self).__init__()
        
        self.H = H
        self.W = W

        self.conv_in = nn.Conv2d(in_channels=im_channels+message_length,out_channels=channels[0],kernel_size=1)

        self.down_sample = nn.ModuleList([
            DownSample(in_channels=channels[i],out_channels=channels[i+1],num_groups=num_groups)
            for i in range(depth-1)
        ])

        self.mid = ConvGroupSiLU(in_channels=channels[-1],out_channels=channels[-1],num_groups=1)

        self.up_sample = nn.ModuleList([
            UpSample(in_channels=channels[i]*2,out_channels=channels[i-1],num_groups=num_groups)
            for i in range(depth-1,0,-1)
        ])

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups,num_channels=channels[0]*2),
            nn.SiLU(),
            nn.Conv2d(in_channels=channels[0]*2,out_channels=im_channels,kernel_size=1),
        )

    def info(self,x:torch.Tensor) -> torch.Tensor:
        x.unsqueeze_(-1)
        x.unsqueeze_(-1)
        return x.expand(-1,-1,self.H,self.W)

    def forward(self,x:torch.Tensor,message:torch.Tensor) -> torch.Tensor:
        out = x
        out = torch.cat([out,self.info(message)],dim=1)
        out = self.conv_in(out)
        downsample = [out]
        for block in self.down_sample:
            out = block(out)
            downsample.append(out)
        out = self.mid(out)
        for block in self.up_sample:
            out = torch.cat([out,downsample.pop()],dim=1)
            out = block(out)
        out = torch.cat([out,downsample.pop()],dim=1)
        out = self.conv_out(out)
        return out
    
    