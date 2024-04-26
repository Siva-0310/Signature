import torch.nn as nn
import torch
import torch.nn.functional as F
from model.modules.blocks import DownSample,MidBlock,UpSample,ResNet

class Unet(nn.Module):
    def __init__(self,H,W,down_channels,mid_channels,up_channels,num_layers,im_channels,num_groups,message_length) -> None:
        super(Unet,self).__init__()
        self.H = H
        self.W = W
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.up_channels = up_channels
        self.num_layers = num_layers
        self.im_channels = im_channels
        self.num_groups = num_groups
        self.message_length = message_length

        self.conv_in = nn.Conv2d(in_channels=self.im_channels,out_channels=self.down_channels[0],kernel_size=1,stride=1)

        self.down = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.down.append(
                DownSample(in_channels=self.down_channels[i],out_channels=self.down_channels[i+1],
                           num_layers=self.num_layers,num_groups=self.num_groups)
            )

        self.mid1 = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mid1.append(
                MidBlock(in_channels=self.mid_channels[i],out_channels=self.mid_channels[i+1],
                         num_layers=self.num_layers,num_groups=self.num_groups)
            )
        self.mid = ResNet(in_channels=self.mid_channels[-1]+self.message_length,out_channels=self.mid_channels[-1],num_groups=1)
        self.mid2 = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1,0,-1):
            self.mid2.append(
                MidBlock(in_channels=self.mid_channels[i],out_channels=self.mid_channels[i-1],
                         num_layers=self.num_layers,num_groups=self.num_groups)
            )
        self.up = nn.ModuleList([])
        for i in range(len(self.up_channels)-1):
            self.up.append(
                UpSample(in_channels=self.up_channels[i]*2,out_channels=self.up_channels[i+1],
                         num_layers=self.num_layers,num_groups=self.num_groups)
            )

        self.norm_out = nn.GroupNorm(num_channels=self.up_channels[-1],num_groups=self.num_groups)
        self.conv_out = nn.Conv2d(self.up_channels[-1], self.im_channels, kernel_size=3, padding=1)

    def info(self,x:torch.Tensor):
        x.unsqueeze_(-1)
        x.unsqueeze_(-1)
        return x.expand(-1,-1,self.H,self.W)

    def forward(self,x:torch.Tensor,message:torch.Tensor) -> torch.Tensor:
        out = x
        out = self.conv_in(x)
        down_outs = []
        for i in self.down:
            out = i(out)
            down_outs.append(out)
        for i in self.mid1:
            out = i(out)
        out = torch.cat([out,self.info(message)],dim=1)
        out = self.mid(out)
        for i in self.mid2:
            out = i(out)
        for i in self.up:
            in_ = torch.cat([out,down_outs.pop()],dim=1)
            out = i(in_)
        out = self.norm_out(out)
        out = self.conv_out(out)
        return out,self.loss(out,x)
    
    def loss(self,recon_im,im):
        return F.mse_loss(recon_im,im)