import torch.nn as nn
import torch
from model.modules.blocks import DownSample,MidBlock,ResNet

class Extractor(nn.Module):
    def __init__(self,down_channels,mid_channels,num_layers,im_channels,num_groups,message_length) -> None:
        super(Extractor,self).__init__()
        
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.num_groups = num_groups
        self.num_layers = num_layers
        self.im_channels = im_channels
        self.message_length = message_length

        self.conv_in = nn.Conv2d(in_channels=self.im_channels,out_channels=self.down_channels[0],kernel_size=1,stride=1)

        self.down = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.down.append(
                DownSample(in_channels=self.down_channels[i],out_channels=self.down_channels[i+1],
                           num_groups=self.num_groups,num_layers=self.num_layers)
            )

        self.mid1 = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mid1.append(
                MidBlock(in_channels=self.mid_channels[i],out_channels=self.mid_channels[i+1],num_groups=self.num_groups,
                         num_layers=self.num_layers)
            )
        self.out_layer =  nn.Sequential(
            ResNet(in_channels=self.mid_channels[-1],out_channels=self.message_length,num_groups=1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.message_layer = nn.Sequential(
              nn.Linear(self.message_length,self.message_length),
        )

    def forward(self,x):
        out = x
        out = self.conv_in(out)
        for i in self.down:
            out = i(out)
        for i in self.mid1:
            out = i(out)
        out = self.out_layer(out)
        out.squeeze_(3).squeeze_(2)
        return self.message_layer(out)