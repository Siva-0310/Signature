
import torch
import torch.nn as nn 
import torch.nn.functional as F

class ConvGroupSiLU(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,num_groups:int) -> None:
        super(ConvGroupSiLU,self).__init__()

        self.layer1 = nn.Sequential(
            nn.GroupNorm(num_channels=in_channels,num_groups=num_groups),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1),
        )

        self.layer2 = nn.Sequential(
            nn.GroupNorm(num_channels=out_channels,num_groups=num_groups),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1),
        )

        self.residual = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1) if in_channels != out_channels else nn.Identity()


    def forward(self,x:torch.Tensor) -> torch.Tensor:
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = out + self.residual(x)
        return out
    
class DownSample(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,num_groups:int) -> None:
        super(DownSample,self).__init__()

        self.layer = nn.Sequential(
            ConvGroupSiLU(in_channels=in_channels,out_channels=out_channels,num_groups=num_groups),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1)
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    

class UpSample(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,num_groups:int) -> None:
        super(UpSample,self).__init__()

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels,in_channels,4,2,1),
            ConvGroupSiLU(in_channels=in_channels,out_channels=out_channels,num_groups=num_groups),
        )
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.layer(x)