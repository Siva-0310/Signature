import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self,in_channels:int,r:int) -> None:
        super().__init__()

        self.mlp =nn.Sequential(
            nn.Linear(in_channels,in_channels//r),
            nn.ReLU(),
            nn.Linear(in_channels//r,in_channels)
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        b,c,_,_ = x.shape

        avg = F.adaptive_avg_pool2d(x,output_size=1)
        max = F.adaptive_max_pool2d(x,output_size=1)

        avg = self.mlp(avg.view(b,c)).view(b,c,1,1)
        max = self.mlp(max.view(b,c)).view(b,c,1,1)

        out = F.sigmoid(avg + max)
        return x*out,out

class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=7,padding=3)

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)

        out = torch.cat((max,avg), dim=1)
        out = self.conv(out)
        out = F.sigmoid(out)

        return out*x,out

class CBAM(nn.Module):
    def __init__(self,channels:int,r:int) -> None:
        super().__init__()

        self.cab = ChannelAttention(channels,r)
        self.sab = SpatialAttention()

    def forward(self, x):
        out,c = self.cab(x)
        out,s= self.sab(out)
        return out+x,c,s

if __name__ == "__main__":

    cbam = CBAM(64, 16)
    x = torch.rand((8, 64, 256, 256))
    out,c,s = cbam(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Channel attention output shape:", c.shape)
    print("Spatial attention output shape:", s.shape)