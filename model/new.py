
import torch
import torch.nn as nn 
import torch.nn.functional as F

class ConvGroupSiLU(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,num_groups:int) -> None:
        super(ConvGroupSiLU,self).__init__()

        self.layer = nn.Sequential(
            nn.GroupNorm(num_channels=in_channels,num_groups=num_groups),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=1),
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    
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
    

class Unet(nn.Module):
    def __init__(self,depth:int,channels:list,num_groups:int,im_channels:int,message_length:int,H:int,W:int) -> None:
        super(Unet,self).__init__()
        
        self.H = H
        self.W = W

        self.conv_in = nn.Conv2d(in_channels=im_channels,out_channels=channels[0],kernel_size=1)

        self.down_sample = nn.ModuleList([
            DownSample(in_channels=channels[i],out_channels=channels[i+1],num_groups=num_groups)
            for i in range(depth-1)
        ])

        self.mid = ConvGroupSiLU(in_channels=channels[-1]+message_length,out_channels=channels[-1],num_groups=1)

        self.up_sample = nn.ModuleList([
            UpSample(in_channels=channels[i]*2,out_channels=channels[i-1],num_groups=num_groups)
            for i in range(depth-1,0,-1)
        ])

        self.conv_out = ConvGroupSiLU(in_channels=channels[0],out_channels=im_channels,num_groups=num_groups,)

    def info(self,x:torch.Tensor) -> torch.Tensor:
        x.unsqueeze_(-1)
        x.unsqueeze_(-1)
        return x.expand(-1,-1,self.H,self.W)

    def forward(self,x:torch.Tensor,message:torch.Tensor) -> torch.Tensor:
        out = x
        out = self.conv_in(out)
        downsample = []
        for block in self.down_sample:
            out = block(out)
            downsample.append(out)
        out = torch.cat([out,self.info(message)],dim=1)
        out = self.mid(out)
        for block in self.up_sample:
            out = torch.cat([out,downsample.pop()],dim=1)
            out = block(out)
        out = self.conv_out(out)
        return out,self.loss(recon_im=out,im=x)
    
    def loss(self,recon_im:torch.Tensor,im:torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon_im,im)
    
class Extractor(nn.Module):
    def __init__(self,depth:int,channels:list,num_groups:int,im_channels:int,message_length:int,H:int,W:int) -> None:
        super(Extractor,self).__init__()

        self.H = H
        self.W = W

        self.conv_in = nn.Conv2d(in_channels=im_channels,out_channels=channels[0],kernel_size=1)

        self.down_sample = nn.ModuleList([
            DownSample(in_channels=channels[i],out_channels=channels[i+1],num_groups=num_groups)
            for i in range(depth-1)
        ])

        self.out_layer =  nn.Sequential(
            ConvGroupSiLU(in_channels=channels[-1],out_channels=message_length,num_groups=1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.message_layer = nn.Sequential(
              nn.Linear(message_length,message_length),
        )
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        out = x
        out = self.conv_in(out)
        for i in self.down_sample:
            out = i(out)
        out = self.out_layer(out)
        out.squeeze_(3).squeeze_(2)
        return self.message_layer(out)