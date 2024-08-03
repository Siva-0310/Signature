import torch
import torch.nn as nn
from noiser import Noiser


class ConvBNRelu(nn.Module):
    def __init__(self,in_channels:int,out_channels:int) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        ) 

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    
class Encoder(nn.Module):
    def __init__(self,channels:int,layers:int,msg:int,H:int,W:int) -> None:
        super().__init__()

        self.H = H
        self.W = W

        self.in_ = ConvBNRelu(3,channels)
        self.features = nn.Sequential(
            *[ConvBNRelu(channels,channels) for _ in range(layers-1)]
        )
        self.msg_layer = ConvBNRelu(in_channels=channels+3+msg,out_channels=channels)
        self.out = nn.Conv2d(channels,3,kernel_size=3,padding=1)

    def forward(self,x:torch.Tensor,msg:torch.Tensor) -> torch.Tensor:

        exp_msg = msg.unsqueeze(-1)
        exp_msg.unsqueeze_(-1)
        exp_msg = exp_msg.expand(-1,-1, self.H, self.W)
        
        out = self.in_(x)
        out = self.features(out)
        out = torch.cat([out,x,exp_msg])
        out = self.msg_layer(out)
        out = self.out(out)
        return out

class Decoder(nn.Module):
    def __init__(self,channels:int,layers:int,msg:int) -> None:
        super().__init__()

        self.in_ = ConvBNRelu(3,channels)
        self.features = nn.Sequential(
            *[ConvBNRelu(channels,channels)  for _ in range(layers-1)]
        )
        self.msg_layer = nn.Sequential(
            ConvBNRelu(channels,msg),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
        )
        self.out = nn.Sequential(
            nn.Linear(msg,msg),
            nn.Sigmoid()
        )
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        out = self.in_(x)
        out = self.features(out)
        out = self.msg_layer(out)
        out = self.out(out)
        return out


class Signature(nn.Module):
    def __init__(self,channels:int,layers:int,msg:int,H:int,W:int) -> None:
        super().__init__()

        self.encoder = Encoder(channels,layers,msg,H,W)
        self.noiser = Noiser()
        self.decoder = Decoder(channels,layers,msg)
    
    def forward(self,img:torch.Tensor,msg:torch.Tensor) -> torch.Tensor:

        encoded_img = self.encoder(img,msg)
        noised_img = self.noiser(encoded_img)
        decoded_msg = self.decoder(noised_img)

        return encoded_img,noised_img,decoded_msg
