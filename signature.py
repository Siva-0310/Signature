import torch
import torch.nn as nn
from cbam import CBAM
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
        return self.layer(x)

    
class Encoder(nn.Module):
    def __init__(self,channels:int,layers:int,msg:int,H:int,W:int,r:int=16,attn:bool = True) -> None:
        super().__init__()

        self.H = H
        self.W = W
        self.attn = attn

        self.in_ = ConvBNRelu(3,channels)
        self.features = nn.Sequential(
            *[ConvBNRelu(channels,channels) for _ in range(layers-1)]
        )
        self.msg_layer = ConvBNRelu(in_channels=channels+3+msg,out_channels=channels)
        self.out = nn.Conv2d(channels,3,kernel_size=3,padding=1)

        self.cbam = CBAM(channels=channels,r=r)

    def forward(self,x:torch.Tensor,msg:torch.Tensor) -> torch.Tensor:

        exp_msg = msg.unsqueeze(-1)
        exp_msg.unsqueeze_(-1)
        exp_msg = exp_msg.expand(-1,-1, self.H, self.W)
        
        out = self.in_(x)
        out = self.features(out)
        c,s = None,None
        if self.attn:
            out,c,s= self.cbam(out)
        out = torch.cat([out,x,exp_msg],dim=1)
        out = self.msg_layer(out)
        out = self.out(out)
        if self.attn:
            return out,c,s
        return out

class Decoder(nn.Module):
    def __init__(self,channels:int,layers:int,msg:int,r:int=16,attn:bool = True) -> None:
        super().__init__()

        self.attn = attn

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

        self.cbam = CBAM(channels=channels,r=r)

    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        out = self.in_(x)
        out = self.features(out)
        c,s = None,None
        if self.attn:
            out,c,s = self.cbam(out)
        out = self.msg_layer(out)
        out = out.squeeze(-1).squeeze(-1)
        out = self.out(out)
        if self.attn:
            return out,c,s
        return out


class Signature(nn.Module):
    def __init__(self,channels:int,enc:int,dec:int,msg:int,H:int,W:int,r:int=16,attn:bool = True) -> None:
        super().__init__()

        self.attn = attn

        self.encoder = Encoder(channels,enc,msg,H,W,r,attn)
        self.noiser = Noiser()
        self.decoder = Decoder(channels,dec,msg,r,attn)
    
    def forward(self,img:torch.Tensor,msg:torch.Tensor) -> torch.Tensor:
        if self.attn:
            encoded_img,ec,es = self.encoder(img,msg)
            noised_img = self.noiser(encoded_img)
            decoded_msg,dc,ds = self.decoder(noised_img)
            return encoded_img,noised_img,decoded_msg,(ec,es,dc,ds)
        else:
            encoded_img = self.encoder(img,msg)
            noised_img = self.noiser(encoded_img)
            decoded_msg = self.decoder(noised_img)
            return encoded_img,noised_img,decoded_msg


if __name__ == "__main__":

    img = torch.rand(8, 3, 256, 256)
    msg = torch.randint(0,2,(8, 30)).float()

    sig = Signature(64, 4, 30, 256, 256,attn=False)
    encoded_img, noised_img, decoded_msg = sig(img, msg)

    print("Encoded Image Shape:", encoded_img.shape)
    print("Noised Image Shape:", noised_img.shape)
    print("Decoded Message Shape:", decoded_msg.shape)