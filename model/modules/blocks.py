import torch.nn as nn
import torch

class DownSample(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,num_groups:int,num_layers:int) -> None:
        super(DownSample,self).__init__()
        
        self.resnets_layers = nn.Sequential(
            *[
                ResNet(in_channels=in_channels if i == 0 else out_channels,out_channels=out_channels,num_groups=num_groups)
                for i in range(num_layers)
            ]
        )

        self.down_sample = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=4,stride=2,padding=1)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        out = x
        out = self.resnets_layers(out)
        return self.down_sample(out)
    
class UpSample(nn.Module):
    def __init__(self, in_channels:int,out_channels:int,num_groups:int,num_layers:int) -> None:
        super(UpSample,self).__init__()

        self.resnets_layers = nn.Sequential(
            *[
                ResNet(in_channels=in_channels if i == 0 else out_channels,out_channels=out_channels,num_groups=num_groups)
                for i in range(num_layers)
            ]
        )

        self.up_sample = nn.ConvTranspose2d(in_channels,in_channels,4,2,1)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        out = self.up_sample(x)
        return self.resnets_layers(out)
    
class MidBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,num_layers:int,num_groups:int) -> None:
        super(MidBlock,self).__init__()

        self.resnet_1 = ResNet(in_channels=in_channels,out_channels=out_channels,num_groups=num_groups)
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    AttnBlock(num_groups=num_groups,in_channels=out_channels),
                    ResNet(in_channels=out_channels,out_channels=out_channels,num_groups=num_groups)
                )
                for _ in range(num_layers)
            ]
        )
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        out = self.resnet_1(x)
        return self.layers(out)
    
class ResNet(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,num_groups:int) -> None:
        super(ResNet,self).__init__()

        self.resnet_conv_1 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups,num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        )

        self.resnet_conv_2 = nn.Sequential(
                nn.GroupNorm(num_groups=num_groups,num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
            )
        self.residual = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        out = x
        out = self.resnet_conv_1(out)
        out = self.resnet_conv_2(out)
        return out + self.residual(x)


# class Attention(nn.Module):
#     def __init__(self, num_groups:int,out_channels:int,num_heads:int) -> None:
#         super(Attention,self).__init__()

#         self.attention_norm = nn.GroupNorm(num_groups=num_groups,num_channels=out_channels)
#         self.attention_layer = nn.MultiheadAttention(num_heads=num_heads,embed_dim=out_channels,batch_first=True)

#     def forward(self,x:torch.Tensor) -> torch.Tensor:
#         out = x

#         batch_size, channels, h, w = out.shape
#         in_attn = out.reshape(batch_size, channels, h * w)
#         in_attn = self.attention_norm(in_attn)
#         in_attn = in_attn.transpose(1, 2)
#         out_attn, _ = self.attention_layer(in_attn, in_attn, in_attn)
#         out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
#         out = out + out_attn

#         return out
    
class AttnBlock(nn.Module):
    def __init__(self, in_channels,num_groups):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=num_groups,num_channels=in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_
