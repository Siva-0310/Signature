import torch
import torch.nn as nn
from model.config import ModelConfig
from model.modules.unet import Unet
from model.modules.extractor import Extractor
from model.noise_layers.noiser import Noiser
from typing import List

class EncoderDecoder(nn.Module):
    def __init__(self,config:ModelConfig) -> None:
        super(EncoderDecoder,self).__init__()

        self.unet = Unet(depth=config.depth,channels=config.channels,
                         num_groups=config.num_groups,im_channels=config.im_channels,
                         message_length=config.message_length,H=config.H,W=config.W)
        
        self.extractor = Extractor(channels=config.ext_channels,depth=config.ext_depth,
                                   message_length=config.message_length,im_channels=config.im_channels,num_groups=config.num_groups)
        
        self.noiser = Noiser(type_=config.noiser)

    def forward(self,images:torch.Tensor,messages:torch.Tensor) -> List[torch.Tensor,torch.Tensor,torch.Tensor]:
        out = images
        recon_images = self.unet(out,messages)
        noised_images = self.noiser(recon_images)
        recon_messages = self.extractor(noised_images)

        return recon_images,noised_images,recon_messages
    
    def watermark(self,images:torch.Tensor,messages:torch.Tensor) -> torch.Tensor:
        return self.unet(images,messages)
    
    def extract(self,images:torch.Tensor) -> torch.Tensor:
        return self.extractor(images)

