import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import ModelConfig,TrainConfig
from model.modules.encoder import Encoder
from model.modules.decoder import Decoder
from model.noise_layers.noiser import Noiser
from model.discriminator.model import NLayerDiscriminator
from model.losses.lpips import LPIPS
from typing import List,Tuple

class EncoderDecoder(nn.Module):
    def __init__(self,config:ModelConfig) -> None:
        super(EncoderDecoder,self).__init__()

        self.encoder = Encoder(depth=config.depth,channels=config.channels,
                         num_groups=config.num_groups,im_channels=config.im_channels,)
        
        self.decoder = Decoder(channels=config.ext_channels,depth=config.ext_depth,
                                   im_channels=config.im_channels,num_groups=config.num_groups)
        
        self.noiser = Noiser(type_=config.noiser)

    def forward(self,images:torch.Tensor,message:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        out = images
        recon_images = self.encoder(out,message)
        noised_images = self.noiser(recon_images)
        real_preds = self.decoder(noised_images)
        fake_preds = self.decoder(images)

        return recon_images,noised_images,real_preds,fake_preds
    
    def watermark(self,images:torch.Tensor,message:torch.Tensor) -> torch.Tensor:
        return self.encoder(images,message)
    
    def extract(self,images:torch.Tensor) -> torch.Tensor:
        return self.decoder(images)
    


class Network:
    def __init__(self,model_config:ModelConfig,train_config:TrainConfig,device:torch.device) -> None:
        
        self.train = train_config

        self.model = EncoderDecoder(model_config).to(device=device)
        self.disc = NLayerDiscriminator().to(device=device)
        self.lpips = LPIPS().eval().to(device=device)
        self.optimizer_d = torch.optim.Adam(self.disc.parameters(),lr=self.train.lr)
        self.optimizer_g = torch.optim.Adam(self.model.parameters(),lr=self.train.lr)

    
    def train_on_batch(self, batch: list):
        
        images,messages = batch
        
        self.model.train()
        self.disc.train()
        losses = {}

        with torch.enable_grad():
            self.optimizer_g.zero_grad()

            recon_images,noised_images,real_preds,fake_preds = self.model(images,messages)
            disc_fake_pred = self.disc(recon_images)
            disc_fake_loss = F.mse_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
            disc_part = self.train.disc * disc_fake_loss
            lpips_loss = self.train.lpips*torch.mean(self.lpips(recon_images, images))
            images_recon_loss = F.mse_loss(recon_images,images)
            message_detect_loss = (F.binary_cross_entropy(input=real_preds,target=torch.ones_like(real_preds)) + F.binary_cross_entropy(input=fake_preds,target=torch.zeros_like(fake_preds)))/2
            losses["lpips_loss"] = lpips_loss.item()
            losses["disc_part"] = disc_part.item()
            losses["message_detect_loss"] = message_detect_loss.item()
            losses["image_recon_loss"] = images_recon_loss.item()

            loss = message_detect_loss + images_recon_loss + lpips_loss + disc_part
            loss.backward()
            self.optimizer_g.step()

            self.optimizer_d.zero_grad()
            disc_fake_pred = self.disc(recon_images.detach())
            disc_real_pred = self.disc(images)
            disc_fake_loss = F.mse_loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = F.mse_loss(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            self.optimizer_d.step()

            losses["disc_loss"] = disc_loss.item()
            losses["total"] = loss.item()

            return losses,(recon_images,noised_images,real_preds,fake_preds)

        
    def validate_on_batch(self, batch: list):

        images, messages = batch

        self.model.eval()
        self.disc.eval()
        losses = {}
        with torch.no_grad():

            recon_images,noised_images,real_preds,fake_preds = self.model(images,messages)
            disc_fake_pred = self.disc(recon_images)
            disc_fake_loss = F.mse_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
            disc_part = self.train.disc * disc_fake_loss
            lpips_loss = self.train.lpips*torch.mean(self.lpips(recon_images, images))
            images_recon_loss = F.mse_loss(recon_images,images)
            message_detect_loss = (F.binary_cross_entropy(input=real_preds,target=torch.ones_like(real_preds)) + F.binary_cross_entropy(input=fake_preds,target=torch.zeros_like(fake_preds)))/2
            losses["lpips_loss"] = lpips_loss.item()
            losses["disc_part"] = disc_part.item()
            losses["message_detect_loss"] = message_detect_loss.item()
            losses["image_recon_loss"] = images_recon_loss.item()

            loss = message_detect_loss + images_recon_loss + lpips_loss + disc_part

            disc_fake_pred = self.disc(recon_images.detach())
            disc_real_pred = self.disc(images)
            disc_fake_loss = F.mse_loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = F.mse_loss(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            losses["disc_loss"] = disc_loss.item()
            losses["total"] = loss.item()

        return losses,(recon_images,noised_images,real_preds,fake_preds)
    
    def save_state_dict(self,path,num):
        torch.save(self.model.state_dict(),os.path.join(path,f"model_{num}.pth"))
        torch.save(self.disc.state_dict(),os.path.join(path,f"disc_{num}.pth"))
    
    def load_state_dict(self,path,num,device):
        self.model.load_state_dict(torch.load(os.path.join(path,f"model_{num}.pth"),map_location=device))
        self.disc.load_state_dict(torch.load(os.path.join(path,f"disc_{num}.pth"),map_location=device))

    def predict(self,x:torch.Tensor):
        x = F.sigmoid(x)
        x = x >= 0.5
        return x.int()
    
    def watermark(self,batch:list):
        images,messages = batch
        with torch.no_grad():
            recon_image,loss = self.model.watermark(images,messages)
        return recon_image,loss
    
    def extract(self,images):
        with torch.no_grad():
            recon_messages = self.model.extract(images)
        return recon_messages