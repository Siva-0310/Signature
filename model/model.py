import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from model.discriminator.model import NLayerDiscriminator
from model.losses.lpips import LPIPS
from model.noise_layers.noiser import Noiser
from model.modules.blocks import Unet,Extractor

class ModelNN(nn.Module):
    def __init__(self,config,noiser:Noiser) -> None:
        super(ModelNN,self).__init__()
        self.message_length = config["message_length"]
        self.im_channels = config["im_channels"]
        self.num_groups = config["num_groups"]
        self.H = config["H"]
        self.W = config["W"]
        self.channels = config["channels"]

        self.unet = Unet(
            message_length=self.message_length,num_groups=self.num_groups,H=self.H,W=self.W,
            im_channels=self.im_channels,channels=self.channels,depth=len(self.channels)
        )

        self.noiser = noiser

        self.ext = Extractor(
            message_length=self.message_length,num_groups=self.num_groups,H=self.H,W=self.W,
            im_channels=self.im_channels,channels=self.channels,depth=len(self.channels)
        )

    def forward(self,x,message):
        out,loss = self.unet(x,message)
        noised_out = self.noiser(out)
        recon_message = self.ext(noised_out)
        return out,recon_message,loss,noised_out
    

class Model:
    def __init__(self,config,train_config,device,noiser) -> None:

        self.lr = train_config['lr']
        self.lpips_weight = train_config["lpips_weight"]
        self.disc_weight = train_config["disc_weight"]
        self.message_recon_weight = train_config["message_recon_weight"]

        self.model = ModelNN(config=config,noiser=noiser).to(device=device)
        self.disc = NLayerDiscriminator().to(device=device)
        self.lpips = LPIPS().eval().to(device=device)
        self.optimizer_d = torch.optim.Adam(self.disc.parameters(),lr=self.lr)
        self.optimizer_g = torch.optim.Adam(self.model.parameters(),lr=self.lr)


    def train_on_batch(self, batch: list):
        
        images, messages = batch

        batch_size = images.shape[0]
        self.model.train()
        self.disc.train()
        losses = {}

        with torch.enable_grad():
            self.optimizer_g.zero_grad()

            recon_images,recon_message,loss,noised_img = self.model(images,messages.clone())
            disc_fake_pred = self.disc(recon_images)
            disc_fake_loss = F.mse_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
            disc_part = self.disc_weight * disc_fake_loss
            lpips_loss = self.lpips_weight*torch.mean(self.lpips(recon_images, images))
            message_recon_loss = self.message_recon_weight*F.binary_cross_entropy_with_logits(recon_message,messages)
            losses["lpips_loss"] = lpips_loss.item()
            losses["disc_part"] = disc_part.item()
            losses["message_recon_loss"] = message_recon_loss.item()
            losses["image_recon_loss"] = loss.item()

            loss += message_recon_loss + lpips_loss + disc_part
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

        return losses,(recon_images,recon_message,noised_img)
    
    def validate_on_batch(self, batch: list):

        images, messages = batch
        batch_size = images.shape[0]

        self.model.eval()
        self.disc.eval()
        losses = {}
        with torch.no_grad():

            recon_images,recon_message,loss,noised_img = self.model(images,messages.clone())
            disc_fake_pred = self.disc(recon_images)
            disc_fake_loss = F.mse_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
            disc_part = self.disc_weight * disc_fake_loss
            lpips_loss = self.lpips_weight*torch.mean(self.lpips(recon_images, images))
            message_recon_loss = self.message_recon_weight*F.binary_cross_entropy_with_logits(recon_message,messages)
            losses["lpips_loss"] = lpips_loss.item()
            losses["disc_part"] = disc_part.item()
            losses["message_recon_loss"] = message_recon_loss.item()
            losses["image_recon_loss"] = loss.item()

            loss += message_recon_loss + lpips_loss + disc_part

            disc_fake_pred = self.disc(recon_images.detach())
            disc_real_pred = self.disc(images)
            disc_fake_loss = F.mse_loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = F.mse_loss(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            losses["disc_loss"] = disc_loss.item()
            losses["total"] = loss.item()
        
        return losses,(recon_images,recon_message,noised_img)
    
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