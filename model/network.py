import os
import torch
import torch.nn.functional as F
from model.config import TrainConfig,ModelConfig
from model.enc_dec import EncoderDecoder
from model.discriminator.model import NLayerDiscriminator
from model.losses.lpips import LPIPS

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

        with torch.no_grad():
            self.optimizer_g.zero_grad()

            recon_images,noised_images,recon_messages = self.model(images,messages.clone())
            disc_fake_pred = self.disc(recon_images)
            disc_fake_loss = F.mse_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
            disc_part = self.train.disc * disc_fake_loss
            lpips_loss = self.train.lpips*torch.mean(self.lpips(recon_images, images))
            message_recon_loss = F.binary_cross_entropy_with_logits(recon_messages,messages)
            images_recon_loss = F.mse_loss(recon_images,images)
            losses["lpips_loss"] = lpips_loss.item()
            losses["disc_part"] = disc_part.item()
            losses["message_recon_loss"] = message_recon_loss.item()
            losses["image_recon_loss"] = images_recon_loss.item()

            loss = message_recon_loss + images_recon_loss + lpips_loss + disc_part
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

            return losses,(recon_images,noised_images,recon_messages)
        
    def validate_on_batch(self, batch: list):

        images, messages = batch

        self.model.eval()
        self.disc.eval()
        losses = {}
        with torch.no_grad():

            recon_images,noised_images,recon_messages = self.model(images,messages.clone())
            disc_fake_pred = self.disc(recon_images)
            disc_fake_loss = F.mse_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
            disc_part = self.train.disc * disc_fake_loss
            lpips_loss = self.train.lpips*torch.mean(self.lpips(recon_images, images))
            message_recon_loss = F.binary_cross_entropy_with_logits(recon_messages,messages)
            images_recon_loss = F.mse_loss(recon_images,images)
            losses["lpips_loss"] = lpips_loss.item()
            losses["disc_part"] = disc_part.item()
            losses["message_recon_loss"] = message_recon_loss.item()
            losses["image_recon_loss"] = images_recon_loss.item()

            loss = message_recon_loss + images_recon_loss + lpips_loss + disc_part

            disc_fake_pred = self.disc(recon_images.detach())
            disc_real_pred = self.disc(images)
            disc_fake_loss = F.mse_loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = F.mse_loss(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            losses["disc_loss"] = disc_loss.item()
            losses["total"] = loss.item()

        return losses,(recon_images,noised_images,recon_messages)
    
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