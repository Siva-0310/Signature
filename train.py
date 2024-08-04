import torch
import torch.nn.functional as F
from signature import Signature
from torch.optim.adam import Adam
from typing import Tuple

class Config:
    def __init__(self,channels:int,layers:int,msg:int,H:int,W:int,r:int,
                 attn:bool,lr:float) -> None:
        self.channels = channels
        self.layers = layers
        self.msg = msg
        self.H = H
        self.W = W
        self.r = r
        self.attn = attn
        self.lr = lr

class Model:
    def __init__(self,config:Config,device:torch.device) -> None:

        self.config = config
        self.device = device
        self.model = Signature(config.channels,config.layers,
                               config.msg,config.H,config.W,
                               config.r,config.attn).to(device=device)
        self.optimizer = Adam(self.model.parameters(),config.lr)
        self.msgloss = F.binary_cross_entropy
        self.imgloss = F.mse_loss
    
    def train_on_batch(self,imgs:torch.Tensor,msgs:torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:

        self.model.train()
        self.optimizer.zero_grad()
        imgs = imgs.to(device=self.device)
        msgs = msgs.to(device=self.device)

        if self.config.attn:
            encoded_img, noised_img, decoded_msg,att = self.model(imgs, msgs)
        else:
            encoded_img, noised_img, decoded_msg = self.model(imgs, msgs)
        
        img_loss = self.imgloss(encoded_img, imgs)
        msg_loss = self.msgloss(decoded_msg, msgs)
        loss = img_loss + msg_loss
        loss.backward()
        self.optimizer.step()

        return loss.item(),img_loss,msg_loss
    
    def evaluate_on_batch(self, imgs: torch.Tensor, msgs: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
        self.model.eval()
        imgs = imgs.to(device=self.device)
        msgs = msgs.to(device=self.device)
        with torch.no_grad():
            if self.config.attn:
                encoded_img, noised_img, decoded_msg, att = self.model(imgs, msgs)
            else:
                encoded_img, noised_img, decoded_msg = self.model(imgs, msgs)
            
            img_loss = self.imgloss(encoded_img, imgs)
            msg_loss = self.msgloss(decoded_msg, msgs)
            loss = img_loss + msg_loss
            
        return loss.item(),img_loss,msg_loss

if __name__ == "__main__":

    config = Config(
        channels=3,
        layers=5,
        msg=30,
        H=256,
        W=256,
        r=16,
        attn=True,
        lr=0.001
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Model(config, device)
    
    imgs = torch.rand(8, 3, 256, 256)
    msgs = torch.rand(8,30)
    
    loss, img_loss, msg_loss = model.train_on_batch(imgs, msgs)
    print(f"Training loss: {loss}, Image loss: {img_loss.item()}, Message loss: {msg_loss.item()}")
    
    val_loss, val_img_loss, val_msg_loss = model.evaluate_on_batch(imgs, msgs)
    print(f"Validation loss: {val_loss}, Image loss: {val_img_loss.item()}, Message loss: {val_msg_loss.item()}")