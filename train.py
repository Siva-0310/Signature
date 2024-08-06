import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from signature import Signature
from torch.optim.adam import Adam
from typing import Tuple
from torch.utils.data import DataLoader
class Config:
    def __init__(self,channels:int,enc_layers:int,dec_layers:int,msg:int,H:int,W:int,r:int,
                 attn:bool,lr:float,epochs:int) -> None:
        self.channels = channels
        self.enc = enc_layers
        self.dec = dec_layers
        self.msg = msg
        self.H = H
        self.W = W
        self.r = r
        self.attn = attn
        self.lr = lr
        self.epochs = epochs

class Model:
    def __init__(self,config:Config,device:torch.device) -> None:

        self.config = config
        self.device = device
        self.model = Signature(config.channels,config.enc,config.dec,
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
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader):
        # Initialize lists to collect data for CSV files
        train_losses = []
        val_losses = []

        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss, train_img_loss, train_msg_loss = 0.0, 0.0, 0.0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{self.config.epochs}", unit="batch") as pbar:
                for imgs, _ in train_loader:
                    msgs = torch.randint(0, 2, (imgs.size()[0], self.config.msg)).float()
                    loss, img_loss, msg_loss = self.train_on_batch(imgs, msgs)
                    train_loss += loss
                    train_img_loss += img_loss.item()
                    train_msg_loss += msg_loss.item()
                    pbar.set_postfix({"Train Loss": train_loss / (pbar.n + 1),
                                      "Train Image Loss": train_img_loss / (pbar.n + 1),
                                      "Train Message Loss": train_msg_loss / (pbar.n + 1)})
                    pbar.update(1)

            # Calculate average losses for the epoch
            train_loss /= len(train_loader)
            train_img_loss /= len(train_loader)
            train_msg_loss /= len(train_loader)
            train_losses.append([epoch + 1, train_loss, train_img_loss, train_msg_loss])

            # Validation phase
            self.model.eval()
            val_loss, val_img_loss, val_msg_loss = 0.0, 0.0, 0.0
            with tqdm(total=len(val_loader), desc="Validation", unit="batch") as pbar:
                for imgs, msgs in val_loader:
                    loss, img_loss, msg_loss = self.evaluate_on_batch(imgs, msgs)
                    val_loss += loss
                    val_img_loss += img_loss.item()
                    val_msg_loss += msg_loss.item()
                    pbar.set_postfix({"Val Loss": val_loss / (pbar.n + 1),
                                      "Val Image Loss": val_img_loss / (pbar.n + 1),
                                      "Val Message Loss": val_msg_loss / (pbar.n + 1)})
                    pbar.update(1)

            # Calculate average losses for the epoch
            val_loss /= len(val_loader)
            val_img_loss /= len(val_loader)
            val_msg_loss /= len(val_loader)
            val_losses.append([epoch + 1, val_loss, val_img_loss, val_msg_loss])

            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Image Loss: {train_img_loss:.4f}, Train Message Loss: {train_msg_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Image Loss: {val_img_loss:.4f}, Val Message Loss: {val_msg_loss:.4f}")

        # Save losses to CSV files using Pandas
        self._save_losses_to_csv("train_losses.csv", train_losses)
        self._save_losses_to_csv("val_losses.csv", val_losses)

    def _save_losses_to_csv(self, filename: str, losses: list):
        # Convert the list of losses to a DataFrame
        df = pd.DataFrame(losses, columns=["Epoch", "Loss", "Image Loss", "Message Loss"])
        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)
        
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
    

if __name__ == "__main__":

    config = Config(
        channels=3,
        enc_layers=4,
        dec_layers=8,
        msg=30,
        H=256,
        W=256,
        r=16,
        attn=True,
        lr=0.001,
        epochs=100
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Model(config, device)
    
    imgs = torch.rand(8, 3, 256, 256)
    msgs = torch.randint(0,2,(8,30)).float()
    
    loss, img_loss, msg_loss = model.train_on_batch(imgs, msgs)
    print(f"Training loss: {loss}, Image loss: {img_loss.item()}, Message loss: {msg_loss.item()}")
    
    val_loss, val_img_loss, val_msg_loss = model.evaluate_on_batch(imgs, msgs)
    print(f"Validation loss: {val_loss}, Image loss: {val_img_loss.item()}, Message loss: {val_msg_loss.item()}")