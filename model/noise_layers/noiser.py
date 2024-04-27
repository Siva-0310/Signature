import torch.nn as nn
import kornia.augmentation as K
from model.noise_layers.jpeg import RandomDiffJPEG
class Noiser(nn.Module):
    def __init__(self,type_:str,device,jpeg_p,jpeg_low) -> None:
        super(Noiser,self).__init__()

        if type_ == "linear":
            self.noiser = nn.Identity().to(device=device)
        elif type_ == "jpeg":
            self.noiser = RandomDiffJPEG(device=device,p=jpeg_p,low=jpeg_low).to(device=device)

    def forward(self,x):
        return self.noiser(x)