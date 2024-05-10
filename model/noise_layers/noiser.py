import torch
import torch.nn as nn

class Noiser(nn.Module):
    def __init__(self,type_:str) -> None:
        super(Noiser,self).__init__()

        self.layer = nn.Identity()

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.layer(x)      