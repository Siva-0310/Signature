import torch
import torch.nn as nn

class Noiser(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layer = nn.Identity()

    def forward(self,x:torch.Tensor):
        return self.layer(x)