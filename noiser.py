import torch
import torch.nn as nn
import torchvision.transforms as T

class Noiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.noise_layers = nn.ModuleList([
            GaussianNoiseLayer(std=0.1),
            DropoutNoiseLayer(p=0.1),
            JpegCompressionLayer(quality=75)
        ])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # for layer in self.noise_layers:
        #     x = layer(x)
        return x

class GaussianNoiseLayer(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class DropoutNoiseLayer(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(p=p)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.dropout(x)

class JpegCompressionLayer(nn.Module):
    def __init__(self, quality=75):
        super().__init__()
        self.quality = quality
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Lambda(lambda img: img.convert("RGB")),
            T.Lambda(lambda img: T.functional.adjust_jpeg_quality(img, self.quality)),
            T.ToTensor()
        ])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.training:
            x = torch.stack([self.transform(img) for img in x])
        return x

