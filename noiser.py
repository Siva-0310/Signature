import torch
import torch.nn as nn
import kornia as K

class Noiser(nn.Module):
    def __init__(self, noise_type: str = "none", p: float = 0.5) -> None:
        super().__init__()
        
        self.p = p

        if noise_type == "gaussian":
            self.noise_layer = GaussianNoiseLayer(std=0.1)
        elif noise_type == "dropout":
            self.noise_layer = DropoutNoiseLayer(p=0.1)
        elif noise_type == "jpeg":
            self.noise_layer = JpegCompressionLayer(quality=75)
        elif noise_type == "crop":
            self.noise_layer = RandomResizedCropLayer(scale=(0.8, 1.0))
        elif noise_type == "blur":
            self.noise_layer = GaussianBlurLayer(kernel_size=(5, 5), sigma=(1.5, 1.5))
        else:
            self.noise_layer = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.rand(1).item() < self.p:
            x = self.noise_layer(x)
        return x

class GaussianNoiseLayer(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std
        return x + noise

class DropoutNoiseLayer(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.dropout = nn.Dropout2d(p=p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)

class JpegCompressionLayer(nn.Module):
    def __init__(self, quality=75):
        super().__init__()
        self.quality = quality

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        x = K.enhance.adjust_jpeg_quality(x, self.quality)
        return x

class RandomResizedCropLayer(nn.Module):
    def __init__(self, scale=(0.8, 1.0)):
        super().__init__()
        self.transform = K.augmentation.RandomResizedCrop(size=(256, 256), scale=scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

class GaussianBlurLayer(nn.Module):
    def __init__(self, kernel_size=(5, 5), sigma=(1.5, 1.5)):
        super().__init__()
        self.transform = K.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=sigma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

