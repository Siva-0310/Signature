import torch.nn as nn
import torch
from model.noise_layers.utils import jpeg_compress
from kornia.augmentation import AugmentationBase2D

class DiffJPEG(nn.Module):
    def __init__(self,quality,device) -> None:
        super(DiffJPEG,self).__init__()
        self.quality = quality
        self.device = device

    def forward(self,x:torch.Tensor):
        with torch.no_grad():
            img_clip = x.clamp(0,1)
            img_jpeg = jpeg_compress(img=img_clip,quality=self.quality,device=self.device)
            img_gap = img_jpeg - x
            img_gap = img_gap.detach()
        img_aug = x+img_gap
        return img_aug
    
class RandomDiffJPEG(AugmentationBase2D):
    def __init__(self,device, p: float = 0.5,low=10,high=100) -> None:
        super().__init__(p=p)
        self.diff_jpegs = [DiffJPEG(quality=qf,device=device).to(device=device) for qf in range(low,high,10)]

    def generate_parameters(self, input_shape: torch.Size):
        qf = torch.randint(high=len(self.diff_jpegs), size=input_shape[0:1])
        return dict(qf=qf)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        qf = params['qf']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.diff_jpegs[qf[ii]](input[ii:ii+1])
        return output
