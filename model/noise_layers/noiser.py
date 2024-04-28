import torch
import torch.nn as nn
import kornia.augmentation as K
from kornia.augmentation import AugmentationBase2D
from model.noise_layers.utils import jpeg_compress

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffJPEG(nn.Module):
    def __init__(self,quality) -> None:
        super(DiffJPEG,self).__init__()
        self.quality = quality

    def forward(self,x:torch.Tensor):
        with torch.no_grad():
            img_clip = x.clamp(0,1)
            img_jpeg = jpeg_compress(img=img_clip,quality=self.quality,device=device)
            img_gap = img_jpeg - x
            img_gap = img_gap.detach()
        img_aug = x+img_gap
        return img_aug
    
class RandomDiffJPEG(AugmentationBase2D):
    def __init__(self,p: float = 0.5,low=10,high=100) -> None:
        super().__init__(p=p)
        self.diff_jpegs = [DiffJPEG(quality=qf).to(device=device) for qf in range(low,high,10)]

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

class RandomBlur(AugmentationBase2D):
    def __init__(self, blur_size, p=1) -> None:
        super().__init__(p=p)
        self.gaussian_blurs = [K.RandomGaussianBlur(kernel_size=(kk,kk), sigma= (kk*0.15 + 0.35, kk*0.15 + 0.35)) for kk in range(1,int(blur_size),2)]

    def generate_parameters(self, input_shape: torch.Size):
        blur_strength = torch.randint(high=len(self.gaussian_blurs), size=input_shape[0:1])
        return dict(blur_strength=blur_strength)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        blur_strength = params['blur_strength']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.gaussian_blurs[blur_strength[ii]](input[ii:ii+1])
        return output
class Noiser(nn.Module):
    def __init__(self, degrees=30, crop_scale=(0.2, 1.0), crop_ratio=(3/4, 4/3), blur_size=17, color_jitter=(1.0, 1.0, 1.0, 0.3), diff_jpeg=10,
                p_crop=0.5, p_aff=0.5, p_blur=0.5, p_color_jitter=0.5, p_diff_jpeg=0.5, 
                cropping_mode='slice', img_size=224
            ):
        super(Noiser, self).__init__()
        self.jitter = K.ColorJitter(*color_jitter, p=p_color_jitter).to(device)
        # self.jitter = K.RandomPlanckianJitter(p=p_color_jitter).to(device)
        self.aff = K.RandomAffine(degrees=degrees, p=p_aff).to(device)
        self.crop = K.RandomResizedCrop(size=(img_size,img_size),scale=crop_scale,ratio=crop_ratio, p=p_crop, cropping_mode=cropping_mode).to(device)
        self.hflip = K.RandomHorizontalFlip().to(device)
        self.blur = RandomBlur(blur_size, p_blur).to(device)
        self.diff_jpeg = RandomDiffJPEG(p=p_diff_jpeg, low=diff_jpeg).to(device)
    
    def forward(self, input):
        input = self.diff_jpeg(input)
        input = self.aff(input)
        input = self.crop(input)
        input = self.blur(input)
        input = self.jitter(input)
        input = self.hflip(input)
        return input

class IdentityNoiser(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,x):
        return x