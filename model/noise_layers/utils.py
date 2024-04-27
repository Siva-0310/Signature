import torch
from augly.image import functional
from torchvision import transforms
from PIL import Image

def jpeg_compress(img,quality,device):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(img,device=device)
    for i,j in enumerate(img):
        pil_img = to_pil(j)
        img_aug[i] = to_tensor(functional.encoding_quality(image=pil_img,quality=quality))
    return img_aug
