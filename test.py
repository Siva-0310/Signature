import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,RandomSampler,Dataset
from model.model import Model
from model.noise_layers.noiser import IdentityNoiser
from torchvision import transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),          
])

class ImageDataSet(Dataset):
    def __init__(self,directory:str,transform:transforms.Compose=None) -> None:
        super().__init__()
        self.directory = directory
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

if __name__ == "__main__":

    enc_config = {
        "im_channels":3,
        "H":16,
        "W":16,
        "channels" : [32,64,128,256],
        "num_groups":32,
        "message_length":30,
    }

    config = {
        "batch_size":12,
        'lr':1e-5,
        "epochs":60,
        "lpips_weight":0.5,
        "disc_weight":0.5,
        "message_recon_weight":1,
    }
    val = 1000
    model = Model(config=enc_config,train_config=config,device=device,noiser=IdentityNoiser())
    model.load_state_dict("/Users/siva/Code/Signature/experiments/no-noise/models/",num=100,device=device)

    dataset  = ImageDataSet("/Users/siva/Code/Signature/train2014/",transform=transform)
    dataloader = DataLoader(dataset=dataset,batch_size=config["batch_size"],sampler=RandomSampler(range(val)),num_workers=4)


    totalpsnr = 0
    accuracy = 0

    i = 0
    for images in dataloader:
        images = images.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (images.shape[0], 30))).to(device)
        losses, (recon_images, recon_message,_) = model.validate_on_batch([images, message])
        totalpsnr += 10*np.log10(1/losses["image_recon_loss"])
        recon_message = model.predict(recon_message)
        message = message.int()
        l = recon_message == message
        accuracy += l.all(dim=1).sum().item()
        i += 1 
        print(f"batch {i} is done accuracy is {accuracy} psnr is {totalpsnr}")

    print("total_psnr is",totalpsnr/len(dataloader))
    print("accuracy is",accuracy/(len(dataloader)*config["batch_size"]))