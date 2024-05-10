import yaml
from typing import List, Tuple

class ModelConfig:
    def __init__(self, H: int, W: int, channels: List[int], depth: int,
                 ext_channels: List[int], ext_depth: int, im_channels: int, 
                 message_length: int, name: str, noiser: str,num_groups:int) -> None:
        self.name = name
        self.H = H
        self.W = W
        self.channels = channels
        self.depth = depth
        self.ext_channels = ext_channels
        self.ext_depth = ext_depth
        self.im_channels = im_channels
        self.message_length = message_length
        self.noiser = noiser
        self.num_groups = num_groups

    def __str__(self) -> str:
        return (f"ModelConfig(name={self.name}, H={self.H}, W={self.W}, "
                f"channels={self.channels}, depth={self.depth}, "
                f"ext_channels={self.ext_channels}, ext_depth={self.ext_depth}, "
                f"im_channels={self.im_channels}, message_length={self.message_length}, "
                f"num_groups={self.num_groups}, "
                f"noiser={self.noiser})")

class TrainConfig:
    def __init__(self, lr: float, batch: int, epochs: int, lpips: float, disc: float) -> None:
        self.lr = lr
        self.batch = batch
        self.epochs = epochs
        self.lpips = lpips
        self.disc = disc

    def __str__(self) -> str:
        return (f"TrainConfig(lr={self.lr}, batch={self.batch}, epochs={self.epochs}, "
                f"lpips={self.lpips}, disc={self.disc})")

def load(path: str) -> Tuple[ModelConfig, TrainConfig]:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    
    model_config = ModelConfig(**config["model"])
    train_config = TrainConfig(**config["train"])
    
    return model_config, train_config

if __name__ == "__main__":
    model_config, train_config = load("/Users/siva/Code/Signature/experiments/no-noise/config.yml")
    print(model_config)
    print(train_config)
