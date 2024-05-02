# Deep Learning Model for Image Watermarking

## Overview

This project is focused on developing a deep learning-based solution for watermarking images by embedding hidden messages. The system uses a U-Net-based architecture for encoding and extracting watermarks into images. This document provides an overview of the neural network models involved in the watermarking process, detailing their components and functionalities. The goal of this project is to achieve robust watermarking that maintains the quality of the original image while ensuring that the watermark (a 30-bit message) is recoverable even after potential image processing disturbances.

## Model Components

### 1. **U-Net Architecture**

The U-Net model is employed for embedding a watermark into the images. It's structured to work effectively with image data, performing operations that modify image pixels in a way that embeds the watermark invisibly. Here’s a breakdown of its architecture:

- **Input Convolution Layer**: The first layer of the model which prepares the image for deeper processing by expanding its channel dimensions.
  
- **Downsampling Path**: A series of layers designed to reduce the spatial dimensions of the input image, while increasing the depth (channels). This path helps in capturing the context of the image which is essential for effective watermark embedding.
  
- **Bottleneck**: This component receives the output from the downsampling path and the watermark message. It combines these inputs to embed the watermark within the feature maps generated from the image.
  
- **Upsampling Path**: Symmetric to the downsampling path, the upsampling path increases the spatial dimensions and decreases the depth, aiming to reconstruct the watermarked image from the deep feature representations.
  
- **Output Convolution Layer**: This layer converts the output of the upsampling path back to the original image dimensions, producing the final watermarked image.

### 2. **Extractor**

The Extractor is a neural network model designed to retrieve the watermark from a watermarked image. It mirrors the architecture of the U-Net’s encoder part:
  
- **Input Layer**: Accepts the possibly distorted watermarked image.
  
- **Downsampling**: Similar to the U-Net, it processes the image to capture necessary features for effective extraction.
  
- **Output Layer**: Processes the features to predict the watermark message.

### 3. **Discriminator**

A discriminator model is used for training the network using adversarial tactics. It helps in making the watermarked images indistinguishable from the original images to human eyes, enhancing the invisibility of the watermark.

### 4. **Loss Functions**

Multiple loss functions are employed to optimize different aspects of the training:
  
- **Image Reconstruction Loss**: Ensures the watermarked image is as close to the original as possible.
  
- **Watermark Reconstruction Loss**: Ensures the watermark is correctly embedded and can be extracted accurately.
  
- **Adversarial Loss (Discriminator)**: Encourages the generator to create realistic images that can fool the discriminator.
  
- **Perceptual Loss (LPIPS)**: Ensures perceptual similarity between the original and watermarked images.

## Experimental Setup

The experiments are conducted to validate the effectiveness of the proposed model under various conditions and disturbances. Results and insights from these experiments will be documented and stored in the `experiments` folder. This directory will be updated regularly with the latest findings from ongoing tests.

## Future Enhancements

This project is currently under development, and future updates will include detailed training procedures, testing scripts, and further enhancements to the model architecture based on experimental outcomes.


## References

- Zhu, J., Kaplan, R., Johnson, J., & Fei-Fei, L. (2018). HiDDeN: Hiding Data With Deep Networks. [GitHub Repository](https://github.com/ando-khachatryan/HiDDeN)
- Fernandez, P., Couairon, G., Jégou, H., Douze, M., & Furon, T. (2022). The Stable Signature: Rooting Watermarks in Latent Diffusion Models. [GitHub Repository](https://github.com/facebookresearch/stable_signature)

### Note:

The project setup and structure are in development stages, and documentation regarding setup, usage, and contributions will be provided in subsequent updates. This documentation primarily focuses on the network architecture and its components as the core of ongoing research in image watermarking using deep learning techniques.