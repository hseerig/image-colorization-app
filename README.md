# 📸 Image Colorization using GANs & U-Net

A deep learning project that colorizes grayscale images using a Conditional GAN (cGAN) with a U-Net generator architecture. This system takes a grayscale image as input and produces a realistic color version.

---

### 📌 Demo
  
> 📦 **Trained Model**: [`res18-unet.pt`](./res18-unet.pt)  
> 🧠 **Training Notebook**: [`colorization_training.ipynb`](./colorization-training.ipynb)

---

### 🧠 Model Overview

- **Architecture**: ResNet18-based U-Net Generator + PatchGAN Discriminator
- **Training Losses**:
  - L1 Loss (for pixel-wise color similarity)
  - GAN Loss (for realism via adversarial training)
- **Frameworks**: PyTorch, FastAI, Streamlit

---

### 🧪 Sample Results


| ![Gray](assets/input1.png) | 
| ![Gray](assets/input2.png) | 



