import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from skimage.color import lab2rgb
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torchvision.models import resnet18
import io

# Build ResNet18-based U-Net
def build_res_unet(n_input=1, n_output=2, size=256):
    body = create_body(resnet18(pretrained=False), n_in=n_input, cut=-2)
    return DynamicUnet(body, n_output, (size, size))

@st.cache_resource
def load_model():
    model = build_res_unet()
    model.load_state_dict(torch.load("res18-unet.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Convert grayscale image to tensor
def preprocess(image):
    image = image.convert("L").resize((256, 256))
    tensor = T.ToTensor()(image).unsqueeze(0)
    return tensor

# Convert LAB tensor to RGB image


def postprocess(L, ab):
    """
    Converts normalized L and ab torch tensors into a colorized RGB image.
    L: shape (1, 1, H, W)
    ab: shape (1, 2, H, W)
    """
    L = L[0, 0].cpu().numpy() * 100      # (H, W), range [0, 100]
    ab = ab[0].cpu().numpy().transpose(1, 2, 0) * 128  # (H, W, 2)

    Lab = np.zeros((256, 256, 3))
    Lab[:, :, 0] = L
    Lab[:, :, 1:] = ab

    rgb = lab2rgb(Lab)  # Output is (H, W, 3), RGB image
    return rgb


st.title("🎨 Image Colorization App")
uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Grayscale Input", use_column_width=True)

    L = preprocess(image)
    with torch.no_grad():
        ab = model(L)

    colorized = postprocess(L, ab)
    st.image(colorized, caption="Colorized Output", use_column_width=True)

    buf = io.BytesIO()
    from PIL import Image

    # Convert NumPy array to PIL Image
    img = Image.fromarray((colorized * 255).astype(np.uint8))  # assuming colorized ∈ [0, 1]
    img.save(buf, format="PNG")


    byte_im = buf.getvalue()
    st.download_button("📥 Download Colorized Image", data=byte_im, file_name="colorized.png", mime="image/png")
