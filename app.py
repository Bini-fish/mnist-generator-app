import streamlit as st
import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, embed_size):
        super(Generator, self).__init__()
        # This line uses `self.gen`, which matches the saved .pth file
        self.gen = nn.Sequential(
            self._block(z_dim + embed_size, features_g * 4, 7, 1, 0),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)

# --- Streamlit App ---

@st.cache_resource
def load_model():
    # --- FINAL OPTIMIZED VERSION (28x28) ---
    # Model parameters must match the training script
    Z_DIM = 100
    IMG_CHANNELS = 1
    FEATURES_GEN = 64
    NUM_CLASSES = 10
    GEN_EMBEDDING = 100
    
    model = Generator(Z_DIM, IMG_CHANNELS, FEATURES_GEN, NUM_CLASSES, GEN_EMBEDDING)
    
    # Ensure 'generator.pth' is the new model you trained and uploaded
    model.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

gen_model = load_model()

st.write("Choose a digit to generate (0-9):")
digit_to_generate = st.selectbox(
    label="Choose a digit to generate (0-9):", 
    options=list(range(10)), 
    label_visibility="collapsed"
)

if st.button("Generate Images"):
    st.subheader(f"Generated images of digit {digit_to_generate}")
    
    with st.spinner("Generating..."):
        num_images = 5
        latent_dim = 100
        device = torch.device('cpu')

        noise = torch.randn(num_images, latent_dim, 1, 1).to(device)
        labels = torch.full((num_images,), digit_to_generate, dtype=torch.long).to(device)

        with torch.no_grad():
            generated_images = gen_model(noise, labels)

        generated_images = (generated_images + 1) / 2.0

        cols = st.columns(num_images)
        for i, image_tensor in enumerate(generated_images):
            with cols[i]:
                image_np = image_tensor.squeeze().cpu().numpy()
                st.image(image_np, caption=f"Sample {i+1}", use_column_width=True)

st.sidebar.info(
    "This app uses a Conditional Generative Adversarial Network (cGAN) "
    "trained on the MNIST dataset to generate handwritten digits."
)