import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np

# --- Define the Generator Class (must be identical to the training script) ---
# This is necessary to load the pre-trained model weights.
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            self._block(z_dim + embed_size, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
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
        return self.net(x)

# --- Streamlit App ---

# Use a cache to load the model only once
@st.cache_resource
def load_model():
    # Model parameters (must match the training script)
    Z_DIM = 100
    IMG_CHANNELS = 1
    FEATURES_GEN = 64
    NUM_CLASSES = 10
    IMAGE_SIZE = 64
    GEN_EMBEDDING = 100
    
    # Instantiate the generator
    model = Generator(Z_DIM, IMG_CHANNELS, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING)
    
    # Load the trained weights
    # Make sure 'generator.pth' is in the same directory as this script
    model.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
    model.eval() # Set model to evaluation mode
    return model

# Main app UI
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

# Load the generator model
gen_model = load_model()

# User input: select a digit
st.write("Choose a digit to generate (0-9):")
digit_to_generate = st.selectbox(
    label="Choose a digit to generate (0-9):", 
    options=list(range(10)), 
    label_visibility="collapsed"
)

if st.button("Generate Images"):
    st.subheader(f"Generated images of digit {digit_to_generate}")
    
    with st.spinner("Generating..."):
        # Parameters for generation
        num_images = 5
        latent_dim = 100
        device = torch.device('cpu')

        # Prepare inputs for the model
        noise = torch.randn(num_images, latent_dim, 1, 1).to(device)
        labels = torch.full((num_images,), digit_to_generate, dtype=torch.long).to(device)

        # Generate images
        with torch.no_grad():
            generated_images = gen_model(noise, labels)

        # Post-process images for display (from [-1, 1] to [0, 1])
        generated_images = (generated_images + 1) / 2.0

        # Display images in 5 columns
        cols = st.columns(num_images)
        for i, image_tensor in enumerate(generated_images):
            with cols[i]:
                # Convert tensor to numpy array for display
                image_np = image_tensor.squeeze().cpu().numpy()
                st.image(image_np, caption=f"Sample {i+1}", use_column_width=True)

st.sidebar.info(
    "This app uses a Conditional Generative Adversarial Network (cGAN) "
    "trained on the MNIST dataset to generate handwritten digits."
)