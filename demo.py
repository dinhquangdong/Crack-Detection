import torch
from torchvision import transforms
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from DeepCrack.codes.model.deepcrack import DeepCrack
from CrackFormerII.CrackFormerII.nets.crackformerII import crackformer

# Set Streamlit's display area to full screen
st.set_page_config(layout="wide")

# Config device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_models():
    deepcrack_model = DeepCrack()
    deepcrack_model.load_state_dict(torch.load("DeepCrack/codes/checkpoints/DeepCrack_CT260_FT1.pth"))
    deepcrack_model.to(device)
    
    crackformer_model = crackformer()
    crackformer_model.load_state_dict(torch.load("CrackFormerII/CrackFormerII/model/cracktree/crack260.pth"))
    crackformer_model.to(device)
    
    return deepcrack_model, crackformer_model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512), antialias=True),
        transforms.Normalize((0, 0, 0), (1.0, 1.0, 1.0))    # Normalize to [0, 1]
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(image)
    img = img.unsqueeze(0)  # reshape img from shape of (C, H, W) to (1, C, H, W)
    return img

##############################################################################
# LOAD MODELS
deepcrack_model, crackformer_model = load_models()

uploaded_file = st.file_uploader("Choose a file",
                                 type=['png', 'jpg', 'jpeg'],
                                 accept_multiple_files=False)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    preprocessed_image = preprocess_image(image)
    preprocessed_image = preprocessed_image.to(device)
    
    with torch.no_grad():
        deepcrack_pred = deepcrack_model(preprocessed_image)[0]
        crackformer_pred = crackformer_model(preprocessed_image)[-1]
        
    deepcrack_predicted_mask = torch.sigmoid(deepcrack_pred.squeeze().cpu()).numpy()
    crackformer_predicted_mask = torch.sigmoid(crackformer_pred.squeeze().cpu()).numpy()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(preprocessed_image.squeeze().cpu().numpy().transpose(1, 2, 0),
                 caption="Original image",
                 use_column_width=True,
                 channels="RGB")
    with col2:
        st.image(deepcrack_predicted_mask, caption="DeepCrack predicts", use_column_width=True)
    with col3:
        st.image(crackformer_predicted_mask, caption="CrackFormer predicts", use_column_width=True)
