import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. Page Configuration Setup
st.set_page_config(page_title="Nail Pathology AI", layout="centered")
st.title("💅 Automated Nail Pathology Classifier")

# --- NEW: USER INSTRUCTIONS SECTION ---
st.markdown("### 📸 Image Upload Guidelines")
st.info("""
Before taking and uploading your photograph, please ensure:
* 🧼 **Clean Your Nails:** Wash your hands and remove any dirt or debris under the nail plate[cite: 19].
* 🚫 **Remove Nail Polish:** Nail polish, acrylics, or artificial extensions block texture features and cause false results[cite: 13].
* 💡 **Good Lighting:** Ensure your hand is in a brightly lit environment without harsh shadows[cite: 13].
* 🎯 **Focus:** Keep the camera steady and focused directly on the single affected nail plate[cite: 19].
""")

# 2. Re-create the Architecture
@st.cache_resource 
def load_nail_model():
    model = models.mobilenet_v3_large(weights=None)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 6) 
    )
    weights_path = "best_nail_classifier.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        st.error(f"Could not find '{weights_path}' weights in the directory.")
        return None

model = load_nail_model()

# Target Disease Classes
class_names = ['Acral Lentiginous Melanoma', 'Healthy Nail', 'Onychogryphosis', 'Blue Finger', 'Clubbing', 'Pitting']

# Input Image Transforms
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Build Frontend Upload Interface
uploaded_file = st.file_uploader("Choose a nail image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Photograph", use_column_width=True)
    
    st.write("🔄 Running deep neural inference features...")
    
    # Preprocess the uploaded image file
    tensor_img = img_transforms(image).unsqueeze(0) 
    
    # Execute prediction inference
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred_idx = torch.max(probabilities, 0)
        confidence_score = confidence.item()
        predicted_class = class_names[pred_idx.item()]
    
    # --- NEW: THRESHOLD CHECK LOGIC ---
    CONFIDENCE_THRESHOLD = 0.60  # 60% Threshold
    
    if confidence_score < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ **Image Not Recognized**")
        st.write(f"The model is highly uncertain (Confidence: {confidence_score * 100:.2f}%).")
        st.error("This image does not appear to match any supported nail pathology. Please ensure you are uploading a clear close-up picture of a nail following the instructions above.")
    else:
        # Display successful high-confidence prediction
        st.success(f"### Predicted Condition: **{predicted_class}**")
        st.metric(label="Diagnostic Confidence Score", value=f"{confidence_score * 100:.2f}%")