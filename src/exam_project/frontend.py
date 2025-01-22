import streamlit as st
import requests
from PIL import Image
import pandas as pd
import io
from base64 import b64decode

# Define the FastAPI backend URL
API_URL = "https://backend-final-424957459314.europe-west1.run.app"

# Set Streamlit page configuration
st.set_page_config(page_title="Image Attribution Visualization", layout="centered")

# App title and description
st.title("Classification of Melanoma Images")
st.markdown(
    """
    Welcome to the **Melanoma Classification App**!
    Upload an image or select a default image to view **attribution visualizations**.
    """
)

# Sidebar for instructions
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    1. Upload an image or select a default image.
    2. Wait for the processing to complete.
    3. View the prediction, probabilities, and attribution results.
    """
)

# Fetch default images from the backend
default_images = requests.get(f"{API_URL}/default_images/").json().get("images", [])
default_image = st.selectbox("Choose a default image:", ["None"] + default_images)

# Upload image or use selected default image
uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
image_to_use = None

if uploaded_file:
    image_to_use = uploaded_file.getvalue()
elif default_image != "None":
    default_image_response = requests.get(f"{API_URL}/default_images/{default_image}")
    if default_image_response.status_code == 200:
        image_to_use = default_image_response.content

# Display and process the selected or uploaded image
if image_to_use:
    st.image(io.BytesIO(image_to_use), caption="Selected Image", use_container_width=True)

    # Submit the image to the backend
    with st.spinner("Analyzing the image..."):
        response = requests.post(f"{API_URL}/predict/", files={"file": image_to_use})

    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        probabilities = result["probabilities"]
        attribution_visualization = result["attribution_visualization"]
        result = ["Benign", "Malignant"]

        # Display prediction
        st.subheader("Prediction:")
        st.write(f"Class: {result[prediction]}")

        # Display probabilities
        st.subheader("Class Probabilities:")
        
        prob_df = pd.DataFrame({"Class": [f"{result[i]}" for i in range(len(probabilities))], "Probability": probabilities})
        st.bar_chart(prob_df.set_index("Class"))

        # Display attribution visualization
        st.subheader("Captum Attribution Visualization:")
        st.write("Captum is used to generate attribution visualizations for the model predictions.")
        image_bytes = b64decode(attribution_visualization)
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption="Attribution Visualization", use_container_width=True)

        st.success("Analysis complete!")
    else:
        st.error("Failed to process the image. Please try again.")
else:
    st.info("Please upload an image or select a default image to get started.")
