import streamlit as st
import requests
from PIL import Image
import pandas as pd
import io
from base64 import b64decode

# Define the FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict/"

# Set Streamlit page configuration
st.set_page_config(page_title="Image Attribution Visualization", layout="centered")

# App title and description
st.title("Image Attribution Visualization")
st.markdown(
    """
    Welcome to the **Image Attribution Visualization App**! 
    Upload an image to view **attribution visualizations** generated using:
    - **Integrated Gradients**

    These visualizations help interpret the predictions of the deep learning model by highlighting important regions in the image.
    """
)

# Add a sidebar
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    1. Upload an image using the uploader.
    2. Wait for the processing to complete.
    3. View the prediction, probabilities, and attribution results.
    """
)

# Image upload
uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Submit the image to the backend
    with st.spinner("Analyzing the image..."):
        response = requests.post(API_URL, files={"file": uploaded_file.getvalue()})

    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        probabilities = result["probabilities"]
        attribution_visualization = result["attribution_visualization"]

        # Display prediction
        st.subheader("Prediction:")
        st.write(f"Class: {prediction}")

        # Display probabilities
        st.subheader("Class Probabilities:")
        prob_df = pd.DataFrame({"Class": [f"Class {i}" for i in range(len(probabilities))], "Probability": probabilities})
        st.bar_chart(prob_df.set_index("Class"))

        # Display attribution visualization
        st.subheader("Attribution Visualization:")
        image_bytes = b64decode(attribution_visualization)
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Attribution Visualization", use_container_width=True)

        st.success("Analysis complete!")
    else:
        st.error("Failed to process the image. Please try again.")
else:
    st.info("Please upload an image to get started.")
