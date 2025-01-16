import streamlit as st
import requests
from PIL import Image
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
    Welcome to the **NAME**! 
    Upload an image to view **attribution visualizations** generated using the following (captum):
    - **Integrated Gradients**
    - **Saliency**
    - **GradientSHAP**

    These visualizations help interpret the predictions of the deep learning model by highlighting important regions in the image.
    """
)

# Add a sidebar
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    1. Upload an image using the uploader
    2. Wait for the processing to complete.
    3. View the attribution results directly in your browser.
    """
)

# Image upload
uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image in a smaller size
    st.markdown("### Uploaded Image:")
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True, width=300)

    # Submit the image to the backend
    with st.spinner("Analyzing the image and generating attributions..."):
        response = requests.post(
            API_URL,
            files={"file": uploaded_file.getvalue()},
        )

    if response.status_code == 200:
        # Extract the Base64 image from the response
        content = response.content.decode("utf-8")
        start_index = content.find("base64,") + len("base64,")
        base64_image = content[start_index:].strip()

        # Decode and display the result
        image_bytes = b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))

        st.markdown("### Attribution Results:")
        st.image(image, caption="Attribution Results", use_container_width=True)

        st.success("Analysis complete!")
    else:
        st.error("Failed to process the image. Please try again.")
else:
    st.info("Please upload an image to get started.")
