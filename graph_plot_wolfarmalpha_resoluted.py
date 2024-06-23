import streamlit as st
import wolframalpha
import requests
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import numpy as np
import cv2  # Import OpenCV for advanced image processing

# Replace 'YOUR_APP_ID' with your Wolfram Alpha API key
app_id = 'E97EWA-5A7GP4YR4Q'
client = wolframalpha.Client(app_id)

# Function to query Wolfram Alpha and get the plot URL
def get_plot_url(equation):
    try:
        res = client.query(equation)
        # Find the first plot result
        for pod in res.pods:
            if 'Plot' in pod.title or 'plot' in pod.title:
                return next(subpod.img.src for subpod in pod.subpods)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return None

# Function to upscale image using OpenCV
def upscale_image(image):
    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Use INTER_CUBIC for better quality upscaling
    upscale_factor = 2  # Adjust this factor based on required quality
    upscaled_image = cv2.resize(cv_image, (cv_image.shape[1] * upscale_factor, cv_image.shape[0] * upscale_factor), interpolation=cv2.INTER_CUBIC)
    # Convert back to PIL image
    upscaled_pil_image = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
    return upscaled_pil_image

# Function to display the plot using Streamlit
def plot_equation(equation):
    plot_url = get_plot_url(equation)
    if plot_url:
        response = requests.get(plot_url)
        img = Image.open(BytesIO(response.content))

        # Upscale the image
        upscaled_img = upscale_image(img)

        # Display the upscaled image
        st.image(upscaled_img, caption=f"Plot of {equation}")
    else:
        st.warning("No plot found for the given equation.")

# Streamlit app interface
st.title("Wolfram Alpha Equation Plotter")
st.write("Enter an equation to see its plot:")

# Input for the equation
equation = st.text_input("Equation", "y = x^2")

# Plot the equation when the button is clicked
if st.button("Plot Equation"):
    plot_equation(equation)
