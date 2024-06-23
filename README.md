# Math Equation graph plot

This project implements an application that plots mathematical equations using Wolfram Alpha and provides advanced image processing features such as upscaling. It uses Streamlit for the web interface and OpenCV for image processing.
### Features
   - **Equation Plotting:** Allows users to input mathematical equations and see their plots.
   - **Image Upscaling:** Upscales the plot images for better quality display.
  
### Technologies Used
- **Frontend:** Streamlit
- **Libraries:** streamlit,wolframalpha,requests,PIL (Python Imaging Library),numpy,opencv-python (OpenCV)

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd math-problem-solver
   ```

2. Install required Python packages:
```bash
pip install streamlit
pip install wolframalpha
pip install requests
pip install pillow
pip install numpy
pip install opencv-python
 ```
### Note: base64 and json are part of the Python Standard Library and do not need to be installed separately.


3.Set up your Wolfram Alpha API key:
   
   - Replace 'YOUR_APP_ID' in the code with your actual Wolfram Alpha API key.
4.Run the Streamlit app:
- streamlit run app.py
-   
5. Open your web browser and navigate to http://localhost:8501 to view the application.
