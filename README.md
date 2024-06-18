Chimpvine Homework Tutor

Chimpvine Homework Tutor is a Streamlit app that helps students solve math problems by providing step-by-step solutions. It supports both text and image inputs.
Features

    Text Input: Enter a math question to get a solution.
    Image Input: Upload an image or provide an image URL to solve a math problem.

Installation

    Clone the Repository:

    sh

git clone https://github.com/your-username/chimpvine-homework-tutor.git
cd chimpvine-homework-tutor

Create and Activate a Virtual Environment:

sh

python3 -m venv venv
source venv/bin/activate

Install Dependencies:

sh

pip install -r requirements.txt

Set Up Environment Variables:
Create a .env file and add your API keys:

env

    GOOGLE_API_KEY=your-google-api-key
    OPENAI_API_KEY=your-openai-api-key

Usage

    Run the App:

    sh

    streamlit run app.py

    Open your web browser and go to http://localhost:8501.

How to Use

    Select Input Type: Choose "Text" or "Image".
    Text Input: Enter your math question and click "Solve".
    Image Input:
        URL: Enter the image URL and click "Solve".
        Upload: Upload an image file and click "Solve".
