# Math Problem Solver  App

This project implements an  application that solves mathematical problems either from text input, image URLs, or image uploads. It provides step-by-step solutions for various types of mathematical equations and expressions.
### Features
   - **Text Input Solver:** Allows users to input math problems directly as text.
   - **Image URL Solver:** Solves math problems from images hosted online.
   - **Image Upload Solver:** Solves math problems from locally uploaded images.
   - **API History:** Displays a history of API requests made by the user.
### Technologies Used
- **Frontend:** Streamlit
- **Libraries:** Streamlit, long-chain, dotenv, PIL (Python Imaging Library), google-generative, openai, base64, requests, json

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd math-problem-solver
   ```

2. Install dependencies:

  Python Packages:

    -Streamlit
    -langchain
    -dotenv
    -PIL (Python Imaging Library)
    -google-generativeai
    -openai
    -base64
    -requests
    -son

3. Set up environment variables:
   
   - Create a `.env` file in the root directory.
   - Define necessary environment variables (e.g., API keys, configurations).

4. Run the Flask application:

   ```bash
   streamlit run app.py
   ```

5. Open your web browser and navigate to http://localhost:8501 to view the application.
