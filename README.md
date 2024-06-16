# Math Problem Solver Web App

This project implements a web application that solves mathematical problems either from text input, image URLs, or image uploads. It provides step-by-step solutions for various types of mathematical equations and expressions.

## Features

- **Text Input Solver:** Allows users to input math problems directly as text.
- **Image URL Solver:** Solves math problems from images hosted online.
- **Image Upload Solver:** Solves math problems from locally uploaded images.
- **API History:** Displays a history of API requests made by the user.

## Technologies Used

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python (Flask framework)
- **Libraries:** PIL (Python Imaging Library), Flask-CORS
- **Additional Tools:** Postman (for API testing)

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd math-problem-solver
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   
   - Create a `.env` file in the root directory.
   - Define necessary environment variables (e.g., API keys, configurations).

4. Run the Flask application:

   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://localhost:5000` to view the application.

## Usage

- Enter a math problem in the text input field and click "Solve" to see the solution.
- Provide an image URL or upload an image containing a math problem to get the solution.
- The API history section displays a list of all previous requests along with their inputs and responses.

## API Endpoints

- **POST `/solve_text`**: Solve math problems from text input.
- **POST `/solve_image_url`**: Solve math problems from an image URL.
- **POST `/solve_image_upload`**: Solve math problems from an uploaded image file.


