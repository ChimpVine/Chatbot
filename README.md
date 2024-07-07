# Flask Application for Solving Math Questions and More

This Flask application provides various endpoints to solve text-based math questions, handle image uploads for math problems, and respond to normal questions. It also includes features for audio processing, plotting equations, and maintaining an API call history.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)

## Installation

### Prerequisites

- Python 3.8 or higher
- Flask
- Pillow
- Flask-CORS
- requests

### Setting Up the Environment

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:

   ```bash
   flask run
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000/` to access the application.

## Endpoints

### `GET /`

- Renders the index page.

### `POST /solve_text`

- Processes a text-based math question.
- **Request body:** JSON with `user_input`.
- **Response:** JSON with `steps`.

### `POST /solve_image_url`

- Processes an image URL for a math question.
- **Request body:** JSON with `image_url`.
- **Response:** JSON with `steps`.

### `POST /solve_image_upload`

- Processes an uploaded image for a math question.
- **Request body:** Form data with `image`.
- **Response:** JSON with `steps`.

### `POST /plot`

- Plots an equation and returns the image.
- **Request body:** JSON with `equation`.
- **Response:** PNG image.

### `GET /api_history`

- Retrieves the history of API calls.
- **Response:** JSON with history data.

### `POST /normal_text`

- Processes a normal text question.
- **Request body:** JSON with `user_input`.
- **Response:** JSON with `steps`.

### `POST /upload_audio`

- Processes an uploaded audio file.
- **Request body:** Form data with `audioFile`.
- **Response:** JSON with `steps`.
