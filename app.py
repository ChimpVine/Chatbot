from functools import wraps
import re
import requests
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session, flash
from PIL import Image
import base64
from io import BytesIO
from flask_cors import CORS
import os
import csv
from dotenv import load_dotenv
# Importing functions from utils.processor
from utils.processor import process_math_question, image_url, upload_question, get_plot_url, upscale_image, normal_question, process_audio
from pathlib import Path as p
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser
from utils.Chat_with_lessonpanner import extract_text_from_pdf, generate_lesson_plan
from utils.normal_image import input_image_setup, get_gemini_response, fetch_image_from_url
# Load environment variables from .env file
load_dotenv()
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session management

# Allow requests from any origin for the specified routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize empty list to store history
api_history = []

# Get credentials from environment variables
USER_EMAIL = os.getenv('USER_EMAIL')
USER_PASSWORD = os.getenv('USER_PASSWORD')
# Load FAQ data from CSV file
# faq_data = {}
# with open('ChimpVine_FAQ.csv', mode='r') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         faq_data[row['Question'].strip().lower()] = row['Answer']


# def find_faq_answer(user_input):
#     # Use regex to find a match in FAQ data
#     user_input = user_input.strip().lower()
#     for question, answer in faq_data.items():
#         if re.search(re.escape(user_input), question):
#             return answer
#     return None



df = pd.read_csv('Customer_Support1.csv')


questions = df['Question'].tolist()
answers = df['Answer'].tolist()
links = df['Link'].tolist()

# Initializing TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fitting and transforming the questions
question_vectors = vectorizer.fit_transform(questions)



def get_response(user_input):
    # Vectorizing the user input
    user_input_vector = vectorizer.transform([user_input])
    
    # Computing cosine similarity between user input and stored questions
    similarities = cosine_similarity(user_input_vector, question_vectors)
    
    # print(similarities)
    # Finding the index of the best matching question
    best_match_index = similarities.argmax()
    
    # print(similarities)
    
    if similarities.max() > 0.15:
        answer = answers[best_match_index]
        link = links[best_match_index]
        if link and isinstance(link, str):
            webbrowser.open(link)
        return answer
    else:
        return None


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == USER_EMAIL and password == USER_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))


@app.route('/solve_text', methods=['POST'])
def solve_text():
    data = request.json
    user_input = data.get('user_input')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    try:
        # Check the FAQ data first
        faq_answer = get_response(user_input)
        if faq_answer:
            record_history('Text', user_input, faq_answer)
            return jsonify({'steps': faq_answer}), 200

        # If no FAQ match, process the input using the model
        steps = process_math_question(user_input)
        steps = steps.replace("```", "")
        steps = steps.replace("html", "").replace("{", "").replace("}", "")
        record_history('Text', user_input, steps)
        return jsonify({'steps': steps}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/solve_image_url', methods=['POST'])
def solve_image_url():
    data = request.json
    image_url_input = data.get('image_url')
    if not image_url_input or not image_url_input.startswith(('http://', 'https://')):
        return jsonify({'error': 'No valid image URL provided'}), 400

    try:
        steps = image_url(image_url_input)
        steps = steps.replace("```", "")
        steps = steps.replace("html", "").replace("{", "").replace("}", "")
        steps = steps.replace("```", "")
        steps = steps.replace("<html>", "")
        steps = steps.replace("</html>", "")
        steps = steps.replace("<body>", "")
        steps = steps.replace("</body>", "")
        steps = steps.replace("html", "")
        steps = steps.replace("<!DOCTYPE html>", "")
        steps = steps.replace('< lang="en">', "")
        steps = steps.replace("\n", "")
        steps = steps.replace("<>", "")
        record_history('Image URL', image_url_input, steps)
        return jsonify({'steps': steps}), 200
    except Exception as e:
        print(f"Error processing image URL: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/solve_image_upload', methods=['POST'])
def solve_image_upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    image_file = request.files['image']
    filename = image_file.filename.lower()

    if not (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')):
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        image = Image.open(image_file)

        # Convert RGBA image to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Debugging: log the base64 image size and sample
        print(f"Base64 Image Size: {len(base64_image)}")
        print(f"Base64 Image Sample: {base64_image[:100]}")

        output = upload_question(base64_image)
        output = output.replace("```", "")
        output = output.replace("html", "").replace("{", "").replace("}", "")
        output = output.replace("```", "")
        output = output.replace("<html>", "")
        output = output.replace("</html>", "")
        output = output.replace("<body>", "")
        output = output.replace("</body>", "")
        output = output.replace("html", "")
        output = output.replace("<!DOCTYPE html>", "")
        output = output.replace('< lang="en">', "")
        output = output.replace("\n", "")
        output = output.replace("<>", "")
        record_history('Image Upload', image_file.filename, output)
        return jsonify({'steps': output}), 200
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500
    
def record_history(type, input, response):
    # Create a dictionary to store history details
    history_item = {
        'type': type,
        'input': input,
        'response': response
    }
    # Add history item to the beginning of the list
    api_history.insert(0, history_item)
    # Trim history to last 10 items
    if len(api_history) > 10:
        api_history.pop()


@app.route('/plot', methods=['POST'])
def plot_equation():
    data = request.json
    equation = data.get('equation', '')
    plot_url = get_plot_url(equation)
    if plot_url:
        response = requests.get(plot_url)
        img = Image.open(BytesIO(response.content))
        upscaled_img = upscale_image(img)
        img_io = BytesIO()
        upscaled_img.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    else:
        return jsonify({'error': 'No plot found for the given equation.'}), 404


@app.route('/api_history', methods=['GET'])
def get_api_history():
    return jsonify(api_history)


@app.route('/normal_text', methods=['POST'])
def normal_text():
    data = request.json
    user_input = data.get('user_input')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    try:
        # Check if the question is in the FAQ data
        faq_answer = get_response(user_input)
        if faq_answer:
            return jsonify({'steps': faq_answer})

        # Call the normal_question function to get the response
        response = normal_question(user_input)
        response = response.replace("```", "").replace(
            "html", "").replace("{", "").replace("}", "")
        response = response.replace("```", "")
        response = response.replace("<html>", "")
        response = response.replace("</html>", "")
        response = response.replace("<body>", "")
        response = response.replace("</body>", "")
        response = response.replace("html", "")
        response = response.replace("<!DOCTYPE html>", "")
        response = response.replace('< lang="en">', "")
        response = response.replace("\n", "")
        response = response.replace("<>", "")
        return jsonify({'steps': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audioFile']
    try:
        # Call your function from another file
        response = process_audio(audio_file)
        response = response.replace("```", "")
        response = response.replace("<html>", "")
        response = response.replace("</html>", "")
        response = response.replace("<body>", "")
        response = response.replace("</body>", "")
        response = response.replace("html", "")
        response = response.replace("<!DOCTYPE html>", "")
        response = response.replace('< lang="en">', "")
        response = response.replace("\n", "")
        response = response.replace("<>", "")
        return jsonify({'steps': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
from werkzeug.utils import secure_filename
import tempfile
# Ensure the temporary upload directory exists
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/generate_lesson_plan', methods=['POST'])
def api_generate_lesson_plan():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    lesson = request.form.get('command')
    grade = request.form.get('grade')
    duration = request.form.get('duration')
    subject = request.form.get('subject')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not all([lesson, grade, duration, subject]):
        return jsonify({"error": "Missing required fields"}), 400

    # Save the file temporarily
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(pdf_path)

    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    command = f"Lesson: {lesson}\nGrade: {grade}\nDuration: {duration}\nSubject: {subject}"
    lesson_plan = generate_lesson_plan(pdf_text, command)
    
    # Clean up the temporary file
    os.remove(pdf_path)
    
    lesson_plan = lesson_plan.replace("```", "")
    lesson_plan = lesson_plan.replace("<html>", "")
    lesson_plan = lesson_plan.replace("</html>", "")
    lesson_plan = lesson_plan.replace("<body>", "")
    lesson_plan = lesson_plan.replace("</body>", "")
    lesson_plan = lesson_plan.replace("html", "")
    lesson_plan = lesson_plan.replace("<!DOCTYPE html>", "")
    lesson_plan = lesson_plan.replace('< lang="en">', "")
    lesson_plan = lesson_plan.replace("\n", "")
    
    return jsonify({"lesson_plan": lesson_plan})
    
@app.route('/math-cal')
def math():
    return render_template('math-keypad.html')

@app.route('/normal_image', methods=['GET', 'POST'])
def normal_image():
    response = None
    if request.method == 'POST':
        uploaded_file = request.files.get('image')
        if uploaded_file:
            image_data = input_image_setup(uploaded_file)
            input_prompt = """You are an AI tutor. Your job is to help students with their questions across various subjects. If the input is a question, provide a clear and concise answer. If the input is a casual greeting or non-question statement, respond appropriately.Please provide the output as HTML code with only the content within the `<body>` tags, just the body part so that I can use it directly in the code. It should always be provided in HTML code format and please do not use `/n` in the code."""
            response = get_gemini_response(input_prompt, image_data, "")
            response = response.replace("```", "")
            response = response.replace("<html>", "")
            response = response.replace("</html>", "")
            response = response.replace("<body>", "")
            response = response.replace("</body>", "")
            response = response.replace("html", "")
            response = response.replace("<!DOCTYPE html>", "")
            response = response.replace('< lang="en">', "")
            response = response.replace("\n", "")
    return render_template('index.html', response=response)

@app.route('/process_image_url', methods=['POST'])
def process_image_url():
    try:
        data = request.get_json()
        image_url = data.get('image_url')

        # Fetch the image from the provided URL
        image_data = fetch_image_from_url(image_url)

        # Define your input prompt
        input_prompt = """You are an AI tutor. Your job is to help students with their questions across various subjects. If the input is a question, provide a clear and concise answer. If the input is a casual greeting or non-question statement, respond appropriately. Please provide the output as HTML code with only the content within the `<body>` tags. Use `<strong>` tags for bold text instead of `<b>`. Ensure the response is formatted correctly in HTML code format, and do not include any newline characters (`/n`)."""

        # Generate response using the image and prompt
        response_text = get_gemini_response("", image_data, input_prompt)

        # Clean up the response to ensure it has the correct HTML format
        response_text = response_text.replace("```", "")
        response_text = response_text.replace("<html>", "")
        response_text = response_text.replace("</html>", "")
        response_text = response_text.replace("<body>", "")
        response_text = response_text.replace("</body>", "")
        response_text = response_text.replace("html", "")
        response_text = response_text.replace("<!DOCTYPE html>", "")
        response_text = response_text.replace('< lang="en">', "")
        response_text = response_text.replace("\n", "")

        return jsonify({'response': response_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

