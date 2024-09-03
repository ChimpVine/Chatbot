import os
import string
import base64
import tempfile
from functools import wraps
from io import BytesIO
from pathlib import Path as p

import requests
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session, flash
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Importing functions from utils.processor
from utils.processor import process_math_question, image_url, upload_question, get_plot_url, upscale_image, normal_question, process_audio
from utils.normal_image import input_image_setup, get_gemini_response, fetch_image_from_url
from flask import Flask, request, render_template, redirect
import os
import tempfile
#Ask_AI Main file
from utils.Ask_AI import call_gpt
# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session management

# Allow requests from any origin for the specified routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize empty list to store history
api_history = []

# Get credentials from environment variables
USER_EMAIL = os.getenv('USER_EMAIL')
USER_PASSWORD = os.getenv('USER_PASSWORD')

# Download NLTK stopwords
nltk.download('stopwords')

# # Load FAQ data from CSV file
# faq_data = {}
# with open('ChimpVine_FAQ.csv', mode='r') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         # Convert questions to lowercase for case-insensitive matching
#         question = row['Question'].strip().lower()
#         answer = row['Answer']
#         faq_data[question] = answer

# def find_faq_answer(user_input):
#     # Use regex to find a match in FAQ data
#     user_input = user_input.strip().lower()
#     for question, answer in faq_data.items():
#         if re.search(re.escape(user_input), question):
#             return answer
#     return None


# Preprocessing functions
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # Stem words
    ps = PorterStemmer()
    text = ' '.join(ps.stem(word) for word in text.split())
    return text


df = pd.read_csv('Customer_Support__updated.csv')

df['Question'] = df['Question'].apply(preprocess_text).tolist()
answers = df['Answer'].tolist()


# Initializing TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fitting and transforming the questions
question_vectors = vectorizer.fit_transform(df['Question'])



def get_response(user_input, threshold=0.1):
    # Preprocess the user input
    processed_input = preprocess_text(user_input)
    # Vectorize the user input
    user_input_vector = vectorizer.transform([processed_input])
    
    # Compute cosine similarity
    similarities = cosine_similarity(user_input_vector, question_vectors)
    # print(similarities)

    # Find the best match
    max_similarity = similarities.max()
    best_match_index = similarities.argmax()

    # Check if the similarity is above the threshold
    if max_similarity > threshold:
        response = answers[best_match_index]
        return response
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
        steps = steps.replace("```", "").replace("html", "").replace("{", "").replace("}", "").replace("<html>", "").replace("</html>", "").replace("<body>", "").replace("</body>", "").replace("<!DOCTYPE html>", "").replace('< lang="en">', "").replace("\n", "").replace("<>", "")
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
        steps = steps.replace("```", "").replace("html", "").replace("{", "").replace("}", "").replace("<html>", "").replace("</html>", "").replace("<body>", "").replace("</body>", "").replace("<!DOCTYPE html>", "").replace('< lang="en">', "").replace("\n", "").replace("<>", "")
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
        output = output.replace("```", "").replace("html", "").replace("{", "").replace("}", "").replace("<html>", "").replace("</html>", "").replace("<body>", "").replace("</body>", "").replace("<!DOCTYPE html>", "").replace('< lang="en">', "").replace("\n", "").replace("<>", "")
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
        response = response.replace("```", "").replace("html", "").replace("{", "").replace("}", "").replace("<html>", "").replace("</html>", "").replace("<body>", "").replace("</body>", "").replace("<!DOCTYPE html>", "").replace('< lang="en">', "").replace("\n", "").replace("<>", "")

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
        response = response.replace("```", "").replace("html", "").replace("{", "").replace("}", "").replace("<html>", "").replace("</html>", "").replace("<body>", "").replace("</body>", "").replace("<!DOCTYPE html>", "").replace('< lang="en">', "").replace("\n", "").replace("<>", "")
        return jsonify({'steps': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
from werkzeug.utils import secure_filename
import tempfile
# Ensure the temporary upload directory exists
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
            response = response.replace("```", "").replace("html", "").replace("{", "").replace("}", "").replace("<html>", "").replace("</html>", "").replace("<body>", "").replace("</body>", "").replace("<!DOCTYPE html>", "").replace('< lang="en">', "").replace("\n", "").replace("<>", "")

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
        response_text = response_text.replace("```", "").replace("html", "").replace("{", "").replace("}", "").replace("<html>", "").replace("</html>", "").replace("<body>", "").replace("</body>", "").replace("<!DOCTYPE html>", "").replace('< lang="en">', "").replace("\n", "").replace("<>", "")


        return jsonify({'response': response_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
#Ask AI
@app.route('/ask-ai', methods=['POST'])
def ask_ai():
    if request.is_json:
        question_content = request.get_json()

        answer = call_gpt(question_content)

        return jsonify(answer), 200 # 200 to denote sucess

    else:

        return 400 # 400 to denote no success


if __name__ == '__main__':
    app.run(debug=True)

