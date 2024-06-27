import requests
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
import base64
from io import BytesIO
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Importing functions from utils.processor
from utils.processor import process_math_question, image_url, upload_question, get_plot_url, upscale_image

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Allow requests from any origin for the specified routes
CORS(app, resources={r"/solve_text": {"origins": "*"},
                     r"/solve_image_url": {"origins": "*"},
                     r"/solve_image_upload": {"origins": "*"}})

# Initialize empty list to store history
api_history = []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/solve_text', methods=['POST'])
def solve_text():
    data = request.json
    user_input = data.get('user_input')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    try:
        steps = process_math_question(user_input)
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
    try:
        image = Image.open(image_file)

        # Convert RGBA image to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Debugging: log the base64 image size and sample
        print(f"Base64 Image Size: {len(base64_image)}")
        print(f"Base64 Image Sample: {base64_image[:100]}")

        output = upload_question(base64_image)
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


if __name__ == '__main__':
    app.run(debug=True)
