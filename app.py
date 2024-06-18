import os
import openai
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Set the OpenAI API key
# api_key = os.getenv("OPENAI_API_KEY", "sk-proj-GUwTcSjSIxnMHoDXihsbT3BlbkFJQ5jratjTeaIIOQ5nBrQ3")
# openai.api_key = api_key
api_key = os.getenv("OPENAI_API_KEY", " ")
openai.api_key = api_key
# Load the FAQ content from the text file
def load_faq(file_path):
    with open(file_path, 'r') as file:
        faq_content = file.read()
    return faq_content

# Function to interact with the OpenAI API
def ask_openai(question, faq_content):
    prompt = f"Based on the following FAQ content, please answer the question:\n\n{faq_content}\n\nQ: {question}\nA:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=100,
        temperature=0.7,
    )
    return response.choices[0].message['content'].strip()

# Load the FAQ content once when the server starts
faq_content = load_faq('faq.txt')

@app.route('/')
def index():
    return render_template('faq.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json['question']
        answer = ask_openai(question, faq_content)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
