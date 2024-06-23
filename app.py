import os
import requests
import openai
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Set the OpenAI API key using environment variables
api_key = os.getenv("sk-proj-GUwTcSjSIxnMHoDXihsbT3BlbkFJQ5jratjTeaIIOQ5nBrQ3")  # Make sure to set this in your environment
openai.api_key ="sk-proj-GUwTcSjSIxnMHoDXihsbT3BlbkFJQ5jratjTeaIIOQ5nBrQ3"

# Function to fetch real-time data from the website
def fetch_realtime_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text  # Return the content of the webpage as text
        else:
            raise Exception(f"Failed to fetch content from {url}. Status code: {response.status_code}")
    except Exception as e:
        raise Exception(f"Failed to fetch content from {url}: {str(e)}")

# Function to extract relevant information from the website content
def extract_information(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Example: Extracting text from specific elements or classes
        offers = soup.find_all(class_='ct-container')  # Adjust based on actual structure
        deadlines = soup.find_all(class_='menu-affilate-menu-container')  # Adjust based on actual structure
        limits = soup.find_all(class_='class="elementor-heading-title elementor-size-default"')
        
        # Construct a dictionary with extracted information
        extracted_info = {
            'offers': [offer.get_text(strip=True) for offer in offers],
            'deadlines': [deadline.get_text(strip=True) for deadline in deadlines],
            'limits': [limit.get_text(strip=True) for limit in limits]
            # Add more categories as needed based on your website content
        }
        return extracted_info
    
    except Exception as e:
        raise Exception(f"Error extracting information: {str(e)}")

def generate_openai_response(question, extracted_info):
    try:
        # Prepare context based on extracted information
        context_parts = []
        for key, value_list in extracted_info.items():
            context_parts.append(f"{key.capitalize()}:\n" + "\n".join(value_list))
        context = "\n\n".join(context_parts)
        
        # Construct the message for the chat model
        messages = [
            {"role": "system", "content": "You are a professional and highly knowledgeable assistant for ChimpVine, providing precise and detailed information about their services and offers. Maintain a friendly yet professional tone."},
            {"role": "user", "content": f"Context:\n{context}\n\nPlease answer the following question based on the above context. You can help yourself to find more about ChimpVine but be precise"},
            {"role": "user", "content": f"Q: {question}"}
        ]
        
        # Use OpenAI API to generate response based on the question and context
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",  # Use the appropriate model name here
            messages=messages,
            max_tokens=4095,
            temperature=0.7,
            top_p=1.0,
            n=1  # Number of responses to return
        )

        
        
        if response.choices and len(response.choices) > 0:
            # Select the first choice (you may modify this logic based on your requirements)
            chosen_response = response.choices[0].message['content'].strip()
            return chosen_response
        else:
            return "OpenAI did not provide a valid response."
    
    except Exception as e:
        raise Exception(f"Error generating OpenAI response: {str(e)}")


# Function to handle user queries
def handle_user_query(question, website_url):
    try:
        # Fetch real-time data from the website
        website_content = fetch_realtime_data(website_url)
        
        # Extract information from the fetched website content
        extracted_info = extract_information(website_content)
        
        # Generate OpenAI response based on the extracted information and user question
        answer = generate_openai_response(question, extracted_info)
        
        return answer
    
    except Exception as e:
        raise Exception(f"Error handling user query: {str(e)}")

# Route for serving index page
@app.route('/')
def index():
    return render_template('index.html')  

# Route for handling user queries
@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Get user question from the request
        question = request.json.get('question', '')
        
        if not question:
            raise ValueError("Question not provided in JSON request.")
        
        # Fetch real-time data from the website (replace with your actual URL)
        website_url = "https://site.chimpvine.com/"
        
        # Handle user query to generate answer
        answer = handle_user_query(question, website_url)
        
        # Return answer as JSON response
        return jsonify({"answer": answer})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)