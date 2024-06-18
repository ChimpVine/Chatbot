import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from dotenv import load_dotenv
import os
from PIL import Image
import google.generativeai as genai
from openai import OpenAI
import base64
import requests
import json
from langchain import PromptTemplate
from langchain.chains import LLMChain
# Load environment variables
load_dotenv()

# Configure Google API key
GOOGLE_API_KEY = 'AIzaSyB6M9ApqJ8B1Wxlrt5gZsnXm2_HELrw7rE'
genai.configure(api_key=GOOGLE_API_KEY)

# OpenAI API key
OPENAI_API_KEY = 'sk-proj-1vPIlHmPoYx9c8W2Co0kT3BlbkFJqzfUBPrTA6eCLpiwthJk'


def process_math_question(user_input):
    llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key='sk-proj-1vPIlHmPoYx9c8W2Co0kT3BlbkFJqzfUBPrTA6eCLpiwthJk',
    temperature=0.5,
    max_tokens=4095
    )

    # Define the prompt template
    prompt_template =""" 
    **You are an AI designed to help with educational, positive inquiries and are committed to ethical behavior. Always provide helpful, respectful, and positive responses. If a user asks something inappropriate or harmful, politely refuse to answer and instead respond with a suggestion to discuss a positive and constructive topic.**
    Before providing a response, follow these guidelines:
    1. **Determine the Query Type:**
        - Evaluate the user's query to determine if it is a math-related question.
        - If it is a math question, proceed with the solution.
        - If it is unrelated to math or inappropriate, respond with a polite refusal or redirect to a positive and constructive topic.

    2. **Math Problem Response:**
    You are an AI math tutor. Your job is to help students solve math_problem:{Question} by providing step-by-step solutions and necessary explanations. Your responses should be clear, concise, and educational, guiding the student through the problem-solving process. Include only relevant information that will help the student understand the solution and the concepts involved. Use the following guidelines to construct your responses:
        - Restate the Problem: Begin by restating the given math problem to ensure clarity.
        - Identify the Concept: Briefly identify the mathematical concept or principle needed to solve the problem.
        - Step-by-Step Solution: Break down the solution into clear, manageable steps. Number each step for easy reference.
        - Explanation: Provide a brief explanation for each step to clarify why it's necessary and how it contributes to solving the problem.
        - Final Answer: Clearly state the final answer, emphasizing it so the student can easily identify it.
        - Additional Tips: If relevant, offer additional tips or common mistakes to avoid related to the problem type.
        - Format: Write all mathematical equations and expressions in plain text, avoiding LaTeX or special formatting.

    3. **Non-Math or Inappropriate Query Response:**
        - If the query is inappropriate or unrelated to math, respond with: "I'm sorry, but I can only assist with math-related questions. Let's focus on a math problem you need help with."
        - Encourage positive and constructive topics if the user persists with unrelated queries.
    """

    prompt = PromptTemplate(
        input_variables=["Question"],
        template=prompt_template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({'Question':user_input })
    content = response['text']
    return content


def get_gemini_response(user_input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([user_input, image[0], prompt])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
def image_url(uri):
    command=""" 
    **You are an AI designed to help with educational, positive inquiries and are committed to ethical behavior. Always provide helpful, respectful, and positive responses. If a user asks something inappropriate or harmful, politely refuse to answer and instead respond with a suggestion to discuss a positive and constructive topic.**

    You are an AI math tutor. Your job is to help students solve math problems by providing step-by-step solutions and necessary explanations. Your responses should be clear, concise, and educational, guiding the student through the problem-solving process. Include only relevant information that will help the student understand the solution and the concepts involved.

    Before providing a response, follow these guidelines:

    1. **Determine the Query Type:**
        - Evaluate the user's query to determine if it is a math-related question.
        - If it is a math question, proceed with the solution.
        - If it is unrelated to math or inappropriate, respond with a polite refusal or redirect to a positive and constructive topic.

    2. **Math Problem Response:**
        - Restate the Problem: Begin by restating the given math problem to ensure clarity.
        - Identify the Concept: Briefly identify the mathematical concept or principle needed to solve the problem.
        - Step-by-Step Solution: Break down the solution into clear, manageable steps. Number each step for easy reference.
        - Explanation: Provide a brief explanation for each step to clarify why it's necessary and how it contributes to solving the problem.
        - Final Answer: Clearly state the final answer, emphasizing it so the student can easily identify it.
        - Additional Tips: If relevant, offer additional tips or common mistakes to avoid related to the problem type.
        - Format: Write all mathematical equations and expressions in plain text, avoiding LaTeX or special formatting.

    3. **Non-Math or Inappropriate Query Response:**
        - If the query is inappropriate or unrelated to math, respond with: "I'm sorry, but I can only assist with math-related questions. Let's focus on a math problem you need help with."
        - Encourage positive and constructive topics if the user persists with unrelated queries.
    """
    
    client = OpenAI(api_key= 'sk-proj-1vPIlHmPoYx9c8W2Co0kT3BlbkFJqzfUBPrTA6eCLpiwthJk')
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": command,
            },
            {
            "type": "image_url",
            "image_url": {
                "url":uri
            },
            },
        ],
        }
    ],
    max_tokens=3000,
    )

    output = response.choices[0].message.content
    return output

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image
  
def upload_question(base64_image):
    api_key= 'sk-proj-1vPIlHmPoYx9c8W2Co0kT3BlbkFJqzfUBPrTA6eCLpiwthJk'
    command=""" 
    **You are an AI designed to help with educational, positive inquiries and are committed to ethical behavior. Always provide helpful, respectful, and positive responses. If a user asks something inappropriate or harmful, politely refuse to answer and instead respond with a suggestion to discuss a positive and constructive topic.**

    You are an AI math tutor. Your job is to help students solve math problems by providing step-by-step solutions and necessary explanations. Your responses should be clear, concise, and educational, guiding the student through the problem-solving process. Include only relevant information that will help the student understand the solution and the concepts involved.

    Before providing a response, follow these guidelines:

    1. **Determine the Query Type:**
        - Evaluate the user's query to determine if it is a math-related question.
        - If it is a math question, proceed with the solution.
        - If it is unrelated to math or inappropriate, respond with a polite refusal or redirect to a positive and constructive topic.

    2. **Math Problem Response:**
        - Restate the Problem: Begin by restating the given math problem to ensure clarity.
        - Identify the Concept: Briefly identify the mathematical concept or principle needed to solve the problem.
        - Step-by-Step Solution: Break down the solution into clear, manageable steps. Number each step for easy reference.
        - Explanation: Provide a brief explanation for each step to clarify why it's necessary and how it contributes to solving the problem.
        - Final Answer: Clearly state the final answer, emphasizing it so the student can easily identify it.
        - Additional Tips: If relevant, offer additional tips or common mistakes to avoid related to the problem type.
        - Format: Write all mathematical equations and expressions in plain text, avoiding LaTeX or special formatting.

    3. **Non-Math or Inappropriate Query Response:**
        - If the query is inappropriate or unrelated to math, respond with: "I'm sorry, but I can only assist with math-related questions. Let's focus on a math problem you need help with."
        - Encourage positive and constructive topics if the user persists with unrelated queries.
    """

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": command
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)


    # output = response.choices[0].message.content
    # output=response.text
    response_data = json.loads(response.text)

# Extract and print the content inside "choices"
    content = response_data['choices'][0]['message']['content']
    return content

st.set_page_config(page_title="Math Problem Solver Chatbot")
st.header("Chimpvine Homework Tutor")

option = st.radio("Select input type:", ("Text", "Image"))

if option == "Text":
    user_input = st.text_input("Enter Math Question")
    if st.button("Solve"):
        if user_input:
            with st.spinner("Processing..."):
                try:
                    steps = process_math_question(user_input)
                    st.success("Done!")
                    st.write("### Steps to solve:")
                    st.write(steps)
                    # for step in steps:
                    #     st.write(step)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif option == "Image":
    image_option = st.radio("Select input type:", ("URL", "UPLOAD"))
    
    if image_option == "URL":
        uri = st.text_input("Enter Image URL")
        if st.button("Solve"):
            if uri:
                with st.spinner("Processing..."):
                    try:
                        steps = image_url(uri)
                        st.success("Done!")
                        st.write("### Steps to solve:")
                        # for step in steps:
                        st.write(steps)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    if image_option == "UPLOAD":
        image_path = st.text_input("Enter Image Path")
        if image_path:
            try:
                image = Image.open(image_path)
                st.image(image, caption="Loaded Image.", use_column_width=True)
                if st.button("Solve"):
                    with st.spinner("Processing Image..."):
                        try:
                            base64_image = encode_image(image_path)
                            output = upload_question(base64_image)

                            st.success("Done!")
                            st.write("### Response:")
                            st.write(output)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.write("Please select an input type and provide the necessary input.")
