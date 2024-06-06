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
        openai_api_key=OPENAI_API_KEY,
        temperature=0.5,
        max_tokens=4095
    )

    # Load the tools
    tools = load_tools(["llm-math"], llm=llm)

    # Initialize the agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_execution_time=10
    )

    # Create the system message
    sys_message = f"You are an AI math tutor. Provide me answer along with the steps of solving: {user_input}. If relevant, offer additional tips or common mistakes to avoid related to the problem type. Don't give answer except math problem"
    sys_message = sys_message.replace("{", "(").replace("}", ")")
    agent.agent.llm_chain.prompt.template = sys_message
    response = agent({"input": sys_message})
    
    output = response["intermediate_steps"]
    steps = [result_tuple[0].log for result_tuple in output]
    
    return steps

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
    You are an AI math tutor. Your job is to help students solve math_problem in picture by providing step-by-step solutions and necessary explanations. Your responses should be clear, concise, and educational, guiding the student through the problem-solving process. Include only relevant information that will help the student understand the solution and the concepts involved. Use the following guidelines to construct your responses:
    Instructions:
        1)Restate the Problem: Begin by restating the given math problem to ensure clarity.
        2)Identify the Concept: Briefly identify the mathematical concept or principle needed to solve the problem.
        3)Step-by-Step Solution: Break down the solution into clear, manageable steps. Number each step for easy reference.
        4)Explanation: Provide a brief explanation for each step to clarify why it's necessary and how it contributes to solving the problem.
        5)Final Answer: Clearly state the final answer, emphasizing it so the student can easily identify it.
        6)Additional Tips: If relevant, offer additional tips or common mistakes to avoid related to the problem type.
        7)7) Format: Write all mathematical equations and expressions in plain text, avoiding LaTeX or special formatting.
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
    You are an AI math tutor. Your job is to help students solve math_problem in picture by providing step-by-step solutions and necessary explanations. Your responses should be clear, concise, and educational, guiding the student through the problem-solving process. Include only relevant information that will help the student understand the solution and the concepts involved. Use the following guidelines to construct your responses:
    Instructions:
        1)Restate the Problem: Begin by restating the given math problem to ensure clarity.
        2)Identify the Concept: Briefly identify the mathematical concept or principle needed to solve the problem.
        3)Step-by-Step Solution: Break down the solution into clear, manageable steps. Number each step for easy reference.
        4)Explanation: Provide a brief explanation for each step to clarify why it's necessary and how it contributes to solving the problem.
        5)Final Answer: Clearly state the final answer, emphasizing it so the student can easily identify it.
        6)Additional Tips: If relevant, offer additional tips or common mistakes to avoid related to the problem type.
        7)7) Format: Write all mathematical equations and expressions in plain text, avoiding LaTeX or special formatting.
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
                    for step in steps:
                        st.write(step)
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
