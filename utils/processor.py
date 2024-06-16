from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
import google.generativeai as genai
from openai import OpenAI
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


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


def image_url(uri):
    command = """
    You are an AI math tutor. Your job is to help students solve math_problem in picture by providing step-by-step solutions and necessary explanations. Your responses should be clear, concise, and educational, guiding the student through the problem-solving process. Include only relevant information that will help the student understand the solution and the concepts involved. Use the following guidelines to construct your responses:
    Instructions:
        1) Restate the Problem: Begin by restating the given math problem to ensure clarity.
        2) Identify the Concept: Briefly identify the mathematical concept or principle needed to solve the problem.
        3) Step-by-Step Solution: Break down the solution into clear, manageable steps. Number each step for easy reference.
        4) Explanation: Provide a brief explanation for each step to clarify why it's necessary and how it contributes to solving the problem.
        5) Final Answer: Clearly state the final answer, emphasizing it so the student can easily identify it.
        6) Additional Tips: If relevant, offer additional tips or common mistakes to avoid related to the problem type.
        7) Format: Write all mathematical equations and expressions in plain text, avoiding LaTeX or special formatting.
    """

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": command},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
        ],
        max_tokens=3000,
    )

    output = response.choices[0].message.content
    return output


def upload_question(base64_image):
    command = """
    You are an AI math tutor. Your job is to help students solve math_problem in picture by providing step-by-step solutions and necessary explanations. Your responses should be clear, concise, and educational, guiding the student through the problem-solving process. Include only relevant information that will help the student understand the solution and the concepts involved. Use the following guidelines to construct your responses:
    Instructions:
        1) Restate the Problem: Begin by restating the given math problem to ensure clarity.
        2) Identify the Concept: Briefly identify the mathematical concept or principle needed to solve the problem.
        3) Step-by-Step Solution: Break down the solution into clear, manageable steps. Number each step for easy reference.
        4) Explanation: Provide a brief explanation for each step to clarify why it's necessary and how it contributes to solving the problem.
        5) Final Answer: Clearly state the final answer, emphasizing it so the student can easily identify it.
        6) Additional Tips: If relevant, offer additional tips or common mistakes to avoid related to the problem type.
        7) Format: Write all mathematical equations and expressions in plain text, avoiding LaTeX or special formatting.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": command},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}},
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = json.loads(response.text)

    # Extract and return the content inside "choices"
    content = response_data['choices'][0]['message']['content']
    return content
