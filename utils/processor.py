import wolframalpha
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
import google.generativeai as genai
from openai import OpenAI
import requests
import json
import os
from langchain import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# OpenAI API key
OPENAI_API_KEY = "sk-proj-GUwTcSjSIxnMHoDXihsbT3BlbkFJQ5jratjTeaIIOQ5nBrQ3"


def process_math_question(user_input):
    question = user_input
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key='sk-proj-GUwTcSjSIxnMHoDXihsbT3BlbkFJQ5jratjTeaIIOQ5nBrQ3',
        temperature=0.5,
        max_tokens=4095
    )

    prompt_template = """
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

        **Additional Tips:**

        4. **Identify the Core Mathematical Concept:**
        - Focus on understanding the mathematical principle or problem presented by the user.
        - Ensure that the response addresses the math concept directly without delving into programming or coding aspects.

        5. **Use Mathematical Terminology:**
        - Employ appropriate mathematical terminology and notation in explanations.
        - Avoid terms and phrases that are typically associated with programming, such as "function," "loop," "variable," "script," or "code."

        6. **Explain Steps Clearly:**
        - Break down the solution into clear, step-by-step instructions or explanations.
        - Use mathematical symbols and equations to illustrate each step.

        7. **Provide Examples:**
        - Give examples that highlight the mathematical process or concept.
        - Ensure the examples are purely mathematical and free from any programming context.

        8. **Include Visual Aids:**
        - Where applicable, suggest the use of graphs, charts, or diagrams to visualize mathematical problems and solutions.
        - Describe how to draw or interpret these visuals using mathematical methods.

        9. **Use Real-World Applications:**
        - Connect the mathematical concept to real-world scenarios that do not involve programming.
        - Explain how the math problem or solution can be applied in everyday life, science, engineering, etc., without referencing computational methods.

        10. **Avoid Ambiguous Language:**
        - Be precise and clear to prevent any interpretation that might lead to a programming solution.
        - Use language that distinctly pertains to mathematics.

        11. **Encourage Further Mathematical Exploration:**
        - Suggest additional related math problems or concepts for the user to explore.
        - Recommend resources such as math textbooks, online math courses, or educational math websites.
    Please always provide the response in HTML format: just the body part so that I can use it directly in the code. It should always be provided in HTML code format and please do not use `/n` in the code.


    """

    prompt = PromptTemplate(
        input_variables=["Question"],
        template=prompt_template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({'Question': question})
    content = response['text']
    content = content.replace("```", "")
    content = content.replace("html", "")
    content = content.replace("{", "")
    content = content.replace("}", "")
    steps = content
    print(steps)
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
        8)Please always provide the response in HTML format: just the body part so that I can use it directly in the code. It should always be provided in HTML code format and please do not use `/n` in the code.
        **Additional Tips:**

        4. **Identify the Core Mathematical Concept:**
        - Focus on understanding the mathematical principle or problem presented by the user.
        - Ensure that the response addresses the math concept directly without delving into programming or coding aspects.

        5. **Use Mathematical Terminology:**
        - Employ appropriate mathematical terminology and notation in explanations.
        - Avoid terms and phrases that are typically associated with programming, such as "function," "loop," "variable," "script," or "code."

        6. **Explain Steps Clearly:**
        - Break down the solution into clear, step-by-step instructions or explanations.
        - Use mathematical symbols and equations to illustrate each step.

        7. **Provide Examples:**
        - Give examples that highlight the mathematical process or concept.
        - Ensure the examples are purely mathematical and free from any programming context.

        8. **Include Visual Aids:**
        - Where applicable, suggest the use of graphs, charts, or diagrams to visualize mathematical problems and solutions.
        - Describe how to draw or interpret these visuals using mathematical methods.

        9. **Use Real-World Applications:**
        - Connect the mathematical concept to real-world scenarios that do not involve programming.
        - Explain how the math problem or solution can be applied in everyday life, science, engineering, etc., without referencing computational methods.

        10. **Avoid Ambiguous Language:**
        - Be precise and clear to prevent any interpretation that might lead to a programming solution.
        - Use language that distinctly pertains to mathematics.

        11. **Encourage Further Mathematical Exploration:**
        - Suggest additional related math problems or concepts for the user to explore.
        - Recommend resources such as math textbooks, online math courses, or educational math website

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

    
**Additional Tips:**

        4. **Identify the Core Mathematical Concept:**
        - Focus on understanding the mathematical principle or problem presented by the user.
        - Ensure that the response addresses the math concept directly without delving into programming or coding aspects.

        5. **Use Mathematical Terminology:**
        - Employ appropriate mathematical terminology and notation in explanations.
        - Avoid terms and phrases that are typically associated with programming, such as "function," "loop," "variable," "script," or "code."

        6. **Explain Steps Clearly:**
        - Break down the solution into clear, step-by-step instructions or explanations.
        - Use mathematical symbols and equations to illustrate each step.

        7. **Provide Examples:**
        - Give examples that highlight the mathematical process or concept.
        - Ensure the examples are purely mathematical and free from any programming context.

        8. **Include Visual Aids:**
        - Where applicable, suggest the use of graphs, charts, or diagrams to visualize mathematical problems and solutions.
        - Describe how to draw or interpret these visuals using mathematical methods.

        9. **Use Real-World Applications:**
        - Connect the mathematical concept to real-world scenarios that do not involve programming.
        - Explain how the math problem or solution can be applied in everyday life, science, engineering, etc., without referencing computational methods.

        10. **Avoid Ambiguous Language:**
        - Be precise and clear to prevent any interpretation that might lead to a programming solution.
        - Use language that distinctly pertains to mathematics.

        11. **Encourage Further Mathematical Exploration:**
        - Suggest additional related math problems or concepts for the user to explore.
        - Recommend resources such as math textbooks, online math courses, or educational math website

Please always provide the response in HTML format: just the body part so that I can use it directly in the code. It should always be provided in HTML code format and please do not use `/n` in the code.

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
    content
    print(content)
    return content


app_id = 'QYQY68-4ERVU37E82'
client = wolframalpha.Client(app_id)
def get_plot_url(equation):
    try:
        res = client.query(equation)
        for pod in res.pods:
            if 'Plot' in pod.title or 'plot' in pod.title:
                return next(subpod.img.src for subpod in pod.subpods)
    except Exception as e:
        return str(e)
    return None


def upscale_image(image):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    upscale_factor = 2
    upscaled_image = cv2.resize(
        cv_image, (cv_image.shape[1] * upscale_factor, cv_image.shape[0] * upscale_factor), interpolation=cv2.INTER_CUBIC)
    upscaled_pil_image = Image.fromarray(
        cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
    return upscaled_pil_image
