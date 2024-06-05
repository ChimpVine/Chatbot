import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

# Initialize the LLM with OpenAI's GPT-4o model
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key='sk-proj-1vPIlHmPoYx9c8W2Co0kT3BlbkFJqzfUBPrTA6eCLpiwthJk',
    temperature=0.5,
    max_tokens=4095
)

# Define the prompt template
prompt_template = """ 
You are an AI math tutor. Your job is to help students solve math_problem:{Question} by providing step-by-step solutions and necessary explanations. Your responses should be clear, concise, and educational, guiding the student through the problem-solving process. Include only relevant information that will help the student understand the solution and the concepts involved. Use the following guidelines to construct your responses:
 Instructions:
    1)Restate the Problem: Begin by restating the given math problem to ensure clarity.
    2)Identify the Concept: Briefly identify the mathematical concept or principle needed to solve the problem.
    3)Step-by-Step Solution: Break down the solution into clear, manageable steps. Number each step for easy reference.
    4)Explanation: Provide a brief explanation for each step to clarify why it's necessary and how it contributes to solving the problem.
    5)Final Answer: Clearly state the final answer, emphasizing it so the student can easily identify it.
    6)Additional Tips: If relevant, offer additional tips or common mistakes to avoid related to the problem type.
    7)7) Format: Write all mathematical equations and expressions in plain text, avoiding LaTeX or special formatting.
"""

prompt = PromptTemplate(
    input_variables=["Question"],
    template=prompt_template,
)

chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit app
st.title("AI Math Tutor")

# User input for the math question
question = st.text_input("Enter a math question:")

if st.button("Solve"):
    # Invoke the LLM chain with the user's question
    with st.spinner('Generating solution...'):
        response = chain.invoke({'Question': question})
        content = response['text']
    
    # Display the result
    st.write("### Solution:")
    st.write(content)

if st.button("Solve Another Question"):
    st.session_state.query = ""
    st.experimental_rerun()    
