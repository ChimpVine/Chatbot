import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key='sk-proj-1vPIlHmPoYx9c8W2Co0kT3BlbkFJqzfUBPrTA6eCLpiwthJk',
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

def process_math_question(user_input):
    # Create the system message
    sys_message = f"You are an AI math tutor. Provide me answer along with the steps of solving: {user_input}.If relevant, offer additional tips or common mistakes to avoid related to the problem type.Don't give answer except math problem"
#   sys_message=f""" 
# You are an AI math tutor. Your job is to help students solve math_problem: {user_input} by providing step-by-step solutions and necessary explanations. Your responses should be clear, concise, and educational, guiding the student through the problem-solving process. Include only relevant information that will help the student understand the solution and the concepts involved. Use the following guidelines to construct your responses:
#  Instructions:
#     1)Restate the Problem: Begin by restating the given math problem to ensure clarity.
#     2)Identify the Concept: Briefly identify the mathematical concept or principle needed to solve the problem.
#     3)Step-by-Step Solution: Break down the solution into clear, manageable steps. Number each step for easy reference.
#     4)Explanation: Provide a brief explanation for each step to clarify why it's necessary and how it contributes to solving the problem.
#     5)Final Answer: Clearly state the final answer, emphasizing it so the student can easily identify it.
#     6)Additional Tips: If relevant, offer additional tips or common mistakes to avoid related to the problem type.
#     7) Format: Write all mathematical equations and expressions in plain text, avoiding LaTeX or special formatting.
# """
    sys_message = sys_message.replace("{", "(").replace("}", ")")
    agent.agent.llm_chain.prompt.template=sys_message
    
    # Process the message through the agent
    response = agent( sys_message)
    # print(response.choices[0])
    # Extract and format the intermediate steps
    
    output = response["intermediate_steps"]
    
    steps = [result_tuple[0].log for result_tuple in output]
    
    return steps

# Streamlit interface
st.title("Math Problem Solver Chatbot")


# User input
user_input = st.text_input("Math Question", "")

# Process the input and display the result
if st.button("Solve"):
    if user_input:
        with st.spinner("Processing..."):
            steps = process_math_question(user_input)
            st.success("Done!")
            st.write("### Steps to solve:")
            for step in steps:
                st.write(step)
    else:
        st.write("Please enter a math question.")
if st.button("Solve Another Question"):
    st.session_state.query = ""
    st.experimental_rerun()

