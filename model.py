from langchain import OpenAI
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent

# Initialize OpenAI LLM
llm = OpenAI(model='gpt-3.5-turbo-instruct',
             api_key="sk-proj-1vPIlHmPoYx9c8W2Co0kT3BlbkFJqzfUBPrTA6eCLpiwthJk", temperature=0)

# Define the word problem template
word_problem_template = """You are a reasoning agent tasked with solving the user's logic-based questions.
Logically arrive at the solution, and be factual. In your answers, clearly detail the steps involved and give
the final answer. Provide the response in bullet points. Question: {question} Answer:"""

math_assistant_prompt = PromptTemplate(
    input_variables=["question"],
    template=word_problem_template
)

# Create chains and tools
word_problem_chain = LLMChain(llm=llm, prompt=math_assistant_prompt)
word_problem_tool = Tool.from_function(
    name="Reasoning Tool",
    func=word_problem_chain.run,
    description="Useful for when you need to answer logic-based/reasoning questions.",
)

problem_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool.from_function(
    name="Calculator",
    func=problem_chain.run,
    description="Useful for when you need to answer numeric questions. This tool is only for math questions and nothing else. Only input math expressions, without text",
)

wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.",
)

agent = initialize_agent(
    tools=[wikipedia_tool, math_tool, word_problem_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)


def process_query(query):
    response = agent(query)
    return response["output"]
