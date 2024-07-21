import os
import warnings
from pathlib import Path as p
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chat_models import ChatOpenAI
import chromadb
import PyPDF2
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
load_dotenv()
# Set the environment variables for Google and OpenAI APIs
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize the embeddings and model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.5,
    max_tokens=4095
)

prompt_template = """Role: Act as a lesson planner. You are responsible for creating detailed and effective {question} to facilitate student learning in your academic setting.
Task: Generate a comprehensive lesson plan template for Context:{context} tailored to specific grade levels, subjects, and durations, including sections for learning objectives, instructional strategies, assessment methods, and reflection. Generate the output as shown in the example below:

Example:
Lesson Plan:Introduction to Addition and Subtraction

Grade Level: 2nd Grade

Duration: 1 hour

Subject: Mathematics

Learning Objectives:

- Students will be able to understand and apply the concepts of addition and subtraction.
- Students will be able to solve basic addition and subtraction problems up to 20.
- Students will develop problem-solving skills.

Prior knowledge:

- Basic counting principles and number recognition.
- Understanding of arithmetic symbols (+ and -) and their meanings.
- Familiarity with basic arithmetic operations and number relationships.

Materials Needed:

- Whiteboard and markers
- Worksheets for addition and subtraction problems
- Counting manipulatives (e.g., blocks, beads)
- Number charts
- Pencils and erasers

Standards Alignment: (Align this to your local/state educational standards)

Introduction (10 minutes):

Begin with a short story or a set of real-life scenarios where addition and subtraction are used (e.g., buying candies, sharing toys). Discuss the importance of these operations in daily life.

Instruction (15 minutes):

Introduce the concept of addition as putting together and increasing, using visual aids like blocks or beads to demonstrate. Then, introduce the concept of subtraction as taking away and decreasing, also utilizing visual aids. Explain terms like sum, total, difference, minus, and plus.

Activities (20 minutes):

Guided Practice:

Utilize a number chart to help students visualize adding and subtracting numbers. Solve some problems together on the whiteboard, involving students in the process.

Independent Practice:

Distribute worksheets that include a variety of problems. Have students attempt these individually. Circulate the room to assist any students who are struggling.

Assessment (10 minutes):

Review the worksheet answers as a class, correcting any misunderstandings. Ask students to verbally explain how they solved one of the problems to assess understanding.

Conclusion (5 minutes):

Recap what was learned in today's lesson. Provide a couple of extra problems for early finishers or for homework to reinforce today's lesson. Mention that the next lesson will build on these skills, introducing more complex problems.

Reflection (After class):

Reflect on which strategies worked well and which didn’t. Consider individual student needs for the next lesson based on today's observations.

Instruction:
1) Input Grade Level, Duration, and Subject: Specify the grade level, duration (in hours or minutes), and subject for which you are creating the lesson plan. For example, "Grade Level: 5th Grade", "Duration: 45 minutes", "Subject: Science".

2) Learning Objectives: List the specific learning objectives for your lesson. These objectives should clearly articulate what students will learn or be able to do by the end of the lesson. For example, "Students will be able to identify the three states of matter" or "Students will be able to write a persuasive essay using appropriate argumentative techniques."

3) Prior Knowledge: Briefly outline any prerequisite knowledge or skills students should have before engaging in this lesson. This could include concepts previously taught, skills students should have developed, or specific terminology they should understand.

4) Materials Needed: List all materials and resources required to deliver the lesson effectively. This may include textbooks, worksheets, multimedia resources, lab equipment, or art supplies. Be as detailed as possible to ensure you have everything prepared before the lesson.

5) Standards Alignment: Align the lesson with relevant educational standards or curriculum guidelines. This ensures that your lesson is meeting specific learning objectives outlined by your institution or educational authority.

6) Introduction: Describe how you will introduce the topic to students and engage their interest. You may use anecdotes, real-life examples, multimedia presentations, or other strategies to capture their attention and provide context for the lesson.

7) Instruction: Detail the main content delivery methods you will use during the lesson. This could include lectures, demonstrations, group activities, or multimedia presentations. Ensure that the instruction aligns with the learning objectives and covers key points effectively.

8) Activities: Divide the lesson into guided practice and independent practice activities. Describe how students will practice and apply the new concepts or skills introduced during instruction. Include details about group work, individual tasks, and any materials or resources students will use.

9) Assessment: Describe how you will assess student understanding throughout the lesson. This may include formative assessments, such as quizzes or discussions, as well as summative assessments, such as tests or projects. Explain how you will provide feedback to students to support their learning.

10) Conclusion: Summarize the key points covered in the lesson and provide closure. Mention any follow-up tasks or homework assignments students should complete to reinforce their learning. Connect the lesson to future lessons or real-world applications to reinforce its importance.

11) Reflection: After teaching the lesson, reflect on its effectiveness and student engagement. Consider what aspects worked well and what could be improved. Use this reflection to plan adjustments for future lessons based on today's outcomes.

Please always provide the response in HTML format: just the body part so that I can use it directly in the code. It should always be provided in HTML code format and please do not use `/n` in the code.




"""

prompt = PromptTemplate(
    input_variables=["context"],
    template=prompt_template,
)

# Load the question answering chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

vector_store = None

class PyPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pages = [page.extract_text() for page in reader.pages]
        return pages

def initialize_vector_store(pdf_path):
    print(f"Initializing vector store with PDF: {pdf_path}")
    pdf_loader = PyPDFLoader(pdf_path)
    data = pdf_loader.load()
    print(f"Loaded {len(data)} pages from the PDF.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    context = "\n\n".join(data)

    texts = text_splitter.split_text(context)
    print(f"Split text into {len(texts)} chunks.")

    # Embed the chunks and save them to the vector store
    global vector_store
    vector_store = Chroma.from_texts(texts, embeddings).as_retriever()
    print("Vector store initialized.")

def handle_question(question):
    print(f"Handling question: {question}")
    # Check if the vector store is initialized
    if vector_store is None:
        return "Vector store is not initialized. Please initialize it first."
    
    docs = vector_store.get_relevant_documents(question)
    print(f"Retrieved {len(docs)} relevant documents.")

    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )
    lesson_plan_text = response['output_text']
    lesson_plan_text = lesson_plan_text.replace("```", "")
    lesson_plan_text = lesson_plan_text.replace("html", "")
    lesson_plan_text = lesson_plan_text.replace("{", "")
    lesson_plan_text = lesson_plan_text.replace("}", "")
    print(f"Generated lesson plan: {lesson_plan_text}")
    return lesson_plan_text

