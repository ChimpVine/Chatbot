from flask import Flask, render_template, request, redirect, jsonify
import fitz  # PyMuPDF
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.5,
    max_tokens=4095
)

prompt_template = """Role: Act as a lesson planner. You are responsible for creating detailed and effective {question} to facilitate student learning in your academic setting.
Task: Generate a comprehensive lesson plan template for Context:{context} tailored to specific grade levels, subjects, and durations, including sections for learning objectives, instructional strategies, assessment methods, and reflection. Generate the output as shown in the example below:

Example:
<body>
    <h1>Lesson Plan: Introduction to Addition and Subtraction</h1>
    <p><strong>Grade Level:</strong> 2nd Grade</p>
    <p><strong>Duration:</strong> 1 hour</p>
    <p><strong>Subject:</strong> Mathematics</p>

    <h2>Learning Objectives:</h2>
    <ul>
        <li>Students will be able to understand and apply the concepts of addition and subtraction.</li>
        <li>Students will be able to solve basic addition and subtraction problems up to 20.</li>
        <li>Students will develop problem-solving skills.</li>
    </ul>

    <h2>Prior Knowledge:</h2>
    <ul>
        <li>Basic counting principles and number recognition.</li>
        <li>Understanding of arithmetic symbols (+ and -) and their meanings.</li>
        <li>Familiarity with basic arithmetic operations and number relationships.</li>
    </ul>

    <h2>Materials Needed:</h2>
    <ul>
        <li>Whiteboard and markers</li>
        <li>Worksheets for addition and subtraction problems</li>
        <li>Counting manipulatives (e.g., blocks, beads)</li>
        <li>Number charts</li>
        <li>Pencils and erasers</li>
    </ul>

    <h2>Standards Alignment:</h2>
    <p>(Align this to your local/state educational standards)</p>

    <h2>Introduction (10 minutes):</h2>
    <p>Begin with a short story or a set of real-life scenarios where addition and subtraction are used (e.g., buying candies, sharing toys). Discuss the importance of these operations in daily life.</p>

    <h2>Instruction (15 minutes):</h2>
    <p>Introduce the concept of addition as putting together and increasing, using visual aids like blocks or beads to demonstrate. Then, introduce the concept of subtraction as taking away and decreasing, also utilizing visual aids. Explain terms like sum, total, difference, minus, and plus.</p>

    <h2>Activities (20 minutes):</h2>
    <h3>Guided Practice:</h3>
    <p>Utilize a number chart to help students visualize adding and subtracting numbers. Solve some problems together on the whiteboard, involving students in the process.</p>

    <h3>Independent Practice:</h3>
    <p>Distribute worksheets that include a variety of problems. Have students attempt these individually. Circulate the room to assist any students who are struggling.</p>

    <h2>Assessment (10 minutes):</h2>
    <p>Review the worksheet answers as a class, correcting any misunderstandings. Ask students to verbally explain how they solved one of the problems to assess understanding.</p>

    <h2>Conclusion (5 minutes):</h2>
    <p>Recap what was learned in today's lesson. Provide a couple of extra problems for early finishers or for homework to reinforce today's lesson. Mention that the next lesson will build on these skills, introducing more complex problems.</p>

    <h2>Reflection (After class):</h2>
    <p>Reflect on which strategies worked well and which didn’t. Consider individual student needs for the next lesson based on today's observations.</p>
</body>
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

Please provide the output as HTML code with only the content within the `<body>` tags, just the body part so that I can use it directly in the code. It should always be provided in HTML code format and please do not use `/n` in the code.
"""

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def generate_lesson_plan(context, command):
    prompt = prompt_template.replace("{context}", context).replace("{question}", command)
    response = llm.predict(prompt)
    return response


# @app.route('/api/generate_lesson_plan', methods=['POST'])
# def api_generate_lesson_plan():
#     data = request.json
#     pdf_path = data.get('pdf_path')
#     lesson = data.get('command')
#     grade = data.get('grade')
#     duration = data.get('duration')
#     subject = data.get('subject')

#     if not all([pdf_path, lesson, grade, duration, subject]):
#         return jsonify({"error": "Missing required fields"}), 400

#     pdf_text = extract_text_from_pdf(pdf_path)
#     command = f"Lesson: {lesson}\nGrade: {grade}\nDuration: {duration}\nSubject: {subject}"
#     lesson_plan = generate_lesson_plan(pdf_text, command)
#     lesson_plan = lesson_plan.replace("```", "")
#     lesson_plan = lesson_plan.replace("<html>", "")
#     lesson_plan = lesson_plan.replace("</html>", "")
#     lesson_plan = lesson_plan.replace("<body>", "")
#     lesson_plan = lesson_plan.replace("</body>", "")
#     lesson_plan = lesson_plan.replace("html", "")
#     lesson_plan = lesson_plan.replace("<!DOCTYPE html>", "")
#     lesson_plan = lesson_plan.replace("< lang=>", "")

#     return jsonify({"lesson_plan": lesson_plan})


