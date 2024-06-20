import openai
import os
from gtts import gTTS
from moviepy.editor import *

# Step 1: Generate Script
api_key = os.getenv("OPENAI_API_KEY", "sk-proj-GUwTcSjSIxnMHoDXihsbT3BlbkFJQ5jratjTeaIIOQ5nBrQ3")
openai.api_key = api_key

def generate_script(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message['content'].strip()

prompt = "Create a short script for a video about the importance of education."
script = generate_script(prompt)
print("Generated Script:", script)

# Step 2: Convert Text to Audio
def text_to_audio(script, filename="output.mp3"):
    tts = gTTS(script, lang='en')
    tts.save(filename)

audio_file = "output.mp3"
text_to_audio(script, audio_file)

# Step 3: Create Video with Audio
def create_video_with_audio(audio_file, output_file="output_video.mp4"):
    # Generate a blank video clip
    audio_clip = AudioFileClip(audio_file)
    video_clip = ColorClip(size=(640, 480), color=(255, 255, 255), duration=audio_clip.duration)
    
    # Add the audio file to the video
    video_clip = video_clip.set_audio(audio_clip)
    
    # Write the final video file
    video_clip.write_videofile(output_file, fps=24)

output_video_file = "output_video.mp4"
create_video_with_audio(audio_file, output_video_file)

print("Video created successfully!")
