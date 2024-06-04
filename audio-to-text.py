
import openai
import os
from openai import OpenAI

# Replace 'your-api-key' with your actual OpenAI API key or use an environment variable
api_key = os.getenv("OPENAI_API_KEY", "'your-api-key")

openai.api_key = api_key

# Open the audio file from the music library
audio_file_path = r"A:\danson\FAQ\music\Blank Space.mp3"
audio_file = open(audio_file_path, "rb")

# Transcribe the audio file using the OpenAI API
transcription = openai.Audio.transcribe(
  model="whisper-1", 
  file=audio_file
)

# Directory to save the transcribed text
output_directory = r"A:\danson\FAQ\transcriptions"

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Path to the output text file
output_file_path = os.path.join(output_directory, "Blank_Space_Transcription.txt")

# Save the transcription to the file
with open(output_file_path, "w") as output_file:
    output_file.write(transcription["text"])

print(f"Transcription saved to {output_file_path}")
