import os
import time
from pytube import YouTube
import openai

# Function to download YouTube audio
def download_youtube_audio(url, download_path):
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    out_file = stream.download(output_path=download_path)
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    return new_file

# Function to transcribe audio to text using OpenAI
def transcribe_audio_to_text(audio_file_path, output_directory):
    # Replace 'your-api-key' with your actual OpenAI API key or use an environment variable
    api_key = os.getenv("OPENAI_API_KEY", "sk-proj-1vPIlHmPoYx9c8W2Co0kT3BlbkFJqzfUBPrTA6eCLpiwthJk")
    openai.api_key = api_key

    # Open the audio file
    with open(audio_file_path, "rb") as audio_file:
        # Transcribe the audio file using the OpenAI API
        transcription = openai.Audio.transcribe(
            model="whisper-1", 
            file=audio_file
        )

    # Create the directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Path to the output text file
    base = os.path.basename(audio_file_path)
    filename = os.path.splitext(base)[0] + "_Transcription.txt"
    output_file_path = os.path.join(output_directory, filename)

    # Save the transcription to the file
    with open(output_file_path, "w") as output_file:
        output_file.write(transcription["text"])

    return output_file_path

def main():
    url = input("Enter the YouTube video URL: ")
    
    download_path = 'music'
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    start_time = time.time()

    print("Downloading audio...")
    audio_file = download_youtube_audio(url, download_path)
    print(f"Audio saved as {audio_file}")

    # Directory to save the transcribed text
    output_directory = os.path.join(download_path, "transcriptions")

    print("Transcribing audio...")
    transcription_file = transcribe_audio_to_text(audio_file, output_directory)
    print(f"Transcription saved to {transcription_file}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
