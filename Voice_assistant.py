import openai
import speech_recognition as sr
import pyttsx3
import os
import json

personality = "p.txt"
usewhisper = True
key = os.getenv('OPENAI_API_KEY',
                'sk-proj-GUwTcSjSIxnMHoDXihsbT3BlbkFJQ5jratjTeaIIOQ5nBrQ3')

# openAI set-up
openai.api_key = key
with open(personality, "r") as file:
    mode = file.read()
messages = [{"role": "system", "content": f"{mode}"}]

# pyttsx3 setup
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 0 for male, 1 for female

# speech recognition set-up
r = sr.Recognizer()
mic = sr.Microphone(device_index=0)
r.dynamic_energy_threshold = False
r.energy_threshold = 400


def whisper(audio):
    with open('speech.wav', 'wb') as f:
        f.write(audio.get_wav_data())
    try:
        with open('speech.wav', 'rb') as speech:
            wcompletion = openai.Audio.transcribe(
                model="whisper-1",
                file=speech
            )
        user_input = wcompletion['text']
        print(user_input)
        return user_input
    except Exception as e:
        print(f"Error with Whisper: {e}")
        return ""


def save_conversation(save_foldername):
    os.makedirs(save_foldername, exist_ok=True)
    base_filename = 'conversation'
    suffix = 0
    filename = os.path.join(save_foldername, f'{base_filename}_{suffix}.txt')

    while os.path.exists(filename):
        suffix += 1
        filename = os.path.join(
            save_foldername, f'{base_filename}_{suffix}.txt')

    with open(filename, 'w') as file:
        json.dump(messages, file, indent=4)

    return suffix


def save_inprogress(suffix, save_foldername):
    os.makedirs(save_foldername, exist_ok=True)
    base_filename = 'conversation'
    filename = os.path.join(save_foldername, f'{base_filename}_{suffix}.txt')

    with open(filename, 'w') as file:
        json.dump(messages, file, indent=4)


# grab script location
script_dir = os.path.dirname(os.path.abspath(__file__))
foldername = "voice_assistant"
save_foldername = os.path.join(script_dir, f"conversations/{foldername}")
suffix = save_conversation(save_foldername)

# main while loop where the conversation occurs
while True:
    with mic as source:
        print("\nListening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)
        try:
            if usewhisper:
                user_input = whisper(audio)
            else:
                user_input = r.recognize_google(audio)
        except sr.UnknownValueError:
            print("Could not understand audio")
            continue
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

    if user_input:
        messages.append({"role": "user", "content": user_input})

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=messages,
                temperature=0.8
            )
            response = completion.choices[0].message.content
            messages.append({"role": "assistant", "content": response})
            print(f"\n{response}\n")
            save_inprogress(suffix, save_foldername)

            engine.say(response)
            engine.runAndWait()
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
