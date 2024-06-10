from pytube import YouTube
import os

def download_youtube_audio(url, download_path):
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    out_file = stream.download(output_path=download_path)
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)
    return new_file

def main():
    url = input("Enter the YouTube video URL: ")
    
    download_path = 'music'
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    print("Downloading audio...")
    audio_file = download_youtube_audio(url, download_path)
    print(f"Audio saved as {audio_file}")

if __name__ == "__main__":
    main()
