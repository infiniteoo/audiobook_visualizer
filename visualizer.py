import time
from openai import OpenAI
import speech_recognition as sr
from PIL import Image
import matplotlib.pyplot as plt
from pydub import AudioSegment
import requests
from io import BytesIO
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)  # Replace with your OpenAI API key


def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    text = recognizer.recognize_google(audio)
    return text

def generate_image(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url

def get_current_audio_chunk(audio_file, start_time=0, end_time=10):
    audio = AudioSegment.from_file(audio_file)
    chunk = audio[start_time * 1000:end_time * 1000]
    chunk.export("current_chunk.wav", format="wav")
    return "current_chunk.wav"

def display_or_save_image(image_url, save_path=None):
    # Download the image from the URL
    image_data = requests.get(image_url).content

    # Save the image to the specified path
    with open(save_path, 'wb') as image_file:
        image_file.write(image_data)

    # Display the image
    img = Image.open(BytesIO(image_data))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def save_generated_image(generated_text, save_path):
    response = client.images.generate(
        model="dall-e-3",
        prompt=generated_text,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url

    # Download the image from the URL
    image_data = requests.get(image_url).content

    # Save the image to the specified path
    with open(save_path, 'wb') as image_file:
        image_file.write(image_data)


# Example usage
audio_file = "book.mp3"
start_time = 0  # Replace with the desired start time in seconds
end_time = 10   # Replace with the desired end time in seconds

current_audio_chunk = get_current_audio_chunk(audio_file, start_time, end_time)
transcribed_text = transcribe_audio(current_audio_chunk)

# Generate image URL based on transcribed text
generated_image_url = generate_image(transcribed_text)

# Display or save the corresponding image using the URL
save_path = "./images/generated_image.png"
display_or_save_image(generated_image_url, save_path)
