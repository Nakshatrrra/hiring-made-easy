from flask import Flask, render_template, request
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import subprocess
import os
import speech_recognition as sr

app = Flask(__name__)

model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

# Initialize the speech recognition engine
recognizer = sr.Recognizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():
    if request.method == "POST":
        video_file = request.files['video']  # Assuming you have a file input with the name 'video' in your form
        if video_file:
            video_file.save("uploaded_video.mp4")

            # Convert video to audio using FFmpeg
            command = 'ffmpeg -i uploaded_video.mp4 -ab 160k -ar 44100 -vn audio.wav'
            subprocess.call(command, shell=True)

            audio_file = "audio.wav"
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)

                    if text:
                        # Ensure the "static" directory exists
                        static_directory = "static"
                        if not os.path.exists(static_directory):
                            os.makedirs(static_directory)

                        # Save recognized text to an output.txt file in the "static" directory
                        output_file_path = os.path.join(static_directory, "output.txt")
                        with open(output_file_path, "w") as output_file:
                            output_file.write(text)

                        # Perform text summarization
                        input_text = "summarize: " + text
                        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
                        summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
                        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

                        return render_template("output.html", data={"summary": summary})
                    else:
                        return "No text recognized from the audio."

                except sr.UnknownValueError:
                    return "Speech recognition could not understand audio"
                except sr.RequestError as e:
                    return f"Could not request results from Google Web Speech API; {e}"
        else:
            return "No video file provided."

if __name__ == '__main__':
    app.run()