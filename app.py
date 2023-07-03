import whisper
import gradio as gr

from gtts import gTTS
import io
import os
import time
from gtts.lang import _main_langs

model = whisper.load_model("large")

def transcribe(audio):
    
    #time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    return result.text
    

AUDIO_DIR = 'audio_files'
MAX_FILE_AGE = 24 * 60 * 60  # maximum age of audio files in seconds (24 hours)

def text_to_speech(text, lang, tld):
    # map the language name to its corresponding code
    lang_codes = {lang_name: lang_code for lang_code, lang_name in _main_langs().items()}
    lang_code = lang_codes[lang]

    # create the text-to-speech audio
    tts = gTTS(text, lang=lang_code, tld='com')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)

    # create the audio directory if it does not exist
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # generate a unique file name for the audio file
    file_name = str(time.time()) + '.wav'
    file_path = os.path.join(AUDIO_DIR, file_name)

    # save the audio stream to a file
    with open(file_path, 'wb') as f:
        f.write(fp.read())

    # delete old audio files
    delete_old_audio_files()

    # return the file path
    return file_path, f.name

def delete_old_audio_files():
    # delete audio files older than MAX_FILE_AGE
    now = time.time()
    for file_name in os.listdir(AUDIO_DIR):
        file_path = os.path.join(AUDIO_DIR, file_name)
        if now - os.path.getmtime(file_path) > MAX_FILE_AGE:
            os.remove(file_path)

text_to_speech_interface = gr.Interface(fn=text_to_speech, 
                     inputs=[gr.inputs.Textbox(lines=10, label="Enter your text here:"),
                             gr.inputs.Dropdown(choices=list(_main_langs().values()), label="Select language:")],
                     outputs=[gr.Audio(label="Audio"), gr.File(label="Audio File")],
                     allow_flagging="never",live=True)


with gr.Blocks() as demo:
    with gr.Tab("Transcrição"):
        audio = gr.inputs.Audio(source="microphone", type="filepath")
        transcribe_button = gr.Button("transcrever")
        text_output = gr.Textbox()
    
    with gr.Tab("texto para Audio"):
        text_input = gr.Textbox()
        dropdown = gr.Dropdown(choices=list(_main_langs().values()), label="Select language:")
        text2audio_button = gr.Button("Gerar Audio")
        audio_output = gr.Audio(type="filepath")
        

    transcribe_button.click(transcribe,inputs=audio,outputs= text_output)
    text2audio_button.click(text_to_speech, inputs=[text_input,dropdown],outputs=[audio_output,gr.File(label="Audio File")])

demo.launch()
