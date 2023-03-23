print('Initializing... Dependencies')
from conversation import character_msg_constructor
from anime_tts.custom_inference import tts # text to speech from huggingface
from vtube_studio import Char_control
import romajitable # temporary use this since It'll blow up our ram if we use Machine Translation Model
import pyaudio
import soundfile as sf
import scipy.io.wavfile as wavfile
import requests
import random
import os
talk = character_msg_constructor("Lilia", None) # initialize character_msg_constructor

print('Initializing... Vtube Studio')
# initialize Vstudio Waifu Controller
waifu = Char_control(port=8001, plugin_name='MyBitchIsAI', plugin_developer='HRNPH')
print('Initialized')
# vtube.express(express) # use this to express feeling

def chat(msg, reset=False):
    command = 'chat'
    if reset:
        command = 'reset'
    params = {
        'command': f'{command}',
        'data': msg,
    }
    r = requests.get('http://localhost:8267/waifuapi', params=params)
    return r.text



# run ping to keep connection alive in the background
import threading
import time
# def ping():
#     while True:
#         waifu.express('netural')
#         time.sleep(3)


# threading.Thread(target=ping).start()

split_counter = 0
history = ''
while True:
    con = str(input("You: "))
    if con.lower() == 'exit':
        print('Stopping...')
        break # exit prototype

    if con.lower() == 'reset':
        print('Resetting...')
        print(chat('None', reset=True))
        continue # reset story skip to next loop

    # ----------- Create Response --------------------------
    answer = chat(con) # send message to api
    answer = answer.split('<split_token>')
    answer, japanese_answer = answer[0], answer[1]
    answer.replace('Lilia:', '') # remove name from answer
    # ------------------------------------------------------
    print(f'{answer}')
    if answer.strip().endswith(f'{talk.name}:') or answer.strip() == '':
        continue # skip audio processing if the answer is just the name (no talking)

    # ----------- Waifu Create Talking Audio -----------------------
    ## generate speaking voice (Translates to Japanese and then TTS)/(Japanglish TTS)[take lower memory but not as enjoyable]
    ## we'll use Japanglish TTS for prototyping, we'll use Japanese TTS for production
    # cleaned_text = talk.clean_emotion_action_text_for_speech(answer) # delete *describe* in text and left with only "speaking" part
    # translated = romajitable.to_kana(cleaned_text).hiragana # translate to Japanese

    # using Japanglish TTS we don't need to clean the text since server already did it before translating
    translated = japanese_answer
    _, (sr, audio) = tts(translated, 0)
    random_name = '_cache' #random.randint(0, 1000)
    wavfile.write(f'./audio_cache/dialog{random_name}.wav', sr, audio)

    # --------------------------------------------------
    
    # ----------- Waifu Talking -----------------------
    # play audio directly from cache
    p = pyaudio.PyAudio()
    data, samplerate = sf.read('./audio_cache/dialog_cache.wav', dtype='float32')
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=samplerate,
                    output=True)
    stream.write(data.tobytes())
    stream.stop_stream()
    stream.close()

    # --------------------------------------------------

    # ----------- Waifu Expressing ----------------------- (emotion expressed)
    emotion = talk.emotion_analyze(answer) # get emotion from waifu answer (last line)
    print(f'Emotion Log: {emotion}')
    emotion_to_express = None
    if 'joy' in emotion:
        emotion_to_express = 'happy'

    elif 'anger' in emotion:
        emotion_to_express = 'angry'

    print(f'Emotion to express: {emotion_to_express}')
    if emotion_to_express: ## express emotion
        waifu.express(emotion_to_express) # express emotion in Vtube Studio
    # --------------------------------------------------
