print('Initializing... Dependencies')
from Conversation.conversation import character_msg_constructor
from vtube_studio import Char_control
import romajitable # temporary use this since It'll blow up our ram if we use Machine Translation Model
import pyaudio
import soundfile as sf
import scipy.io.wavfile as wavfile
import requests
import random
import os
import logging
logging.getLogger("requests").setLevel(logging.WARNING) # make requests logging only important stuff
logging.getLogger("urllib3").setLevel(logging.WARNING) # make requests logging only important stuff

talk = character_msg_constructor("Lilia", None) # initialize character_msg_constructor

# ----------- Waifu Vocal Pipeline -----------------------
from AIVoifu.client_pipeline import tts_pipeline
vocal_pipeline = tts_pipeline()

# initialize Vstudio Waifu Controller
print('Initializing... Vtube Studio')
waifu = Char_control(port=8001, plugin_name='MyBitchIsAI', plugin_developer='HRNPH')
print('Initialized')


# chat api
def chat(msg, reset=False):
    command = 'chat'
    if reset:
        command = 'reset'
    params = {
        'command': f'{command}',
        'data': msg,
    }
    try:
        r = requests.get('http://localhost:8267/waifuapi', params=params)
    except requests.exceptions.ConnectionError as e:
        print('--------- Exception Occured ---------')
        print('if you have run the server on different device, please specify the ip address of the server with the port')
        print('Example: http://192.168.1.112:8267 or leave it blank to use localhost')
        print('***please specify the ip address of the server with the port*** at:')
        print(f'*Line {e.__traceback__.tb_lineno}: {e}')
        print('-------------------------------------')
        exit()
    return r.text

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
    answer, translated_answer = answer[0], answer[1]

    use_answer = answer # use translated answer if available
    if len(translated_answer) > 2:
        use_answer = translated_answer

    answer.replace('Lilia:', '') # remove name from answer
    # ------------------------------------------------------
    print(f'{answer}')
    if answer.strip().endswith(f'{talk.name}:') or answer.strip() == '':
        continue # skip audio processing if the answer is just the name (no talking)

    # ----------- Waifu Create Talking Audio -----------------------
    vocal_pipeline.tts(use_answer, save_path=f'./audio_cache/dialog_cache.wav')
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
