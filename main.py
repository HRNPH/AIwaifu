from Conversation.conversation import character_msg_constructor
# import romajitable # temporary use this since It'll blow up our ram if we use Machine Translation Model
import pyaudio
import soundfile as sf
# import scipy.io.wavfile as wavfile
import requests
# import random
# import os
import logging
import config
logging.getLogger("requests").setLevel(logging.WARNING) # make requests logging only important stuff
logging.getLogger("urllib3").setLevel(logging.WARNING) # make requests logging only important stuff

# ----------- Waifu Vocal Pipeline -----------------------
if config.use_tts:
    print('Initializing... tts')
    from AIVoifu.client_pipeline import tts_pipeline
    vocal_pipeline = tts_pipeline(config.tts_model_name)
else:
    print("disabling tts")

if config.use_vtuber_studio:
    # initialize Vstudio Waifu Controller
    print('Initializing... Vtuber Studio')
    from vtube_studio import Char_control
    waifu = Char_control(port=8001, plugin_name='MyBitchIsAI', plugin_developer='HRNPH')
    print('Initialized')
else:
    print("disabling vtuber studio")


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
        # r = requests.get('http://localhost:8267/waifuapi', params=params)
        r = requests.post('http://localhost:8267/waifuapi', json=params)
    except requests.exceptions.ConnectionError as e:
        print('--------- Exception Occured ---------')
        print('if you have run the server on different device, please specify the ip address of the server with the port')
        print('Example: http://192.168.1.112:8267 or leave it blank to use localhost')
        print('***please specify the ip address of the server with the port*** at:')
        print(f'*Line {e.__traceback__.tb_lineno}: {e}')
        print('-------------------------------------')
        exit()
    return r.text

def main():
    while True:
        try:    
            con = str(input("You: "))
            if con.lower() == 'exit':
                print('Stopping...')
                break # exit prototype

            if con.lower() == 'reset':
                print('Resetting...')
                print(chat('None', reset=True))
                continue # reset story skip to next loop

            # ----------- Create Response --------------------------
            emo_answer = chat(con).replace("\"","") # send message to api
            emo, answer = emo_answer.split("<split_token>")
            print("**"+emo)
            if len(answer) > 2:
                use_answer = answer

                # ------------------------------------------------------
                print(f'Answer: {answer}')
                if answer.strip().endswith(f'{config.character_name}:') or answer.strip() == '':
                    continue # skip audio processing if the answer is just the name (no talking)
                    
                
                if config.use_tts:
                    # ----------- Waifu Create Talking Audio -----------------------
                    vocal_pipeline.tts(use_answer, save_path=f'./audio_cache/dialog_cache.wav', voice_conversion=True)
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
                if config.use_vtuber_studio:
                    # --------------------------------------------------
                    if emo:  ## express emotion
                        waifu.express(emo)  # express emotion in Vtube Studio
                    # --------------------------------------------------
        except BaseException as e:
            print(e.with_traceback(None))

if __name__ == "__main__":
    main()