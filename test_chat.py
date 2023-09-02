from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from Conversation.conversation import character_msg_constructor
from Conversation.translation.pipeline import Translate
from AIVoifu.tts import tts  # text to speech from huggingface
from vtube_studio import Char_control
import romajitable  # temporary use this since It'll blow up our ram if we use Machine Translation Model
import scipy.io.wavfile as wavfile
import torch
import wget

# ----- oobabooga Config -----
import requests

# For local streaming, the websockets are hosted without ssl - http://
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'

# For reverse-proxied streaming, the remote will likely host with ssl - https://
# URI = 'https://your-uri-here.trycloudflare.com/api/v1/generate'


def run(prompt):
    request = {
        'prompt': prompt,
        'max_new_tokens': 250,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'guidance_scale': 1,
        'negative_prompt': '',

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        result = response.json()['results'][0]['text']
        #print(prompt + result)
        return prompt + result
# ---------- Config ----------
translation = bool(input("Enable translation? (Y/n): ").lower() in {'y', ''})

print("Use oobabooga api? (Y/n): ")
if input().lower() == 'y':
    oobabooga_api = True
    response = requests.post(f'http://{HOST}/api/v1/model', json={'action': 'info'})
    if response.status_code != 200:
        print("API status_code != 200, oobabooga_api = False") 
        oobabooga_api = False
    else:
        print("Connected to oobabooga API!")
else:
    oobabooga_api = False
    
if oobabooga_api == False:
    device = torch.device('cpu')  # default to cpu
    use_gpu = torch.cuda.is_available()
    print("Detecting GPU...")
    if use_gpu:
        print("GPU detected!")
        device = torch.device('cuda')
        print("Using GPU? (Y/N)")
        if input().lower() == 'y':
            print("Using GPU...")
        else:
            print("Using CPU...")
            use_gpu = False
            device = torch.device('cpu')

# ---------- load Conversation model ----------
        print("Initilizing model....")
        print("Loading language model...")
        tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-1.3b", use_fast=True)
        config = AutoConfig.from_pretrained("PygmalionAI/pygmalion-1.3b", is_decoder=True)
        model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-1.3b", config=config, )

        if use_gpu:  # load model to GPU
            model = model.to(device)
            print("Inference at half precision? (Y/N)")
            if input().lower() == 'y':
                print("Loading model at half precision...")
                model.half()
            else:
                print("Loading model at full precision...")

if translation:
    print("Translation enabled!")
    print("Loading machine translation model...")
    translator = Translate(device, language="jpn_Jpan")  # initialize translator #todo **tt fix translation
else:
    print("Translation disabled!")
    print("Proceeding... wtih pure english conversation")

print('--------Finished!----------')
# --------------------------------------------------

# --------- Define Waifu personality ----------
talk = character_msg_constructor('Lilia', """Species("Elf")
Mind("sexy" + "cute" + "Loving" + "Based as Fuck")
Personality("sexy" + "cute"+ "kind + "Loving" + "Based as Fuck")
Body("160cm tall" + "5 foot 2 inches tall" + "small breasts" + "white" + "slim")
Description("Lilia is 18 years old girl" + "she love pancake")
Loves("Cats" + "Birds" + "Waterfalls")
Sexual Orientation("Straight" + "Hetero" + "Heterosexual")""")
# ---------------------------------------------

from fastapi.responses import JSONResponse

def get_waifuapi(command: str, data: str):
    if command == "chat":
        msg = data
        # ----------- Create Response --------------------------
        msg = talk.construct_msg(msg, talk.history_loop_cache)  # construct message input and cache History model
        ## ----------- Will move this to server later -------- (16GB ram needed at least)
        if oobabooga_api == False:
            inputs = tokenizer(msg, return_tensors='pt')
            if use_gpu:
                inputs = inputs.to(device)
            print("generate output ..\n")
            out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 80, #todo 200 ?
                                pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=50, top_p=0.95)
            conversation = tokenizer.batch_decode(out, skip_special_tokens=True)
            print(conversation)
            # print("conversation .. \n" + conversation)
        else:
            inputs = msg
            conversation = run(inputs)
        ## --------------------------------------------------

        ## get conversation in proper format and create history from [last_idx: last_idx+2] conversation
        talk.split_counter += 0
        print("get_current_converse ..\n")
        current_converse = talk.get_current_converse(conversation[1])
        print("answer ..\n") # only print waifu answer since input already show
        print(current_converse)
        # talk.history_loop_cache = '\n'.join(current_converse)  # update history for next input message

        # -------------- use machine translation model to translate to japanese and submit to client --------------
        print("cleaning ..\n")
        cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse)  # clean text for speech
        print("cleaned_text\n"+ cleaned_text)

        translated = ''  # initialize translated text as empty by default
        if translation:
            translated = translator.translate(cleaned_text)  # translate to [language] if translation is enabled
            print("translated\n" + translated)

        # return JSONResponse(content=f'{current_converse[-1]}<split_token>{translated}')

    if command == "reset":
        talk.conversation_history = ''
        talk.history_loop_cache = ''
        talk.split_counter = 0
        # return JSONResponse(content='Story reseted...')


get_waifuapi("reset", "")
get_waifuapi("chat", "hi, how are you ?")

get_waifuapi("chat", "Can you recommend good place to relax in tokyo ?")