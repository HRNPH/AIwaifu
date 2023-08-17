from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from Conversation.conversation import character_msg_constructor
from Conversation.translation.pipeline import Translate
from AIVoifu.tts import tts # text to speech from huggingface
from vtube_studio import Char_control
import romajitable # temporary use this since It'll blow up our ram if we use Machine Translation Model
import scipy.io.wavfile as wavfile
import torch
import wget 

# ---------- Config ----------
translation = bool(input("Enable translation? (Y/n): ").lower() in {'y', ''})

device = torch.device('cpu') # default to cpu
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

if use_gpu: # load model to GPU
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
    translator = Translate(device, language="jpn_Jpan") # initialize translator #todo **tt fix translation
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

### --- websocket server setup
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import json
import asyncio

# use fast api instead
app = FastAPI()

# do a http server instead
@app.get("/waifuapi")
async def get_waifuapi(command: str, data: str):
    if command == "chat":
        msg = data
        # ----------- Create Response --------------------------
        msg = talk.construct_msg(msg, talk.history_loop_cache)  # construct message input and cache History model
        ## ----------- Will move this to server later -------- (16GB ram needed at least)
        inputs = tokenizer(msg, return_tensors='pt')
        if use_gpu:
            inputs = inputs.to(device)
        print("generate output ..\n")
        out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 80, #todo 200 ?
                             pad_token_id=tokenizer.eos_token_id)
        conversation = tokenizer.decode(out[0])
        print("conversation .. \n" + conversation)

        ## --------------------------------------------------

        ## get conversation in proper format and create history from [last_idx: last_idx+2] conversation
        # talk.split_counter += 0
        print("get_current_converse ..\n")
        current_converse = talk.get_current_converse(conversation)
        print("answer ..\n") # only print waifu answer since input already show
        print(current_converse)
        # talk.history_loop_cache = '\n'.join(current_converse)  # update history for next input message

        # -------------- use machine translation model to translate to japanese and submit to client --------------
        print("cleaning ..\n")
        cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse[1])  # clean text for speech
        cleaned_text = cleaned_text.split("Lilia: ")[-1]
        cleaned_text = cleaned_text.replace("<USER>", "Fuse-kun")
        cleaned_text = cleaned_text.replace("\"", "")
        if cleaned_text:
            print("cleaned_text\n"+ cleaned_text)

            txt = cleaned_text  # initialize translated text as empty by default
            if translation:
                txt = translator.translate(cleaned_text)  # translate to [language] if translation is enabled
                print("translated\n" + txt)

            # ----------- Waifu Expressing ----------------------- (emotion expressed)
            emotion = talk.emotion_analyze(current_converse[1])  # get emotion from waifu answer (last line)
            print(f'Emotion Log: {emotion}')
            emotion_to_express = 'netural'
            if 'joy' in emotion:
                emotion_to_express = 'happy'

            elif 'anger' in emotion:
                emotion_to_express = 'angry'

            print(f'Emotion to express: {emotion_to_express}')

            return JSONResponse(content=f'{emotion_to_express}<split_token>{txt}')
        else:
            return JSONResponse(content=f'NONE<split_token> ')
    elif command == "story":
        msg = data
        # ----------- Create Response --------------------------
        msg = talk.construct_msg(msg, talk.history_loop_cache) # construct message input and cache History model
        ## ----------- Will move this to server later -------- (16GB ram needed at least)
        inputs = tokenizer(msg, return_tensors='pt')
        if use_gpu:
            inputs = inputs.to(device)
        out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 100, pad_token_id=tokenizer.eos_token_id)
        conversation = tokenizer.decode(out[0])
        print("conversation" + conversation)

        ## --------------------------------------------------

        ## get conversation in proper format and create history from [last_idx: last_idx+2] conversation
        talk.split_counter += 0
        current_converse = talk.get_current_converse(conversation)[:talk.split_counter][talk.split_counter-2:talk.split_counter]
        print("answer" + conversation) # only print waifu answer since input already show
        talk.history_loop_cache = '\n'.join(current_converse) # update history for next input message

        # -------------- use machine translation model to translate to japanese and submit to client --------------
        cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse[-1]) # clean text for speech
        
        translated = '' # initialize translated text as empty by default
        if translation:
            translated = translator.translate(cleaned_text) # translate to [language] if translation is enabled

        return JSONResponse(content=f'{current_converse[-1]}<split_token>{translated}')
    
    if command == "reset":
        talk.conversation_history = ''
        talk.history_loop_cache = ''
        talk.split_counter = 0
        return JSONResponse(content='Story reseted...')

if __name__ == "__main__":
    import uvicorn
    import socket # check if port is available
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 8267
    try:
        s.bind(("localhost", port))
        s.close()
    except socket.error as e:
        print(f"Port {port} is already in use")
        exit()
    uvicorn.run(app, host="0.0.0.0", port=port)
