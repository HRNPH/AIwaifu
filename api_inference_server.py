from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from conversation import character_msg_constructor
from anime_tts.custom_inference import tts # text to speech from huggingface
from vtube_studio import Char_control
import romajitable # temporary use this since It'll blow up our ram if we use Machine Translation Model
from playsound import playsound # play talking sound
import scipy.io.wavfile as wavfile

# ---------- load Conversation model ----------
# ----------- Will move this to server later -------- (16GB ram needed at least) for 1.3b
# tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-350m")
# config = AutoConfig.from_pretrained("PygmalionAI/pygmalion-350m")
# config.is_decoder = True
# model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-350m", config=config)
# load model at half precision

print("Initilizing model....")
print("Loading langugage model...")
tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-1.3b", use_fast=True)
config = AutoConfig.from_pretrained("PygmalionAI/pygmalion-1.3b", is_decoder=True)
model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-1.3b", config=config)
# model.half() # load model at half precision Only work for GPU
print("Loading machine translation model...")
lmtokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
lmmodel = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
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
        msg = talk.construct_msg(msg, talk.history_loop_cache) # construct message input and cache History model
        ## ----------- Will move this to server later -------- (16GB ram needed at least)
        inputs = tokenizer(msg, return_tensors='pt')
        out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 100, pad_token_id=tokenizer.eos_token_id)
        conversation = tokenizer.decode(out[0])
        ## --------------------------------------------------

        ## get conversation in proper format and create history from [last_idx: last_idx+2] conversation
        talk.split_counter += 2
        current_converse = talk.get_current_converse(conversation)[:talk.split_counter][talk.split_counter-2:talk.split_counter]
        print(conversation) # only print waifu answer since input already show
        talk.history_loop_cache = '\n'.join(current_converse) # update history for next input message

        # -------------- use machine translation model to translate to japanese and submit to client --------------
        cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse[-1]) # clean text for speech
        inputs = lmtokenizer([cleaned_text], return_tensors='pt')
        outs = lmmodel.generate(**inputs, forced_bos_token_id=lmtokenizer.lang_code_to_id['jpn_Jpan'])
        translated = lmtokenizer.batch_decode(outs, skip_special_tokens=True)[0]

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