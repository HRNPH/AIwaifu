
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from Conversation.conversation import character_msg_constructor
from transformers import pipeline
from pysentimiento import create_analyzer

### --- websocket server setup
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config

def get_text_model(pretrained_name, device):

    
    model = pipeline("text-generation", model=pretrained_name, device=device, max_length=100)
    # , torch_dtype="auto"
    
    # tokenizer = AutoTokenizer.from_pretrained("openthaigpt/openthaigpt-1.0.0-7b-chat", cache_dir=None)
    # model = AutoModelForCausalLM.from_pretrained("openthaigpt/openthaigpt-1.0.0-7b-chat")

    # tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt2-base-thai")
    # model = AutoModelForCausalLM.from_pretrained("flax-community/gpt2-base-thai")

    # pretrained_name = "flax-community/gpt2-base-thai"
    # model = pipeline(
    #     "text-generation",
    #     model=pretrained_name,
    #     tokenizer=pretrained_name
    # ) 
    return model

# use fast api instead
app = FastAPI()
emotion_analyzer = create_analyzer(task="emotion", lang="en")
talk = character_msg_constructor(config.character_name, config.character_persona, emotion_analyzer)
tokenizer = AutoTokenizer.from_pretrained(config.pretrained_name, use_fast=True)
model_config = AutoConfig.from_pretrained(config.pretrained_name, is_decoder=True)
model = AutoModelForCausalLM.from_pretrained(config.pretrained_name, config=model_config, )

def text_model_inference(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt')
    inputs = inputs.to(device)
    out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 80, #todo 200 ?
                            pad_token_id=tokenizer.eos_token_id)
    conversation = tokenizer.decode(out[0])
    return conversation

class WaifuRequest(BaseModel):
    command: str
    data: str
    
# do a http server instead
@app.post("/waifuapi")
def get_waifuapi(rb: WaifuRequest):
    command = rb.command 
    data = rb.data
    if command == "chat":
        msg = data
        # ----------- Create Response --------------------------
        print("generate output ..")
        msg = talk.construct_msg(msg)  # construct message input and cache History model
        conversation = text_model_inference(msg, model, tokenizer, config.device)
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
        cleaned_text = cleaned_text.split(f"{config.character_name}: ")[-1]
        # cleaned_text = cleaned_text.replace("<USER>", "Fuse-kun")
        cleaned_text = cleaned_text.replace("\"", "")
        if cleaned_text:
            print("cleaned_text\n"+ cleaned_text)

            txt = cleaned_text  # initialize translated text as empty by default
            
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
        raise NotImplementedError("using command 'story', not supported yet")
        msg = data
        # ----------- Create Response --------------------------
        msg = talk.construct_msg(msg) # construct message input and cache History model
        ## ----------- Will move this to server later -------- (16GB ram needed at least)
        inputs = tokenizer(msg, return_tensors='pt')
        inputs = inputs.to(config.device)
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
        # talk.history_loop_cache = ''
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
