from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from conversation import character_msg_constructor
from anime_tts.custom_inference import tts # text to speech from huggingface
from vtube_studio import Char_control
import romajitable # temporary use this since It'll blow up our ram if we use Machine Translation Model
from playsound import playsound # play talking sound

# ---------- load Conversation model ----------
# ----------- Will move this to server later -------- (16GB ram needed at least) for 1.3b
# tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-350m")
# config = AutoConfig.from_pretrained("PygmalionAI/pygmalion-350m")
# config.is_decoder = True
# model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-350m", config=config)
# load model at half precision
tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-350m", use_fast=False)
config = AutoConfig.from_pretrained("PygmalionAI/pygmalion-350m", is_decoder=True)
model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-350m", config=config)
model.half() # load model at half precision
# --------------------------------------------------

# --------- Define Waifu personality ----------
talk = character_msg_constructor('Lilia', """Species("Elf")
Mind("sexy" + "Horny" + "Loving" + "Based as Fuck")
Personality("sexy" + "Horny"+ "sadistic + "Loving" + "Based as Fuck")
Body("160cm tall" + "5 foot 2 inches tall" + "small breasts" + "white" + "slim")
Description("Lilia likes giving handjob" + "She has an obsession with you")
Loves("Cats" + "Birds" + "Waterfalls")
Sexual Orientation("Straight" + "Hetero" + "Heterosexual")""")
# -----------------------------

# initialize Vstudio Waifu Controller
waifu = Char_control()
# vtube.express(express) # use this to express feeling

split_counter = 0
history = ''
while True:
    con = str(input("You: "))
    if con.lower() == 'exit':
        print('Stopping...')
        break # exit prototype
    
    # ----------- Create Response --------------------------
    msg = talk.construct_msg(con, history) # construct message input and cache History model
    ## ----------- Will move this to server later -------- (16GB ram needed at least)
    inputs = tokenizer(msg, return_tensors='pt')
    out = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + 100, pad_token_id=tokenizer.eos_token_id)
    conversation = tokenizer.decode(out[0])
    ## --------------------------------------------------

    ## get conversation in proper format and create history from [last_idx: last_idx+2] conversation
    converses = talk.get_current_converse(conversation)
    split_counter += 2
    current_converse = '\n'.join(talk.get_current_converse(conversation)[:split_counter][split_counter-2:split_counter])
    print(current_converse[-1]) # only print waifu answer since input already show
    history = current_converse # update history for next input message
    # --------------------------------------------------

    # ----------- Waifu Create Talking Audio -----------------------
    ## generate speaking voice (Translates to Japanese and then TTS)/(Japanglish TTS)[take lower memory but not as enjoyable]
    ## we'll use Japanglish TTS for prototyping, we'll use Japanese TTS for production
    cleaned_text = talk.clean_emotion_action_text_for_speech(current_converse) # delete *describe* in text and left with only "speaking" part
    translated = romajitable.to_kana(cleaned_text).hiragana # translate to Japanese
    _, (sr, audio) = tts(translated, 0)
    # --------------------------------------------------


    # ----------- Waifu Expressing ----------------------- (emotion expressed)
    emotion = talk.emotion_analyzer(current_converse[-1]) # get emotion from waifu answer (last line)
    print(f'Emotion Log: {emotion}')
    emotion_to_express = None
    if 'joy' in emotion:
        emotion_to_express = 'happy'

    elif 'anger' in emotion:
        emotion_to_express = 'angry'

    if emotion_to_express: ## express emotion
        waifu.express(emotion_to_express) # express emotion in Vtube Studio

    # --------------------------------------------------

    # ----------- Waifu Talking -----------------------
    playsound(audio, sr)
    # --------------------------------------------------