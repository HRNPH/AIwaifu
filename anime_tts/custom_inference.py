from anime_tts import infer
from anime_tts import config


pth_path = config.pth_path
config_json = config.config_json
net_g_ms, hps = infer.load_model(config_json, pth_path)
sp_dict = {speaker: i for i, speaker in enumerate(hps.speakers)}
# print(sp_dict)

def tts(text, speaker):
    text = infer.clean_text(text)
    audio = infer.infer(text, net_g_ms, speaker, "demo")
    return "Success", (hps.data.sampling_rate, audio)

if __name__ == "__main__":
    # save the audio file
    import soundfile as sf
    _, audio = tts("ど・よう・わんと・ふぇえる・ごおど", 0)