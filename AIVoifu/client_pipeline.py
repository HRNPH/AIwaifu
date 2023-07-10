# ----------- Waifu Vocal Pipeline -----------------------
from AIVoifu.tts import tts
from AIVoifu.voice_conversion import vc_inference as vc
class tts_pipeline:
    def __init__(self) -> None:
        print('Loading Waifu Vocal Pipeline...')
        self.cache_root = './audio_cache'
        self.model = tts.auto_tts()
        self.vc_model = vc.vits_vc_inference(load_model=True)
        print('Loaded Waifu Vocal Pipeline')

    def tts(self, text, voice_conversion=True, save_path=None):
        if not save_path:
            save_path = f'{self.cache_root}/dialog_cache.wav'
        self.model.tts(text, save_path)
        if voice_conversion:
            self.vc_model.convert(save_path, 22050, from_file=True, save_path=save_path)
        return save_path