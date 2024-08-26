# ----------- Waifu Vocal Pipeline -----------------------
from AIVoifu.tts import tts
from AIVoifu.voice_conversion import vc_inference as vc
class tts_pipeline:
    def __init__(self, tts_model=None) -> None:
        print('Loading Waifu Vocal Pipeline...')
        self.cache_root = './audio_cache'
        self.model = tts.auto_tts(tts_model)
        self.vc_model = vc.vc_inference(force_load_model=False)
        print('Loaded Waifu Vocal Pipeline')

    def tts(self, text, language, voice_conversion=True, save_path=None):
        # text to speech
        if not save_path:
            save_path = f'{self.cache_root}/dialog_cache.wav'
        self.model.tts(text, save_path, language)
        # voice conversion
        if voice_conversion:
            self.vc_model.convert(save_path, save_path=save_path, vc_transpose=3)
        return save_path
    

if __name__ == '__main__':
    model = tts_pipeline()
    model.tts('Hello This Is A Test Text Anyway', save_path='./test.wav')