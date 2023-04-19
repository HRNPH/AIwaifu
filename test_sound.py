# new-version of our voicing pipeline
# tts
cache_root = './audio_cache'
from AIVoifu.tts import tts
model = tts.OpenJtalk()
model.tts('こんにちは、私はあなたのお姉さんです。', f'{cache_root}/test.wav')
# voice conversion
from AIVoifu.voice_conversion import vc_inference as vc
vc_model = vc.vits_vc_inference(load_model=True)
vc_model.convert('./audio_cache/test.wav', 22050, from_file=True, save_path='./audio_cache/test_vc.wav')