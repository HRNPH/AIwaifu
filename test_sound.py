from anime_tts.custom_inference import tts # text to speech from huggingface
import romajitable # temporary use this since It'll blow up our ram if we use Machine Translation Model
from playsound import playsound # play talking sound
import scipy.io.wavfile as wavfile

translated = romajitable.to_kana('ah! yamete! shotto! dame!!').hiragana # translate to Japanese
_, (sr, audio) = tts(translated, 0)
# playsound with audio
wavfile.write('./audio_cache/converse.wav', sr, audio)
playsound(u'./audio_cache/converse.wav')