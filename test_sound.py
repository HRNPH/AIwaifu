from anime_tts.custom_inference import tts # text to speech from huggingface
import romajitable # temporary use this since It'll blow up our ram if we use Machine Translation Model
from playsound import playsound # play talking sound
import soundfile as sf
import scipy.io.wavfile as wavfile
import wave
import pyaudio

translated = romajitable.to_kana('ah! yamete! shotto! dame!!').hiragana # translate to Japanese
_, (sr, audio) = tts(translated, 0)
# write file as float32
wavfile.write('./audio_cache/converse.wav', sr, audio)
# playsound(u'./audio_cache/converse.wav')


# play audio laoded using soundfile
data, samplerate = sf.read('./audio_cache/converse.wav', dtype='float32')
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=samplerate,
                output=True)
stream.write(data.tobytes())
stream.stop_stream()
stream.close()
