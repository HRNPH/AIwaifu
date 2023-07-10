from voice_conversion.vc_inference import vits_vc_inference
import torchaudio
import scipy.io.wavfile as wav

converter = vits_vc_inference()
torchaudio.set_audio_backend("soundfile")
source, sr = torchaudio.load("test.wav", format="wav")
converted, sr = converter.convert(source, sr)
wav.write("converted.wav", sr, converted)