from voice_conversion.inference import vits_vc_inference
import torchaudio
import scipy.io.wavfile as wav

converter = vits_vc_inference()
source, sr = torchaudio.load("Untitled.wav")
converted, sr = converter.inference(source, sr)
wav.write("converted.wav", sr, converted)