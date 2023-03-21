# coding:utf-8
import re

import torch
import unicodedata
from pydub import AudioSegment
from scipy.io.wavfile import write

from anime_tts import commons
from anime_tts import config
from anime_tts import utils
from anime_tts.models import SynthesizerTrn
from anime_tts.text import text_to_sequence

pth_path = config.pth_path
config_json = config.config_json


def wav2mp3(file_name):
    sourcefile = AudioSegment.from_wav(f"{file_name}.wav")
    sourcefile.export(f"{file_name}.mp3", format="mp3")


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text


def clean_text(text):
    jap = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7A3]')  # \uAC00-\uD7A3为匹配韩文的，其余为日文
    text = f"[JA]{text}[JA]" if jap.search(text) else f"[ZH]{text}[ZH]"
    text = unicodedata.normalize('NFKC', text)
    return text


def load_model(config_json, pth_path):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hps_ms = utils.get_hparams_from_file(f"{config_json}")
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        **hps_ms.model).to(dev)
    _ = net_g_ms.eval()
    _ = utils.load_checkpoint(pth_path, net_g_ms)
    return net_g_ms, hps_ms


def infer(text, net_g_ms, speaker_id, out_name):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hps_ms = utils.get_hparams_from_file(f"{config_json}")
    with torch.no_grad():
        stn_tst = get_text(text, hps_ms, cleaned=False)
        x_tst = stn_tst.unsqueeze(0).to(dev)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(dev)
        sid = torch.LongTensor([speaker_id]).to(dev)
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)[0][
            0, 0].data.cpu().float().numpy()
        # write(f"./audio_cache/{out_name}.wav", hps_ms.data.sampling_rate, audio)
        # wav2mp3(out_name)
    return audio


if __name__ == "__main__":
    out_name = "水调歌头"
    speaker_id = 0
    text = "明月几时有，把酒问青天。不知天上宫阙，今夕是何年。我欲乘风归去，又恐琼楼玉宇，高处不胜寒。"
    text = clean_text(text)
    net_g_ms = load_model(config_json, pth_path)
    infer(text, net_g_ms, speaker_id, out_name)
