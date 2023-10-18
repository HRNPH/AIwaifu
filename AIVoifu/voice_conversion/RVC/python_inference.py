import os
import json
import traceback
import logging
from typing import Union
import numpy as np
import librosa
import soundfile as sf # used to save audio
import torch
import asyncio
from datetime import datetime
from fairseq import checkpoint_utils
from .infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from .vc_infer_pipeline import VC
from .config import (
    is_half,
    device
)
logging.getLogger("numba").setLevel(logging.WARNING)
limitation = os.getenv("SYSTEM") == "spaces"  # limit audio length in huggingface spaces

def create_vc_fn(tgt_sr, net_g, vc, if_f0, file_index, file_big_npy):
    def vc_fn(
        input_audio,
        f0_up_key,
        f0_method,
        index_rate,
    ):
        try:
            if input_audio:
                audio, sr = librosa.load(input_audio, sr=16000, mono=True)
            else:
                if input_audio is None:
                    return "You need to upload an audio", None
                sampling_rate, audio = input_audio
                duration = audio.shape[0] / sampling_rate
                if duration > 20 and limitation:
                    return "Please upload an audio file that is less than 20 seconds. If you need to generate a longer audio file, please use Colab.", None
                audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio.transpose(1, 0))
                if sampling_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            times = [0, 0, 0]
            f0_up_key = int(f0_up_key)
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                0,
                audio,
                times,
                f0_up_key,
                f0_method,
                file_index,
                file_big_npy,
                index_rate,
                if_f0,
            )
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]: npy: {times[0]}, f0: {times[1]}s, infer: {times[2]}s"
            )
            return "Success", (tgt_sr, audio_opt)
        except:
            info = traceback.format_exc()
            print(info)
            return info, (None, None)
    return vc_fn

def load_hubert(hubert_path: str = "hubert_base.pt"):
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

if __name__ == '__main__':
    load_hubert()
    models = []
    with open("weights/model_info.json", "r", encoding="utf-8") as f:
        models_info = json.load(f)
    print("-" * 20)

    for name, info in models_info.items():
        if not info['enable']:
            continue
        title = info['title']
        author = info.get("author", None)
        cover = f"weights/{name}/{info['cover']}"
        index = f"weights/{name}/{info['feature_retrieval_library']}"
        npy = f"weights/{name}/{info['feature_file']}"
        print("Loading:", name, title, author, cover, index, npy)
        cpt = torch.load(f"weights/{name}/{name}.pth", map_location="cpu")
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        del net_g.enc_q
        print(net_g.load_state_dict(cpt["weight"], strict=False))  # 不加这一行清不干净, 真奇葩
        net_g.eval().to(device)
        if is_half:
            net_g = net_g.half()
        else:
            net_g = net_g.float()
        vc = VC(tgt_sr, device, is_half)
        models.append((name, title, author, cover, create_vc_fn(tgt_sr, net_g, vc, if_f0, index, npy)))
        break
    print(models)

    # inference config
    vc_input = 'tts.mp3'
    tts_mode = True
    tts_text = "Hello Guys, this is a test sentence."

    vc_transpose = 0
    vc_index_ratio = 0.6
    vc_f0method = ["pm", "harvest"] # Pitch extraction algorithm, PM is fast but Harvest is better for low frequencies",
    # inference, using first model from tts
    name, title, author, cover, vc_fn = models[0]
    print("Inference:", name, title, author, cover)
    print("-" * 20)
    info, audio = vc_fn(vc_input, vc_transpose, vc_f0method[1], vc_index_ratio)
    print(info)
    # save audio
    sf.write("output.wav", audio[1], audio[0])
    
