import os
import json
from scipy.io import wavfile
import numpy as np
from typing import Literal


# write your own tts class and place it in this folder
# model weight will be downloaded and save at tts_base_model
# class khanomtal11: deprecated (now was used as an example)
#     def __init__(self) -> None:
#         from TTS.api import TTS

#         root = os.path.dirname(os.path.abspath(__file__))
#         # change config.json to match your model
#         json_path = os.path.join(root, "config.json")
#         configs = json.load(open(json_path, "r"))
#         configs["speakers_file"] = os.path.join(root, "speakers.pth")
#         configs["language_ids_file"] = os.path.join(root, "language_ids.json")
#         configs["model_args"]["speakers_file"] = os.path.join(root, "speakers.pth")
#         configs["model_args"]["speaker_ids_file"] = os.path.join(
#             root, "speaker_ids.json"
#         )
#         configs["model_args"]["speaker_encoder_config_path"] = os.path.join(
#             root, "config_se.json"
#         )
#         configs["model_args"]["speaker_encoder_model_path"] = os.path.join(
#             root, "model_se.pth"
#         )
#         self.sr = configs["audio"]["sample_rate"]
#         # save the new config
#         json.dump(configs, open(json_path.replace("config", "nconfig"), "w"), indent=4)

#         self.model = TTS(
#             model_path=os.path.join(root, "best_model.pth"),
#             config_path=os.path.join(root, "nconfig.json"),
#             progress_bar=False,
#             gpu=False,
#         )
#         self.model.model_name = "khanomtal11"

#     def tts(self, text, out_path, speaker=0, language=3):
#         self.model.tts_to_file(
#             text=text,
#             file_path=out_path,
#             speaker=self.model.speakers[speaker],
#             language=self.model.languages[language],
#         )

#     def supported_languages(self) -> list:
#         return ['tha_Thai']

class OpenJtalk:
    def __init__(self) -> None:
        import pyopenjtalk

        self.model_name = "openjtalk"
        self.sr = 48000
        self.model = pyopenjtalk

    def tts(self, text, out_path):
        wav, sr = self.model.tts(text)
        wavfile.write(out_path, sr, wav.astype(np.int16))

    def supported_languages(self) -> list:
        return ['ja_JPAN']


class Gtts:
    def __init__(self) -> None:
        from gtts import gTTS

        self.model_name = "gtts"
        self.sr = 48000
        self.model = gTTS

    def tts(self, text, out_path, language="en"):
        tts = self.model(text=text, lang=language)
        tts.save(out_path)

    def supported_languages(self) -> list:
        return ['af', 'ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'gu', 'hi', 'hr', 'hu', 'id', 'is', 'it', 'iw', 'ja', 'jw', 'km', 'kn', 'ko', 'la', 'lv', 'ml', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW', 'zh']
    
class auto_tts: # add your tts model mapping 'key' here
    possible_model = Literal[f"khanomtal11", "openjtalk", "gtts"]
    # possible_model = Literal['key', 'key2', 'key3', ...]
    def __init__(
        self,
        model_selection: possible_model=None
    ) -> None:
        self.model_mapping = {
            # "khanomtal11": khanomtal11,
            "openjtalk": OpenJtalk,
            "gtts": Gtts
            # 'key' : your tts class
        }

        # manual model selection and validation if model existed
        if model_selection is None:
            print("---- Available TTS models: ---")
            for key in self.list_all_models():
                print('-', key, self.model_mapping[key]().supported_languages())
            model_selection = input("Select TTS model: ")
        
        if not self.__validate_model(model_selection):
            raise ValueError(f"Invalid model selection: {model_selection}")

        self.model_selection = model_selection

    def tts(self, text, out_path, model_name=None, **kwargs):
        if model_name is None:
            model_name = self.model_selection
        self.model_mapping[model_name]().tts(text, out_path, **kwargs)

    def list_all_models(self) -> list:
        return list(self.model_mapping.keys())
    
    def __validate_model(self, model_name: str) -> bool:
        return model_name in self.model_mapping.keys()