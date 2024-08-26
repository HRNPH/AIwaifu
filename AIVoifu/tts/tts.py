from typing import Literal, Optional


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

class BaseTTS:
    def __init__(self) -> None:
        self.model_name = "base_tts"
        self.sr = 22050
        self.model = None

    def tts(self, text, out_path, language):
        raise NotImplementedError(f"Please Implement all necessery method in '{self.model_name}' model")

    def supported_languages(self) -> list:
        raise NotImplementedError(f"Please Implement all necessery method in '{self.model_name}' model")
    
    def requested_additional_args(self) -> None:
        # this function will be called when user select this model
        # you can request additional args here by overriding this function, please handle it in the class atritube
        return None

class Gtts(BaseTTS):
    def __init__(self) -> None:
        from gtts import gTTS
        # disable gtts debug
        import logging
        logging.getLogger('gtts').setLevel(logging.CRITICAL)

        self.model_name = "gtts"
        self.sr = 16000
        self.model = gTTS
        # self.language = language

    def tts(self, text, out_path, language):
        tts = self.model(text=text, lang=language)
        tts.save(out_path)

    def supported_languages(self) -> list:
        return ['af', 'ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'gu', 'hi', 'hr', 'hu', 'id', 'is', 'it', 'iw', 'ja', 'jw', 'km', 'kn', 'ko', 'la', 'lv', 'ml', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW', 'zh']
    
    def requested_additional_args(self) -> None:
        # this function will be called when user select this model
        # you can request additional args here, please handle it in the class atritube
        # example:
        selected_language = input("Please Select language Of Speech: ")
        if selected_language not in self.supported_languages():
            raise ValueError(f"Invalid language selection: {selected_language}")
        self.language = selected_language
        return None

class auto_tts: # add your tts model mapping 'key' here
    possible_model = Literal["gtts"]
    # possible_model = Literal['khanomtal11', 'key2', 'key3', ...]
    def __init__(
        self,
        model_selection: Optional[possible_model] = None,
    ) -> None:
        self.model_mapping = {
            # "khanomtal11": khanomtal11,
            "gtts": Gtts,
            # 'key' : your tts class
        }

        # manual model selection and validation if model existed
        if model_selection is None:
            print("---- Available TTS models: ---")
            for key in self.list_all_models():
                supported_languages = self.model_mapping[key]().supported_languages()
                if len("".join(supported_languages)) > 400:
                    print('-', key, "| Supported Language/Speakers", len(supported_languages))
                    print('(There is too much) This is Some Of them.', supported_languages[:10])
                else:
                    print('-', key, "| Supported Language/Speakers", self.model_mapping[key]().supported_languages())
            print("-------------------------------")
            model_selection = input("Select TTS model: ")

        if not self.__validate_model(model_selection):
            raise ValueError(f"Invalid model selection: {model_selection}")
        
        self.model:BaseTTS = self.model_mapping[model_selection]()
        # request additional args if needed, will do nothing if it was not implemented
        self.model.requested_additional_args()

    def tts(self, text, out_path, language, model_name=None, **kwargs):
        if model_name is not None: # use this model_name instead
            self.model_mapping[model_name]().tts(text, out_path, **kwargs)
        else:
            # use default model, initialized in __init__
            self.model.tts(text, out_path, language, **kwargs)

    def list_all_models(self) -> list:
        return list(self.model_mapping.keys())
    
    def __validate_model(self, model_name: str) -> bool:
        return model_name in self.model_mapping.keys()