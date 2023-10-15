import os
import json
import math
import wget
from glob import glob
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .Sovits import utils
from .Sovits.data_utils import UnitAudioLoader, UnitAudioCollate
from .Sovits.models import SynthesizerTrn
from scipy.io.wavfile import write
from torchaudio.functional import resample
torchaudio.set_audio_backend("soundfile") # use soundfile backend, due to error with sox backend

class vits_vc_inference:
    def __init__(self, force_load_model=False, F0=None) -> None:
        file_root = os.path.dirname(os.path.abspath(__file__))
        self.zoo_path = f'{file_root}/zoo/'
        
        print("Initializing Waifu Voice Conversion Pipeline...")
        # ask if download zoo model or select from local
        want_to_use_zoo = input('Download New model from zoo? (Y/n): ').lower() in ['y', '']
        if want_to_use_zoo:
            name, model_link, config_link = self.__select_model_from_zoo()
            self.pretrain_model_name = name
            self.model_link = model_link
            self.config_link = config_link
        else:
            print('No zoo model selected. Using local/cached model...')
            
        self.model_root = f'{file_root}/models/'
        if want_to_use_zoo: # download zoo model
            os.makedirs(f'{self.model_root}/{self.pretrain_model_name}/', exist_ok=True)
            load_checkpoint = True
            if not force_load_model:
                print('No checkpoint detected. Downloading checkpoint...')
                print(f'Using model: {self.pretrain_model_name}')
                print(f'Link: {self.model_link}')
                load_checkpoint = input('Load checkpoint? (Y/n): ').lower() in ['y', '']

            if load_checkpoint:
                print('Downloading checkpoint...')
                print(f'Link: {self.model_link}')
                save_model_at = f'{self.model_root}/{self.pretrain_model_name}/model.pth'
                save_config_at = f'{self.model_root}/{self.pretrain_model_name}/config.json'
                if os.path.exists(save_model_at):
                    print('Removing old checkpoint...')
                    print(f'Path: {save_model_at}')
                    os.remove(save_model_at)
                print(f'loading new model...')
                print(f'Saving checkpoint to {save_model_at}')
                wget.download(self.model_link, save_model_at)

                if os.path.exists(save_config_at):
                    print('Removing old config...')
                    print(f'Path: {save_config_at}')
                    os.remove(save_config_at)
                print('Downloading config...')
                print(f'Link: {self.config_link}')
                print(f'Saving config to {save_config_at}')
                wget.download(self.config_link, save_config_at)

        model_list = glob(f'{file_root}/models/*/')
        if len(model_list) > 1:
            print('Multiple models detected. Please select one:')
            for i, model in enumerate(model_list):
                print(f'{i}. {model}')
            model_index = int(input('Model index: '))
            pretrain_model_path = model_list[model_index]
        elif len(model_list) == 1:
            pretrain_model_path = model_list[0]
        else:
            print('No model detected. Please download one from https://huggingface.co/openwaifu/SoVits-VC-Chtholly-Nota-Seniorious-0.1/resolve/main/chtholly.pth')
            exit(0)

        # selected model
        pretrain_model_pth_path = f'{pretrain_model_path}/model.pth'
        pretrain_model_config_path = f'{pretrain_model_path}/config.json'
        
        print(f'Using model {pretrain_model_path}')
        # load content encoder
        self.hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")
        # load synthesizer, construct model and load checkpoint + config
        print(f'Loading config from {pretrain_model_config_path}')
        self.hps = utils.get_hparams_from_file(pretrain_model_config_path) # load config
        self.net_g = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            # n_speakers=self.hps.data.n_speakers,
            **self.hps.model)
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(f"{pretrain_model_pth_path}", self.net_g, None)
    
    def convert(self, audio, sr:int, from_file=False, save_path=None, F0=None):
        if from_file:
            audio, sr = torchaudio.load(audio)
        resampled_sr = 22050
        resampled = resample(audio, sr, resampled_sr)
        source = resampled.unsqueeze(0)
        with torch.inference_mode():
            # Extract speech units
            unit = self.hubert.units(source)
            unit_lengths = torch.LongTensor([unit.size(1)])
            # for multi-speaker inference
            # sid = torch.LongTensor([4])
            # Synthesize audio
            audio = self.net_g.infer(unit, unit_lengths, F0=F0, noise_scale=.667, noise_scale_w=0.8, length_scale=1.0)[0][0,0].data.float().numpy()
            # for multi-speaker inference
            # audio = net_g.infer(unit, unit_lengths, sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1.0)[0][0,0].data.float().numpy()
        if save_path:
            write(save_path, resampled_sr, audio)
        return audio, resampled_sr
    
    def __select_model_from_zoo(self) -> "tuple[str, str, str]":
        """
        return: (model_name, model_link, config_link)
        """
        print('---- List available Waifu Voice models from zoo... ----')
        model_names:list[str] = []
        for model_path in glob(f'{self.zoo_path}/*/meta.json'):
            model_name = os.path.basename(os.path.dirname(model_path)) # folder name
            model_meta = json.load(open(model_path, 'r'))
            print('-', model_name)
            print('  - Author:', model_meta['AUTHOR'])
            print('  - Description:', model_meta['DESCRIPTION'])
            print('  - Link:', model_meta['ORIGIN'])
            print('------------')
            model_names.append(model_name)
        
        selected_model_name = input('Select model(from_name): ')
        # validate model name
        if selected_model_name.strip() in model_names:
            print(f'Selected model: {selected_model_name}')
            meta = json.load(open(f'{self.zoo_path}/{selected_model_name}/meta.json', 'r'))
            return selected_model_name, meta['MODEL_LINK'], meta['CONFIG_LINK']
        else:
            print('Invalid model name. Please try again.')
            return self.__select_model_from_zoo()