import os
import json
import math
import wget
from glob import glob
import soundfile as sf # used to save audio
import torch
import torchaudio
from .RVC.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from .RVC.python_inference import load_hubert, create_vc_fn, VC
from .RVC.config import (
    is_half,
    device
)
from scipy.io.wavfile import write
from torchaudio.functional import resample
torchaudio.set_audio_backend("soundfile") # use soundfile backend, due to error with sox backend

class vc_inference:
    def __init__(self, force_load_model=False, F0=None) -> None:
        file_root = os.path.dirname(os.path.abspath(__file__))
        self.config_path = f'{file_root}/config.json'
        self.zoo_path = f'{file_root}/zoo/'
        
        print("Initializing Waifu Voice Conversion Pipeline...")
        # ask if download zoo model or select from local
        want_to_use_zoo = input('Download New model from zoo? (Y/n): ').lower() in ['y', '']
        if want_to_use_zoo:
            name, checkpoint_link, feature_retrieval_library_link, feature_file_link = self.__select_model_from_zoo()
            self.pretrain_model_name = name
            self.checkpoint_link = checkpoint_link
            self.feature_retrieval_library_link = feature_retrieval_library_link
            self.feature_file_link = feature_file_link

        else:
            print('No zoo model selected. Using local/cached model...')
            
        self.model_root = f'{file_root}/models/'
        if want_to_use_zoo: # download zoo model
            os.makedirs(f'{self.model_root}/{self.pretrain_model_name}/', exist_ok=True)
            load_checkpoint = True
            if not force_load_model:
                print('No checkpoint detected. Downloading checkpoint...')
                print(f'Using model: {self.pretrain_model_name}')
                print(f'Link: {self.checkpoint_link}')
                load_checkpoint = input('Load checkpoint? (Y/n): ').lower() in ['y', '']

            if load_checkpoint:
                print('Downloading checkpoint...')
                print(f'Link: {self.checkpoint_link}')
                save_checkpoint_at = f'{self.model_root}/{self.pretrain_model_name}/model.pth'
                save_feature_retrieval_library_at = f'{self.model_root}/{self.pretrain_model_name}/feature_retrieval_library.index'
                save_feature_file_at = f'{self.model_root}/{self.pretrain_model_name}/feature.npy'
                if os.path.exists(save_checkpoint_at):
                    print('Removing old checkpoint...')
                    print(f'Path: {save_checkpoint_at}')
                    os.remove(save_checkpoint_at)
                print(f'loading new model...')
                print(f'Saving checkpoint to {save_checkpoint_at}')
                wget.download(self.checkpoint_link, save_checkpoint_at)

                if os.path.exists(save_feature_retrieval_library_at):
                    print('Removing old feature retrieval library...')
                    print(f'Path: {save_feature_retrieval_library_at}')
                    os.remove(save_feature_retrieval_library_at)
                print('Downloading feature retrieval library...')
                print(f'Saving feature retrieval library to {save_feature_retrieval_library_at}')
                wget.download(self.feature_retrieval_library_link, save_feature_retrieval_library_at)

                if os.path.exists(save_feature_file_at):
                    print('Removing old feature file...')
                    print(f'Path: {save_feature_file_at}')
                    os.remove(save_feature_file_at)
                print('Downloading feature file...')
                print(f'Saving feature file to {save_feature_file_at}')
                wget.download(self.feature_file_link, save_feature_file_at)

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
        pretrain_model_feature_retrieval_library_path = f'{pretrain_model_path}/feature_retrieval_library.index'
        pretrain_model_feature_file_path = f'{pretrain_model_path}/feature.npy'
        print(f'Using model {pretrain_model_path}')

        # load content encoder (Hubert)
        huber_path = f'{file_root}/hubert.pt'
        Load_Hubert = False
        if not os.path.exists(huber_path):
            Load_Hubert = True
        else:
            print('Hubert model already exists. want to download a new one?')
            if input('(Y/n): ').lower() in ['y', '']:
                self.__load_hubert_model(file_root)
                Load_Hubert = True

        if Load_Hubert:
            print('Downloading Hubert...')
            hubert_model_name, hubert_checkpoint_link = self.__selet_hubert_model()
            if os.path.exists(huber_path):
                print('Removing old Hubert model...')
                print(f'Path: {huber_path}')
                os.remove(f'{file_root}/hubert.pt')

            print('Downloading Hubert...')
            print(f'Using model: {hubert_model_name}')
            print(f'Link: {hubert_checkpoint_link}')
            wget.download(hubert_checkpoint_link, huber_path)
        else:
            print('Using cached Hubert model...')

        load_hubert(huber_path)

        # load pretrain model
        print('Loading pretrain model...')
        cpt = torch.load(pretrain_model_pth_path, map_location="cpu")
        tgt_sr = cpt["config"][-1] # target sample rate
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
        # create voice conversion function
        self.vc_fn = create_vc_fn(tgt_sr, net_g, vc, if_f0, pretrain_model_feature_retrieval_library_path, pretrain_model_feature_file_path)

    def convert(self, audio_path: str, save_path=None, vc_transpose: int = 0, vc_f0method: str = "harvest", vc_index_ratio: int = 0.6):
        if vc_f0method not in ["harvest", "pm"]: # Pitch extraction algorithm, PM is fast but Harvest is better for low frequencies",
            print("Pitch extraction algorithm, PM is fast but Harvest is better for low frequencies")
            raise ValueError("vc_f0method must be one of ['harvest', 'pm']")
        # info, audio = vc_fn(vc_input, vc_transpose, vc_f0method[1], vc_index_ratio)
        info, audio = self.vc_fn(audio_path, vc_transpose, vc_f0method, vc_index_ratio)
        sample_rate, audio_data = audio
        if (sample_rate is None) or (audio_data is None):
            raise ValueError("Audio data is None, There's probably some shit wrong with the Model you downloaded, Please Report it in the repo!")
        if not save_path:
            save_path = "output.wav"
        sf.write(save_path, audio[1], audio[0])
        return audio_data, sample_rate
    
    def __select_model_from_zoo(self) -> "tuple[str, str, str, str]":
        """
        return: (model_name, checkpoint_link, feature_retrieval_library_link, feature_file_link)
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
            return selected_model_name, meta['CHECKPOINT_LINK'], meta['FEATURE_RETRIEVAL_LIBRARY_LINK'], meta['FEATURE_FILE_LINK']
        else:
            print('Invalid model name. Please try again.')
            return self.__select_model_from_zoo()
        
    def __selet_hubert_model(self) -> "tuple[str, str]":
        """
        return: (model_name, checkpoint_link)
        """
        print('---- List available Hubert models... ----')
        model_names:list[str] = []
        model_map = {}
        # read from config
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            for model in config['hubert_models']:
                print('-', model['NAME'])
                print('  - Author:', model['AUTHOR'])
                print('  - Checkpoint:', model['CHECKPOINT_LINK'])
                print('------------')
                model_names.append(model['NAME'])
                model_meta = {
                    'NAME': model['NAME'],
                    'AUTHOR': model['AUTHOR'],
                    'CHECKPOINT_LINK': model['CHECKPOINT_LINK']
                }
                model_map[model['NAME']] = model_meta

        selected_model_name = input('Select model(from_name): ')
        # validate model name
        if selected_model_name.strip() in model_names:
            print(f'Selected model: {selected_model_name}')
            return selected_model_name, model_map[selected_model_name]['CHECKPOINT_LINK']
        else:
            print('Invalid model name. Please try again.')
            return self.__selet_hubert_model()