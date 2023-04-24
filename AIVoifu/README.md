# How to add your own TTS/Voice Conversion pipeline
First of all we need to understand how AIwaifu Vocal Pipeline was design
There's two components
1. TTS: This is a typical text to speech model which you can add your own
2. Voice_Conversion: This is our Heroine on making TTS sound like a cute girl by converting TTS speech using SoVits which you can train easily using our training script

First of all we get TTS voice from TTS pipeline, adn the we do voice conversion on it
## Custom TTS
in the TTS folder you'll found
- tts_base_model: folder
- tts.py

if you wish to add your own TTS model please create a class in tts.py
and in the function constructor please download the weight and cache it under the tts_base_model along with other cache file
```python
# Example code
class CustomTTS:
  def __init__(self) -> None:
    from something import tts_library
    self.model_link = 'Huggingface_link'
    root = os.path.dirname(os.path.abspath(__file__))
    self.model_root_path = f'{root}/base_tts_model/'
    self.model_name = 'custom_tts'
    model_path = f'{self.model_root_path}/{self.model_name}'
    if not os.path.exist(model_path)
      os.mkdir(model_path)
    weigh_path = f'{model_path}/{self.model_name}.pth'
    if not os.path.exist(weigh_path):
      wget.download(self.model_link, weigh_path)
    self.model = tts_library.load(weigh_path)
    print(f'model {self.model_name} initialized')
  
  def tts(self, text:str, save:boolean=True, your_args:any):
    # some preprocessing
    output = self.model.tts(text)
    if save:
      output.save('save_path')
    return output
```
## Custom Voice Conversion
if you're not looking to add new language but just want to custom the model voice easily

Our recommendation is to just Train Voice Conversion pipeline on your own samples using our [Training Notebook](https://colab.research.google.com/drive/1But3y-v4Ok7cnT5bxXLCF49k_CCgp8Hd?usp=sharing)
