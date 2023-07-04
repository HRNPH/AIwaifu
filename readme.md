<div align="center">
  <div id="user-content-toc">
    <ul>
      <summary><h1 style="display: inline-block;">AI Waifu</h1></summary>
    </ul>
  </div>
  <p align='center'>an Open sourced customizable AI waifu for everyone</p>
  <p align='center'><a href='https://discord.gg/scGBZKX4'>Join Our Discord!</a></p>
  
  [![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/makes-people-smile.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/0-percent-optimized.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/check-it-out.svg)](https://forthebadge.com)
  
  [![forthebadge](https://forthebadge.com/images/badges/open-source.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/powered-by-black-magic.svg)](https://forthebadge.com)
</div>



# What is it?
aiwaifu is an Open sourced finetunable customizable simpable AI waifu inspired by neuro-sama

the goal is to just giving everyone a foundational platform to develop their own waifu

Powered by opensource AI model for self-hosted/deploy

For how our waifu look like
[Take A Look At This Video!](https://www.youtube.com/watch?v=Up4lwhPO8m0)

<img src="https://i.imgur.com/lrt6WX3.png" width="1000">

Open-Sourced Talkable, Flirtable, streamable, Modifiable! Finetunable! and even LEWDABLE! AIwaifu!!! what more can you ask for? huh?

-  inspired by neuro-sama 

YOUR ONE AND ONLY WAIFU(if you have your own datasets or custom the personality)
## Minimum Specs
### CPU Only
- 12 GB ram or more
- Acceptable runtime tested on i7 7700k (1.00-2.30 min) faster CPU could be better
### Inference With GPU
- Minimum 8GB VRAM require
- Will take at least 7.2 GB VRAM
- Nvidia GPU Only
- Very Fast On k80(Tested) faster or equivalent GPU will be pretty fast too
## Installation
- Make sure you have python3 >= 3.8 installed and > 10GB storage with decent internet connection(for model weight)
- Make sure you have Installed C/C++ build tools and have Cmake installted (if not follow [this Issue](https://github.com/HRNPH/AIwaifu/issues/20#issuecomment-1507086317))
- Make sure to have [GIT LFS](https://git-lfs.com/) Installed to handle large file download in git
- clone the repo & install packages
```bash
git clone https://github.com/HRNPH/AIwaifu.git
cd ./AIwaifu
```
```bash
# may contain some bloated packge(since I didn't clean the requirements YET)
# so I recommend install this on venv

# ---- optional -----
python -m venv venv
./venv/script/activate  # for windows
# source ./venv/bin/activate # for linux
# --------------------

# you need to uninstall webscoket module and install websocket-client (which was included in the requirements for it to work)
pip uninstall websocket
pip install -r ./requirements.txt

# You need to install the monotonic_align module for sovits to work
cd AIVoifu/voice_conversion/Sovits/monotonic_align
python setup.py build_ext --inplace
```
- Download and start [Vtube-Studio](https://store.steampowered.com/app/1325860/VTube_Studio/) (Just download it from steam)
- Install [VTS desktop audio plugin](https://www.youtube.com/watch?v=IiZ0JrGd6BQ&t=11s) by [Lua Lucky](https://www.youtube.com/watch?v=IiZ0JrGd6BQ&t=11s) CONSIDER SUBSCRIBING TO HER! She's Cute Vtuber & Developer then open it and connect to Vtube Studio
- Just follow lua-lucky videoOpen the plugin API at port 8001 in the app setting (or any port you desired but you need to modify the code)


- Start the server (In your home server in local network or on you computer 12GB ram is a minimum recommendation)
> The software was splited into http server for model inference(since I need to use my home server cause the model take too much RAM(RAM Not VRAM) > 12GB required >= 16 recommended)
```bash
# this run on localhost 8267 by default
python ./api_inference_server.py
```

- Start the client
```bash
# this will connect to all the server (Locally)
# it is possible to host the api model on the external server, just beware of security issue
# I'm planning to make a docker container for hosting on cloud provider for inference, but not soon
python ./main.py
```
- Open the Vtuber Studio(VTS) and allow access

### Quicknote
> - Current TTS model was VITS pretrained model from
> https://huggingface.co/docs/hub/spaces-config-reference
> (This may be change later on for more customizable option)
> - The Language model we're using is Pygmalion1.3b
> - The reason TTS was in japanese's because it's cuter!!!! we translate model outputs from English to Japanese using Facebook/nllb-600m model


> Sometime shit can be broke(Especially in the server)
> If you happen to found what's broken feel free to open an issue or pull requests!!!

# Code Of Conduct
- Everything We Made Is OpenSourced, Free & Customizable To the Very Core
- We'll never include proprietary Model (ChatGPT/ETC...) since it'll conflict with what we state above

# How Can I help?
Take a look at the issue & feel free to ask question or suggest new features or tests our model to the limit!
join the discussion and comment on model performance here https://github.com/HRNPH/AIwaifu/discussions/25

all help will be appreciated :DD
