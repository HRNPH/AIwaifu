# AI-Waifu
[Take A Look At This Video!](https://www.youtube.com/watch?v=Up4lwhPO8m0)


<img src="https://i.imgur.com/lrt6WX3.png" width="1000">

Open-Sourced Talkable, Flirtable, streamable, Modifiable! Finetunable! and even LEWDABLE! AIwaifu!!! what more can you ask for? huh?

-  inspired by neuro-sama 

YOUR ONE AND ONLY WAIFU(if you have your own datasets or custom the personality)

## Installation
- clone the repo & install packages
```bash
# may contain some bloated packge(since I didn't clean the requirements YET)
# so I recommend install this on venv
pip install -t ./requirements.txt
```
- Download and start [Vtube-Studio](https://store.steampowered.com/app/1325860/VTube_Studio/)
- Open the plugin API at port 8001 in the app setting (or any port you desired but you need to modify the code) 
- Install [VTS desktop audio plugin](https://www.youtube.com/watch?v=IiZ0JrGd6BQ&t=11s) by [Lua Lucky](https://www.youtube.com/watch?v=IiZ0JrGd6BQ&t=11s) CONSIDER SUBSCRIBING TO HER! She's Cute Vtuber & Developer then open it and connect to Vtube Studio


- Start the server (In your home server in local network or on you computer 12GB ram is a minimum recommendation)
> The software was splited into http server for model inference(since I need to use my home server cause the model take too much RAM > 12GB required >= 16 recommended)
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
