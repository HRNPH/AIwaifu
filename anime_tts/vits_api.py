import re
import time
import infer
import config
import uvicorn
import asyncio
from starlette.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()

pth_path = config.pth_path
config_json = config.config_json
net_g_ms, hps = infer.load_model(config_json, pth_path)
sp_dict = {speaker: i for i, speaker in enumerate(hps.speakers)}


@app.get("/tts", response_class=FileResponse)
async def read_item(text: str, speaker: str):
    print(text, speaker)
    text = infer.clean_text(text)
    infer.infer(text, net_g_ms, sp_dict[speaker], "demo")
    return "./demo.mp3"


uvicorn.run(app, host="0.0.0.0")
