import gradio as gr
import infer
import config


pth_path = config.pth_path
config_json = config.config_json
net_g_ms, hps = infer.load_model(config_json, pth_path)
sp_dict = {speaker: i for i, speaker in enumerate(hps.speakers)}


def tts(text, speaker):
    text = infer.clean_text(text)
    audio = infer.infer(text, net_g_ms, sp_dict[speaker], "demo")
    return "Success", (hps.data.sampling_rate, audio)

app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Basic"):
            tts_input1 = gr.TextArea(label="Enter Text in Chinese / Japanese (150 words limitation)", value="你好呀。")
            tts_input2 = gr.Dropdown(label="Speaker", choices=hps.speakers, value=hps.speakers[0])
            tts_submit = gr.Button("Generate", variant="primary")
            tts_output1 = gr.Textbox(label="Message")
            tts_output2 = gr.Audio(label="Output")
            tts_submit.click(tts, [tts_input1,tts_input2], [tts_output1, tts_output2])


    app.launch()
