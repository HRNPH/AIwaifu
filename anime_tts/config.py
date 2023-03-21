import os


def get_pth_file(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith("pth"):
                file_list.append(os.path.join(root, file))

    return file_list


pth_path = f"./anime_tts/1374_epochs.pth"
config_json = "./anime_tts/configs/config.json"
