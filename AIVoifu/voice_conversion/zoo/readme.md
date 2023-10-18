# What is Voice Conversion Model Zoo?
This is a collection of voice conversion models trained by Community.
which can be used as a waifu voice conversion model. or a reference for training your own model.

## To add your model to the zoo
1. Fork this repo
2. create folder in `zoo` folder, name it as your model name
3. create a meta.json file in your folder, which contains the following information:
    - DESCRIPTION: a short description of your model
    - ORIGIN: the link to your model
    - AUTHOR: your name
    - LICENSE: the license of your model
    - CHECKPOINT_LINK: the link to your checkpoint file (must be a pytorch model file .pth)
    - FEATURE_RETRIEVAL_LIBRARY_LINK: the link to your feature retrieval library file (.index)
    - FEATURE_FILE_LINK: the link to your feature file (.npy)
4. create a pull request
5. wait for review :DDD
```json
{
    "DESCRIPTION": "alice-rvc",
    "ORIGIN": "https://huggingface.co/spaces/zomehwh/sovits-models/raw/main/models/alice",
    "AUTHOR": "zomehwh",
    "LICENSE": "MIT",
    "CHECKPOINT_LINK": "https://huggingface.co/spaces/zomehwh/rvc-models/resolve/main/weights/alice/alice.pth",
    "FEATURE_RETRIEVAL_LIBRARY_LINK": "https://huggingface.co/spaces/zomehwh/rvc-models/resolve/main/weights/alice/added_IVF141_Flat_nprobe_4.index",
    "FEATURE_FILE_LINK": "https://huggingface.co/spaces/zomehwh/rvc-models/resolve/main/weights/alice/total_fea.npy"
}
```