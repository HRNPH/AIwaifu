from tts.tts import auto_tts

tts_model = auto_tts(model_selection="gtts")
# tts_model.tts('おはよごさいます わたしは さいとちゃんですは', "test.wav", language='en')
tts_model.tts('Hello World', "test.wav", language='en')
