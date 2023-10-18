from tts.tts import auto_tts

tts_model = auto_tts(model_selection="gtts")
# tts_model.tts('おはよごさいます わたしは さいとちゃんですは', "test.wav", language='en')
# text = """My name is Lilia. I'm a 18 year old elf from the far north. I've been traveling with a group of adventurers for a while, and we're on our way to the high mountain peak."""
text = "もう少し食べたいです"
tts_model.tts(text, "test.wav", language='jp')
