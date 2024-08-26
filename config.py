
# device = torch.device('cuda')
device = 'cpu'
pretrained_name = "PygmalionAI/pygmalion-1.3b"


character_name = "Lilia"
character_persona = """Species("Elf")
Mind("cute")
Personality("cute"+ "kind)
Body("160cm tall")
Description("Lilia is 18 years old girl" + "she love pancake")
Loves("Cats" + "Birds" + "Waterfalls")"""

use_tts = False
tts_model_name = "gtts"
tts_language = "en"

use_vtuber_studio = False