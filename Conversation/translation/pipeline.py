from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print("Loading machine translation model...")

class Translate:
    def __init__(self, device, language=None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
        if language is None:
            print('Please Select Output Language code according to BCP47')
            print('See HERE: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200')
            language = input("Translate To?: ")

        self.language = language
        self.device = device

    def translate(self, text:str) -> str:
        inputs = self.tokenizer([text], return_tensors='pt')
        inputs = inputs.to(self.device)
        outs = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.language])
        translated = self.tokenizer.batch_decode(outs, skip_special_tokens=True)[0]
        return translated