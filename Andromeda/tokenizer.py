from transformers import AutoTokenizer

class AndromedaTokenizer:
    def __init__(self):
        self.tokenizer_checkpoint = 'EleutherAI/gpt-neox-20b'
        self.tokenizer            = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint)

        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode(self, text):
        idxs = self.tokenizer.encode(text)
        
        return idxs
    
    def decode(self, idxs):
        text = self.tokenizer.decode(idxs)
        
        return text
