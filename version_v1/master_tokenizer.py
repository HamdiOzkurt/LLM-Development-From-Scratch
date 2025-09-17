import torch
import json
from transformers import PreTrainedTokenizerFast

class MasterTokenizer:
    def __init__(self, vocab):
        # Eğer vocab bir string ise (tokenizer dosya yolu), yükle
        if isinstance(vocab, str):
            self.hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=vocab)
            self.vocab = self.hf_tokenizer.get_vocab()
        else:
            # Eski vocab dict formatı için backward compatibility
            self.vocab = vocab
            self.hf_tokenizer = None
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_pretrained(cls, tokenizer_path):
        """
        Tokenizer dosyasından MasterTokenizer yükler
        """
        return cls(tokenizer_path)

    def encode(self, text):
        if self.hf_tokenizer:
            # Hugging Face tokenizer kullan
            tokens = self.hf_tokenizer.encode(text, add_special_tokens=False)
            return torch.tensor(tokens)
        else:
            # Eski implementasyon
            tokens = []
            for word in text.split():
                i = 0
                while i < len(word):
                    found_match = False
                    for j in range(len(word), i, -1):
                        sub_word = word[i:j]
                        if sub_word in self.vocab:
                            tokens.append(self.vocab[sub_word])
                            i = j
                            found_match = True
                            break
                    if not found_match:
                        tokens.append(self.vocab["<unk>"])
                        i += 1
                tokens.append(self.vocab[" "])
            
            tokens.pop()
            return torch.tensor(tokens)

    def tokenize(self, text):
        if self.hf_tokenizer:
            # Hugging Face tokenizer kullan
            return self.hf_tokenizer.tokenize(text)
        else:
            # Eski implementasyon
            token_ids = self.encode(text)
            token_ids = token_ids.detach().numpy().tolist()
            return [self.reverse_vocab[id] for id in token_ids]

    def decode(self, ids):
        if self.hf_tokenizer:
            # Hugging Face tokenizer kullan
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return self.hf_tokenizer.decode(ids)
        else:
            # Eski implementasyon
            text = ""
            for id in ids:
                text += self.reverse_vocab[id]
            return text
