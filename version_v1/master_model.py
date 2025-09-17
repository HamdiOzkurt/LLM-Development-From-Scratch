import torch
import torch.nn as nn

from .master_decoder_block import MasterDecoderBlock as UstaDecoderBlock
from .master_embedding import MasterEmbedding as UstaEmbedding


class MasterModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, context_length, num_layers):
        super().__init__()

        self.embedding = UstaEmbedding(vocab_size, embedding_dim, context_length)
        self.layers = nn.Sequential(
            *[UstaDecoderBlock(embedding_dim, num_heads, context_length) for _ in range(num_layers)]
        )

        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x) # dictionary meaning of the tokens (words)
        
        x = self.layers(x)
        x = self.lm_head(x)

        return x

    def generate(self, x, max_new_tokens=20, temperature=0.8, top_k=50, do_sample=True):
        tokens = x.detach().cpu().numpy().tolist() if isinstance(x, torch.Tensor) else x
        
        for _ in range(max_new_tokens):
            # Input'u tensor'e çevir
            input_tensor = torch.tensor([tokens],dtype=torch.long)
            
            # Forward pass
            with torch.no_grad():
                out = self.forward(input_tensor)
                logits = out[0, -1, :]  # Son token'in logitleri
            
            if do_sample:
                # Temperature scaling
                if temperature > 0:
                    logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    indices_to_remove = logits < top_k_logits[-1]
                    logits[indices_to_remove] = -float('inf')
                
                # Softmax ve sampling
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            
            tokens.append(next_token.item())
            
            # EOS token veya max length kontrolü
            if next_token.item() == 59 or len(tokens) > 32:
                break
        
        return tokens
