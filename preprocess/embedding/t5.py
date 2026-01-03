import re
import torch
from einops import repeat
from transformers import T5Tokenizer, T5EncoderModel


class ProtT5Embedder():
    def __init__(self, name, model='Rostlab/prot_t5_xl_half_uniref50-enc', device='cuda:0'):
        self.name = name
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model, do_lower_case=False, use_safetensors=True)
        self.model = T5EncoderModel.from_pretrained(model, use_safetensors=True).to(device)
        self.model.requires_grad_(False)
        self.emb_channel = self.model.shared.embedding_dim
        
    @torch.no_grad()
    def get_embedding(self, seqs, tdevice):
        length_seq = max([len(seq) for seq in seqs])
        sequence = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in seqs]
        ids = self.tokenizer(sequence, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
        embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embedding_repr = (embedding_repr.last_hidden_state * attention_mask[..., None]).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        embedding_repr = repeat(embedding_repr.to(tdevice), 'b c -> b l c', l=length_seq)
        return embedding_repr