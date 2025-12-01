import torch
from einops import repeat


class ESMEmbedder():
    def __init__(self, name, model='esm2_t33_650M_UR50D', device='cuda:0'):
        self.name = name
        self.device = device
        self.model, self.alphabet = torch.hub.load('facebookresearch/esm:main', model)
        self.model = self.model.to(device)
        self.model.requires_grad_(False)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.emb_channel = self.model.embed_tokens.embedding_dim
        
    @torch.no_grad()
    def get_embedding(self, seqs, tdevice):
        length_seq = max([len(seq) for seq in seqs])
        data = [(f"protein{i}", seq) for i, seq in enumerate(seqs)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        results = self.model(batch_tokens.to(self.device), repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33].to(tdevice)
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
        embedding_repr = torch.stack(sequence_representations, dim=0)
        embedding_repr = repeat(embedding_repr.to(tdevice), 'b c -> b l c', l=length_seq)
        return embedding_repr