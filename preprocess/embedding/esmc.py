import torch
from einops import repeat
from transformers import AutoModel, AutoTokenizer


# pip install esm@git+https://github.com/Biohub/esm.git@main --ignore-requires-python


class ESMCEmbedder:

    def __init__(self, base_model="esmc_h", device="cuda:0"):  # device 파라미터 추가
        nickname_map = {"esmc_h": "ESMC-6B", "esmc_m": "ESMC-600M", "esmc_s": "ESMC-300M"}
        assert base_model in nickname_map, f"Invalid base_model: {base_model}. Choose from {list(nickname_map.keys())}."
        self.model_size = nickname_map[base_model]
        channel_map = {"ESMC-300M": 960, "ESMC-600M": 1152, "ESMC-6B": 2560}
        self.emb_channel = channel_map[self.model_size]

        model_repo = f"biohub/{self.model_size}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(
            model_repo, trust_remote_code=True
        ).eval()

        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        self.model.requires_grad_(False)

    @torch.inference_mode()
    def get_embedding(self, seqs, tdevice):
        inputs = self.tokenizer(seqs, return_tensors="pt", padding=True, truncation=False)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        last_hidden = output.hidden_states[-1].to(tdevice)

        sequence_representations = []
        for i, mask in enumerate(attention_mask):
            valid_indices = torch.where(mask == 1)[0]
            start_idx = valid_indices[0].item() + 1
            end_idx = valid_indices[-1].item()

            seq_emb = last_hidden[i, start_idx:end_idx].mean(0)
            sequence_representations.append(seq_emb)

        embedding_repr = torch.stack(sequence_representations, dim=0)
        max_length_seq = max([len(seq) for seq in seqs])
        embedding_repr = repeat(embedding_repr, "b c -> b l c", l=max_length_seq)

        return embedding_repr
