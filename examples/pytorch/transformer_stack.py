import torch.nn as nn


class EncoderModel(nn.Module):
    def __init__(self, vocab=32000, d_model=256, depth=6, classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
            for _ in range(6)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, classes)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
