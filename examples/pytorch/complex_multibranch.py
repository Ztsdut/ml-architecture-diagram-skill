"""Synthetic multi-input stress case used by parser/publication regression tests."""
from __future__ import annotations
import torch
from torch import nn


class ComplexFusionNet(nn.Module):
    def __init__(self, *, use_aux: bool = True, use_gate: bool = True):
        super().__init__()
        self.use_aux = use_aux
        self.use_gate = use_gate
        self.map_encoder = nn.Linear(16, 32)
        self.history_encoder = nn.Sequential(nn.Linear(12, 32), nn.LayerNorm(32), nn.SiLU())
        self.query_encoder = nn.Sequential(nn.Linear(9, 32), nn.LayerNorm(32), nn.SiLU())
        self.spatial_attention = nn.MultiheadAttention(32, 4, batch_first=True)
        self.history_attention = nn.MultiheadAttention(32, 4, batch_first=True)
        self.context_gate = nn.Sequential(nn.Linear(69, 32), nn.SiLU(), nn.Linear(32, 1))
        self.main_head = nn.Sequential(nn.Linear(97, 32), nn.SiLU(), nn.Linear(32, 1))
        self.aux_head = nn.Sequential(nn.Linear(65, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, map_features, history, future_geometry, environment):
        map_tokens = self.map_encoder(map_features)
        history_tokens = self.history_encoder(history)
        query = self.query_encoder(torch.cat((future_geometry, environment), dim=-1))
        spatial, weights = self.spatial_attention(query, map_tokens, map_tokens, need_weights=True)
        temporal, _ = self.history_attention(query, history_tokens, history_tokens, need_weights=False)
        gate_features = torch.cat((spatial, temporal, environment), dim=-1)
        if self.use_gate:
            gate = torch.sigmoid(self.context_gate(gate_features))
        else:
            gate = spatial.new_ones(spatial.shape[0], spatial.shape[1], 1)
        main_features = torch.cat((spatial, temporal, query, environment, gate), dim=-1)
        main_logits = self.main_head(main_features).squeeze(-1)
        aux_features = torch.cat((spatial, query, environment, gate), dim=-1)
        aux_logits = self.aux_head(aux_features).squeeze(-1)
        if not self.use_aux:
            aux_logits = aux_logits.detach() * 0.0
        return {"main_logits": main_logits, "aux_logits": aux_logits, "gate": gate.squeeze(-1), "attention_weights": weights}
