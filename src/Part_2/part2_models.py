"""
Part 2.2 — BiLSTM tagger + linear-chain CRF (PyTorch from scratch).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

_P1 = str(Path(__file__).resolve().parent.parent / "Part_1")
if _P1 not in sys.path:
    sys.path.insert(0, _P1)

from part1_vocab_tfidf import LSTM_DROPOUT, LSTM_HIDDEN, LSTM_LAYERS, VOCAB_SIZE, W2V_DIM


class CRFLayer(nn.Module):
    """Linear-chain CRF (log-space forward partition + Viterbi decode)."""

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Input:  (B, T, C) emissions, (B, T) gold tags, (B, T) mask
        # Output: scalar mean negative log-likelihood
        losses = []
        for b in range(emissions.size(0)):
            m = mask[b].bool()
            e = emissions[b][m]
            t = tags[b][m]
            if e.size(0) == 0:
                continue
            log_z = self._forward_log_partition(e)
            gold = self._gold_score(e, t)
            losses.append(-(gold - log_z))
        if not losses:
            return torch.zeros((), device=emissions.device, requires_grad=True)
        return torch.stack(losses).mean()

    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        T, C = emissions.shape
        s = self.start_transitions[tags[0]] + emissions[0, tags[0]]
        for t in range(1, T):
            s = s + self.transitions[tags[t - 1], tags[t]] + emissions[t, tags[t]]
        return s + self.end_transitions[tags[T - 1]]

    def _forward_log_partition(self, emissions: torch.Tensor) -> torch.Tensor:
        T, C = emissions.shape
        alpha = self.start_transitions + emissions[0]
        for t in range(1, T):
            scores = alpha.unsqueeze(1) + self.transitions
            alpha = torch.logsumexp(scores, dim=0) + emissions[t]
        return torch.logsumexp(alpha + self.end_transitions, dim=0)

    def viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = emissions.shape
        out = torch.zeros(B, T, dtype=torch.long, device=emissions.device)
        for b in range(B):
            m = mask[b].bool()
            e = emissions[b][m]
            if e.size(0) == 0:
                continue
            path = self._viterbi_path(e)
            out[b, : path.size(0)] = path
        return out

    def _viterbi_path(self, emissions: torch.Tensor) -> torch.Tensor:
        T, C = emissions.shape
        cur = self.start_transitions + emissions[0]
        back = torch.zeros(T, C, dtype=torch.long, device=emissions.device)
        for t in range(1, T):
            mat = cur.unsqueeze(1) + self.transitions
            best_score, best_idx = mat.max(dim=0)
            cur = best_score + emissions[t]
            back[t] = best_idx
        cur = cur + self.end_transitions
        best_last = int(cur.argmax().item())
        tags = [best_last]
        for t in range(T - 1, 0, -1):
            tags.append(int(back[t, tags[-1]].item()))
        tags.reverse()
        return torch.tensor(tags, dtype=torch.long, device=emissions.device)


def build_bilstm_tagger(
    num_labels: int,
    weights_np: np.ndarray | None,
    *,
    freeze: bool = True,
    bidirectional: bool = True,
    use_dropout: bool = True,
    random_embedding: bool = False,
    use_crf: bool = False,
) -> "BiLSTMTagger":
    return BiLSTMTagger(
        num_labels,
        weights_np,
        freeze=freeze,
        bidirectional=bidirectional,
        use_dropout=use_dropout,
        random_embedding=random_embedding,
        use_crf=use_crf,
    )


class BiLSTMTagger(nn.Module):
    """BiLSTM + emission layer; NER uses CRF when use_crf=True."""

    def __init__(
        self,
        num_labels: int,
        weights_np: np.ndarray | None,
        freeze: bool,
        bidirectional: bool = True,
        use_dropout: bool = True,
        random_embedding: bool = False,
        use_crf: bool = False,
    ) -> None:
        super().__init__()
        hidden_half = LSTM_HIDDEN // 2
        lstm_inter_dropout = (LSTM_DROPOUT if LSTM_LAYERS > 1 and use_dropout else 0.0)

        if random_embedding:
            self.embedding = nn.Embedding(VOCAB_SIZE, W2V_DIM, padding_idx=0)
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            self.embedding.weight.requires_grad = True
        else:
            assert weights_np is not None
            w = torch.tensor(weights_np, dtype=torch.float32)
            self.embedding = nn.Embedding.from_pretrained(w, freeze=freeze, padding_idx=0)

        self.bilstm = nn.LSTM(
            W2V_DIM,
            hidden_half,
            num_layers=LSTM_LAYERS,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=lstm_inter_dropout,
        )
        lstm_out_dim = LSTM_HIDDEN if bidirectional else hidden_half
        self.dropout = nn.Dropout(LSTM_DROPOUT) if use_dropout else nn.Identity()
        self.emission = nn.Linear(lstm_out_dim, num_labels)
        self.use_crf = use_crf
        self.crf: CRFLayer | None = CRFLayer(num_labels) if use_crf else None

    def encode(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Token ids → emission logits (B, T, num_labels)."""
        x = self.embedding(input_ids)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        return self.emission(out)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        tags: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        # Input:  (B, T) token ids, (B,) lengths; optional tags/mask for CRF
        # Output: (B, T, L) logits if no CRF; else scalar loss (train) or (B,T) tags (decode)
        emissions = self.encode(input_ids, lengths)
        if self.crf is not None:
            assert mask is not None
            if tags is not None:
                return self.crf(emissions, tags, mask)
            return self.crf.viterbi_decode(emissions, mask)
        return emissions
