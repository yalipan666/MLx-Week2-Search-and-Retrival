import torch

class QryTower(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.rnn = torch.nn.GRU(input_size=emb, hidden_size=emb, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb, out_features=emb),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=emb, out_features=emb)
        )

    def forward(self, seq):
        # seq: (seq_len, emb) or (batch, seq_len, emb)
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)  # (1, seq_len, emb)
        _, h = self.rnn(seq)
        out = self.mlp(h.squeeze(0))
        return out

class DocTower(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.rnn = torch.nn.GRU(input_size=emb, hidden_size=emb, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb, out_features=emb),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=emb, out_features=emb)
        )

    def forward(self, seq):
        # seq: (batch, seq_len, emb)
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)  # (1, seq_len, emb)
        _, h = self.rnn(seq)
        out = self.mlp(h.squeeze(0))
        return out

class Towers(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.qry = QryTower(emb)
        self.doc = DocTower(emb)

    def forward(self, qry, pos, neg, mrg):
        qry = self.qry(qry)
        pos = self.doc(pos)
        neg = self.doc(neg)
        pos = 1 - torch.nn.functional.cosine_similarity(qry, pos)
        neg = 1 - torch.nn.functional.cosine_similarity(qry, neg)
        return torch.max(pos - neg + mrg, torch.tensor(0.0)).mean() 