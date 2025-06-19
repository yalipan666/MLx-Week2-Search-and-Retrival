#
#
#
import torch


#
#
#
# The network is essentially two embedding tables, with dot products and a 
# sigmoid for binary classification
class SkipGram(torch.nn.Module):
  def __init__(self, voc, emb):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
    self.sig = torch.nn.Sigmoid()
    # sigmoid is needed here as the results are either 1(positive exampels) or 0 (negative exampels)

  def forward(self, inpt, trgs, rand):
    emb = self.emb(inpt)
    ctx = self.ffw.weight[trgs]
    rnd = self.ffw.weight[rand]
    # here we only use the weights rather than the ouput of the linear layer, whcih represents the
    # embeddings of the context/random words; if use the output of the linear layer, then we get
    # a vector of scores for all vocab, which is only needed for full softmax training (very expensive)  
    # why two embedding tabels (emb for center word, ffw.weight for context word)?
    # to learn different representations for a word depends on its role as center or context word 
    out = torch.bmm(ctx, emb.unsqueeze(-1)).squeeze()
    rnd = torch.bmm(rnd, emb.unsqueeze(-1)).squeeze()
    out = self.sig(out)
    rnd = self.sig(rnd)
    pst = -out.log().mean()
    ngt = -(1 - rnd + 10**(-3)).log().mean()
    return pst + ngt
  # the loss function here is the binary cross-entropy loss
  #  - (y * log(p) + (1 - y) * log(1 - p))
  # for pos eg., y = 1, then loss is -log(P) 
  # for neg eg., y = 0, then loss is - log(1-p), here we add 10**(-3) to aviod log(0) 

# dropout: since word2vec is a shallow NN, no need for dropout, also since there're 
# huge number of training examples, it's less of a concern of overfitting 
# dropout would help in deep and complex NN, e.g., a deep NN on top of embeddings
# 
# layer normalization: same as dropout, no need for simple embedding models as word2vec, 
# as it doesn't have non-linearities or deep layers

#
#
#
#
class QryTower(torch.nn.Module):
  def __init__(self, emb):
    super().__init__()
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(in_features=emb, out_features=emb),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=emb, out_features=emb)
    );

  def forward(self, avg):
    out = self.mlp(avg)
    return out


#
#
#
#
class DocTower(torch.nn.Module):
  def __init__(self, emb):
    super().__init__()
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(in_features=emb, out_features=emb),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=emb, out_features=emb)
    );

  def forward(self, avg):
    out = self.mlp(avg)
    return out


#
#
#
#
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


#
#
#
#
if __name__ == '__main__':
  # towers = Towers(emb=128, mrg=0.1)
  # qry = torch.randn(1, 128)
  # pos = torch.randn(5, 128)
  # neg = torch.randn(5, 128)
  # print(towers(qry, pos, neg))
  pass
