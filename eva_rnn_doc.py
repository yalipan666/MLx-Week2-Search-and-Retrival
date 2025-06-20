import torch
import models_rnn_doc
import pickle
import numpy as np
import dataset_rnn
from torch.nn.utils.rnn import pad_sequence
import dataset

with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_matrix = np.load('./corpus/glove_embeddings.npy')
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32).to(dev)
embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

two = models_rnn_doc.Towers(emb=300).to(dev)
two.load_state_dict(torch.load('./checkpoints/RNN_doc_2025_06_20__14_08_26.4.300.two.pth'))  # Replace xxx with actual checkpoint

qry = torch.tensor([words_to_ids[w] for w in 'what animal bark'.split(' ')]).to(dev)
doc = torch.stack([torch.tensor([words_to_ids[w] for w in x.split(' ')]) for x in ['dog cute', 'cat meows', 'computers fast']]).to(dev)

qry_emb = embedding_layer(qry).mean(dim=0)  # (emb,)
doc_emb = embedding_layer(doc)  # (batch, seq_len, emb)

qry = two.qry(qry_emb)
doc = two.doc(doc_emb)

res = torch.nn.functional.cosine_similarity(qry, doc)
print(res)

print('Evaluating on test set...')
test_ds = dataset_rnn.Triplets(embedding_layer, words_to_ids, split='test')

def collate_rnn_doc(batch):
    qrys, poss, negs = zip(*batch)
    qrys = [q for q in qrys if q is not None]
    poss = [p for p in poss if p is not None]
    negs = [n for n in negs if n is not None]
    qrys = torch.stack(qrys)  # already avg pooled
    poss = pad_sequence(poss, batch_first=True)
    negs = pad_sequence(negs, batch_first=True)
    return qrys, poss, negs

train_ds = HybridTripletsDoc(embedding_layer, words_to_ids, split='train')
val_ds = HybridTripletsDoc(embedding_layer, words_to_ids, split='validation')
dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_rnn_doc)
dl_val = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_rnn_doc)

two.eval()
test_losses = []
with torch.no_grad():
    for qry, pos, neg in dl_test:
        qry, pos, neg = qry.to(dev), pos.to(dev), neg.to(dev)
        loss = two(qry, pos, neg, mrg=0.4)
        test_losses.append(loss.item())
print(f"Test Loss: {sum(test_losses)/len(test_losses):.4f}")
two.train()

class HybridTripletsDoc(torch.utils.data.Dataset):
    def __init__(self, embs, tkns, split='train'):
        self.qry_ds = dataset.Triplets(embs, tkns, split=split)
        self.doc_ds = dataset_rnn.Triplets(embs, tkns, split=split)
        assert len(self.qry_ds) == len(self.doc_ds)
    def __len__(self):
        return len(self.qry_ds)
    def __getitem__(self, idx):
        qry, _, _ = self.qry_ds[idx]
        _, pos, neg = self.doc_ds[idx]
        return qry, pos, neg 