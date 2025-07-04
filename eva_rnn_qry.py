import torch
import models_rnn_qry
import pickle
import numpy as np
import dataset_rnn
import dataset
from torch.nn.utils.rnn import pad_sequence

with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_matrix = np.load('./corpus/glove_embeddings.npy')
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32).to(dev)
embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

two = models_rnn_qry.Towers(emb=300).to(dev)
two.load_state_dict(torch.load('./checkpoints/RNN_qry_2025_06_20__12_42_51.4.300.two.pth'))  # Replace xxx with actual checkpoint

qry = torch.tensor([words_to_ids[w] for w in 'what animal bark'.split(' ')]).to(dev)
doc = torch.stack([torch.tensor([words_to_ids[w] for w in x.split(' ')]) for x in ['dog cute', 'cat meows', 'computers fast']]).to(dev)

qry_emb = embedding_layer(qry)  # (seq_len, emb)
doc_emb = embedding_layer(doc).mean(dim=1)  # (batch, emb)

qry = two.qry(qry_emb)
doc = two.doc(doc_emb)

res = torch.nn.functional.cosine_similarity(qry, doc)
print(res)

print('Evaluating on test set...')

def collate_rnn_qry(batch):
    qrys, poss, negs = zip(*batch)
    qrys = [q for q in qrys if q is not None]
    poss = [p for p in poss if p is not None]
    negs = [n for n in negs if n is not None]
    qrys = pad_sequence(qrys, batch_first=True)
    poss = torch.stack([p.mean(dim=0) for p in poss])
    negs = torch.stack([n.mean(dim=0) for n in negs])
    return qrys, poss, negs

test_ds = dataset_rnn.Triplets(embedding_layer, words_to_ids, split='test')
dl_test = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_rnn_qry)
two.eval()
test_losses = []
with torch.no_grad():
    for qry, pos, neg in dl_test:
        qry, pos, neg = qry.to(dev), pos.to(dev), neg.to(dev)
        loss = two(qry, pos, neg, mrg=0.4)
        test_losses.append(loss.item())
print(f"Test Loss: {sum(test_losses)/len(test_losses):.4f}")
two.train() 