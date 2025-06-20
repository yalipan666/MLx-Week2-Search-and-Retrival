import torch
import models_rnn_qry
import pickle
import dataset_rnn
import dataset
import datetime
import wandb
import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(42)
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_matrix = np.load('./corpus/glove_embeddings.npy')
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32).to(dev)
embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

# Use dataset_rnn.Triplets for queries, dataset.Triplets for docs
class HybridTriplets(torch.utils.data.Dataset):
    def __init__(self, embs, tkns, split='train'):
        self.qry_ds = dataset_rnn.Triplets(embs, tkns, split=split)
        self.doc_ds = dataset.Triplets(embs, tkns, split=split)
        assert len(self.qry_ds) == len(self.doc_ds)
    def __len__(self):
        return len(self.qry_ds)
    def __getitem__(self, idx):
        qry, _, _ = self.qry_ds[idx]
        _, pos, neg = self.doc_ds[idx]
        return qry, pos, neg

train_ds = HybridTriplets(embedding_layer, words_to_ids, split='train')
val_ds = HybridTriplets(embedding_layer, words_to_ids, split='validation')

def collate_rnn_qry(batch):
    qrys, poss, negs = zip(*batch)
    qrys = [q for q in qrys if q is not None]
    poss = [p for p in poss if p is not None]
    negs = [n for n in negs if n is not None]
    qrys = pad_sequence(qrys, batch_first=True)  # (batch, max_qry_len, emb)
    poss = torch.stack(poss)
    negs = torch.stack(negs)
    return qrys, poss, negs

dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_rnn_qry)
dl_val = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_rnn_qry)

two = models_rnn_qry.Towers(emb=300).to(dev)
torch.save(two.state_dict(), f'./checkpoints/RNN_qry_{ts}.0.0.two.pth')
print('two:', sum(p.numel() for p in two.parameters()))
opt = torch.optim.Adam(two.parameters(), lr=0.003)
# wandb.init(project='mlx6-week-02-two')

for epoch in range(5):
    prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
    train_losses = []
    for idx, (qry, pos, neg) in enumerate(prgs):
        qry, pos, neg = qry.to(dev), pos.to(dev), neg.to(dev)
        loss = two(qry, pos, neg, mrg=0.4)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_losses.append(loss.item())
        if idx % 50 == 0:
            torch.save(two.state_dict(), f'./checkpoints/RNN_qry_{ts}.{epoch}.{idx}.two.pth')
    print(f"Epoch {epoch+1} - Train Loss: {sum(train_losses)/len(train_losses):.4f}")

    # Validation
    two.eval()
    val_losses = []
    with torch.no_grad():
        for qry, pos, neg in dl_val:
            qry, pos, neg = qry.to(dev), pos.to(dev), neg.to(dev)
            loss = two(qry, pos, neg, mrg=0.4)
            val_losses.append(loss.item())
    print(f"Epoch {epoch+1} - Validation Loss: {sum(val_losses)/len(val_losses):.4f}")
    two.train()

# wandb.finish() 