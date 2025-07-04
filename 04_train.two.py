#
#
#
import torch
import models
import pickle
import dataset
import datetime
import wandb
import tqdm
import numpy as np


#
#
#
torch.manual_seed(42)
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


#
#
#
with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load GloVe embedding matrix
embedding_matrix = np.load('./corpus/glove_embeddings.npy')
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32).to(dev)
embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)


#
#
#
train_ds = dataset.Triplets(embedding_layer, words_to_ids, split='train')
val_ds = dataset.Triplets(embedding_layer, words_to_ids, split='validation')
dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
dl_val = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)


#
#
#
two = models.Towers(emb=300).to(dev)
torch.save(two.state_dict(), f'./checkpoints/{ts}.0.0.two.pth')
print('two:', sum(p.numel() for p in two.parameters())) # 66,048
opt = torch.optim.Adam(two.parameters(), lr=0.003)
# wandb.init(project='mlx6-week-02-two')


#
#
#
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
    if idx % 50 == 0: torch.save(two.state_dict(), f'./checkpoints/{ts}.{epoch}.{idx}.two.pth')
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


#
#
#
# wandb.finish()