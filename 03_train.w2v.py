#
#
#
import torch
import models
import dataset
import datetime
import pickle
# import wandb
import tqdm
import os
import requests
import zipfile
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


#
#
#
# --- GloVe Download and Extraction ---
glove_dir = './corpus/glove/'
glove_file = os.path.join(glove_dir, 'glove.6B.300d.txt')
glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
glove_zip = os.path.join(glove_dir, 'glove.6B.zip')
emb_dim = 300

if not os.path.exists(glove_file):
    os.makedirs(glove_dir, exist_ok=True)
    print('Downloading GloVe embeddings...')
    r = requests.get(glove_url, stream=True)
    with open(glove_zip, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print('Extracting GloVe embeddings...')
    with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
        zip_ref.extract('glove.6B.300d.txt', glove_dir)
    os.remove(glove_zip)


#
#
#
# --- Load GloVe Embeddings ---
glove_embeddings = {}
with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vec = np.array(parts[1:], dtype=np.float32)
        glove_embeddings[word] = vec


#
#
#
# --- Build Embedding Matrix ---
vocab_size = len(words_to_ids)
embedding_matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)
unk_count = 0
for word, idx in words_to_ids.items():
    if word in glove_embeddings:
        embedding_matrix[idx] = glove_embeddings[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(emb_dim,))
        unk_count += 1
print(f"Words not found in GloVe: {unk_count} / {vocab_size}")

# Save embedding matrix for downstream use
np.save('./corpus/glove_embeddings.npy', embedding_matrix)
print(f"Saved GloVe embedding matrix to ./corpus/glove_embeddings.npy")

# Save in word2vec-style text format
with open('./corpus/glove_embeddings.txt', 'w', encoding='utf-8') as f:
    for word, idx in words_to_ids.items():
        vec = embedding_matrix[idx]
        vec_str = ' '.join(map(str, vec))
        f.write(f'{word} {vec_str}\n')
print(f"Saved GloVe embedding matrix to ./corpus/glove_embeddings.txt")


#
#
#
# --- COMMENTED OUT: Word2Vec Training ---
'''
#Model: Creates a SkipGram (Word2Vec) model with vocabulary size and 128-dimensional embeddings.
w2v = models.SkipGram(voc=len(words_to_ids), emb=128).to(dev)
torch.save(w2v.state_dict(), f'./checkpoints/{ts}.0.w2v.pth')
print('w2v:', sum(p.numel() for p in w2v.parameters())) # 35,998,976
opt = torch.optim.Adam(w2v.parameters(), lr=0.003)
# wandb.init(project='mlx6-week-02-mrc')

# prepare dataset and dataloader
ds = dataset.Window('./corpus/tokens.txt')
dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)

for epoch in range(5):
  prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
  for idx, (inpt, trgs) in enumerate(prgs):
    inpt, trgs = inpt.to(dev), trgs.to(dev)
    rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2)).to(dev)
    opt.zero_grad()
    loss = w2v(inpt, trgs, rand)
    loss.backward()
    opt.step()
    # wandb.log({'loss': loss.item()})
    if idx % 10_000 == 0: torch.save(w2v.state_dict(), f'./checkpoints/{ts}.{epoch}.{idx}.w2v.pth')
'''
#
#
#
# wandb.finish()
