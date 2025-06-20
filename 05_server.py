#
#
#
import torch
import models
import pickle
import dataset
import fastapi
import numpy as np


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
two = models.Towers(emb=300).to(dev)
two.load_state_dict(torch.load('./checkpoints/RNN_2025_06_20__11_18_00.4.300.two.pth'))


#
#
#
ds = dataset.Triplets(embedding_layer, words_to_ids)
db = torch.stack([two.doc(ds.to_emb(ds.docs[k]).to(dev)) for k in ds.d_keys])


#
#
#
app = fastapi.FastAPI()


#
#
#
@app.get("/search")
async def search(q: str):
  if q.strip() == '': return []
  qry = ds.to_emb(q)
  if qry is None: return []
  qry = two.qry(qry)
  res = torch.nn.functional.cosine_similarity(qry, db)
  top_scr, top_idx = torch.topk(res, k=4)
  return [{'score': s.item(), 'doc': ds.docs[ds.d_keys[i.item()]]} for s, i in zip(top_scr, top_idx)]
