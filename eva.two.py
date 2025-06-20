#
#
#
import torch
import models
import pickle
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
two = models.Towers(emb=50).to(dev)
two.load_state_dict(torch.load('./checkpoints/2025_06_19__22_23_10.0.150.two.pth'))


#
#
#
qry = torch.tensor([words_to_ids[w] for w in 'what animal bark'.split(' ')]).to(dev)
doc = torch.stack([torch.tensor([words_to_ids[w] for w in x.split(' ')]) for x in ['dog cute', 'cat meows', 'computers fast']]).to(dev)


#
#
#
qry = two.qry(embedding_layer(qry).mean(dim=0))
doc = two.doc(embedding_layer(doc).mean(dim=1))


#
#
#
res = torch.nn.functional.cosine_similarity(qry, doc)
print(res)
